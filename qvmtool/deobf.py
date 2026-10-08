"""Deobfuscation pass: resolve the engine's "hidden direct branch" idioms.

The mutation engine does not emit plain `jmp rel32` inside the VM payload. It hides
a *constant* target behind one of several instruction shapes, so a naive scanner
sees an indirect jump and gives up. Every shape below resolves to a compile-time
constant, which is what makes the control-flow graph recoverable at all -- and it is
why this engine is best described as *threaded control flow with obfuscated targets*
rather than *table dispatch*.

Forms handled (all anchored on a register-indirect `jmp`, which needs no decode to
locate, plus a bounded decode on a real instruction boundary):

  A  "selfpc"   call $+5 ; <junk> ; pop reg ; add reg, imm ; jmp reg  -> p+5+imm
  C  "riplea"   lea reg,[rip+d] ; ... ; jmp reg                       -> end(lea)+d
  D  "imm64"    mov reg, imm64 ; ... ; jmp reg                        -> imm64

Form A carries no control flow at all: `imm` is chosen so that `p+5+imm` is the
instruction immediately after the `jmp`, making the whole construct a semantically
null branch that exists purely to break disassembly. Such sites are flagged
`fallthrough` and must not be counted as CFG edges.

A bare `pop reg ; jmp reg` (no observed shift) is deliberately left *unresolved*:
its target is a stack-resident value, so claiming a fixed target would be wrong.

Form A is anchored forward from `call $+5` (itself a guaranteed instruction
boundary). Forms C/D are anchored backward from the jmp site, decoding a bounded
window that starts at a byte which is *not* guaranteed to be an instruction
boundary -- so C/D results are reported with a confidence marker and a caller may
filter them out.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .pe import PEMap

CALL_SELF = b"\xe8\x00\x00\x00\x00"

JMP_PATTERNS = [bytes([0xFF, 0xE0 + i]) for i in range(8)] + \
               [bytes([0x41, 0xFF, 0xE0 + i]) for i in range(8)]

#: `lea reg,[rip+disp32]` = REX.W 8D <modrm=mod00 reg rm=101> disp32
#: modrm = 0x05 | (reg << 3)  for rax..rdi ; with REX.R for r8..r15
LEA_RIP = []
for _reg in range(8):
    LEA_RIP.append(bytes([0x48, 0x8D, 0x05 | (_reg << 3)]))
    LEA_RIP.append(bytes([0x4C, 0x8D, 0x05 | (_reg << 3)]))

#: `mov reg, imm64` = REX.W B8+reg imm64
MOV_IMM64 = []
for _reg in range(8):
    MOV_IMM64.append(bytes([0x48, 0xB8 + _reg]))
    MOV_IMM64.append(bytes([0x49, 0xB8 + _reg]))

BACKWARD_WINDOW = 48          # bytes to look back from a jmp site
MAX_LEA_GAP = 24              # a lea older than this is not a plausible predecessor
MAX_IMM_GAP = 20


@dataclass
class HiddenBranch:
    jmp_site: int           # VA of the `jmp reg`
    register: str
    target: int             # resolved VA
    form: str               # selfpc | riplea | imm64
    anchor: int             # VA of the instruction that materialised the target
    confidence: str         # high (forward-anchored) | medium (backward-anchored)
    delta: int = 0
    jmp_len: int = 0
    fallthrough: bool = False   # target == the instruction after the jmp
    shift_site: int = 0         # VA of the `add reg,imm` that applies `delta`
    shift_len: int = 0
    pop_site: int = 0           # VA of the `pop reg` that consumes the pushed PC

    @property
    def patch_span(self) -> tuple[int, int]:
        """The smallest byte range that implements the branch decision.

        Rewriting the *whole* idiom (`call $+5` .. `jmp reg`) would delete live
        instructions: everything between the `call` and the `pop` really executes.
        The only part that decides control flow is the shift-plus-jump tail, so that
        is what a de-obfuscator should replace -- leaving the push, the junk and the
        `pop` intact keeps the stack balanced and the side effects present.
        """
        start = self.shift_site or self.pop_site or self.anchor
        return start, self.jmp_site + self.jmp_len

    def to_dict(self) -> dict:
        return {
            "jmp_site": hex(self.jmp_site), "register": self.register,
            "target": hex(self.target), "form": self.form,
            "anchor": hex(self.anchor), "confidence": self.confidence,
            "delta": self.delta, "fallthrough": self.fallthrough,
        }


@dataclass
class DeobfResult:
    branches: list[HiddenBranch] = field(default_factory=list)
    jmp_reg_sites: int = 0
    unresolved: int = 0
    internal_edges: int = 0

    @property
    def resolved(self) -> int:
        return len(self.branches)

    def by_form(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.branches:
            out[b.form] = out.get(b.form, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def targets(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for b in self.branches:
            counts[b.target] = counts.get(b.target, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        high = sum(1 for b in self.branches if b.confidence == "high")
        ft = sum(1 for b in self.branches if b.fallthrough)
        return {
            "jmp_reg_sites": self.jmp_reg_sites,
            "resolved_branches": self.resolved,
            "resolution_rate": round(self.resolved / self.jmp_reg_sites, 4)
                               if self.jmp_reg_sites else 0.0,
            "forms": self.by_form(),
            "high_confidence": high,
            "medium_confidence": self.resolved - high,
            "fallthrough_only": ft,
            "real_edges": self.resolved - ft,
            "unresolved_sites": self.unresolved,
            "internal_edges": self.internal_edges,
            "distinct_targets": len(self.targets()),
        }


def _accumulate_shift(buf: bytes, buf_va: int, start: int, end: int, reg: str, md
                      ) -> tuple[int, bool]:
    """Net constant applied to `reg` over [start, end); returns (delta, trustworthy).

    Why this exists: the backward `imm64` form used to report the raw immediate as the
    target, ignoring the flags-neutral `add reg, imm32` that the engine puts between the
    `mov` and the `jmp`. It therefore reported the **pre-shift** base -- a wrong target
    that still lands inside the VM section, so an in-section sanity check passes and the
    error stays invisible. A delegated validation against executed ground truth found
    exactly that failure mode on the sibling engine (0 of 213 resolved sites correct).

    The second return value matters as much as the first: if anything else writes the
    register inside the window the model is void, and reporting a target anyway would be
    a guess. Callers must treat `ok=False` as unresolved.
    """
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG, X86_REG_INVALID, X86_REG_RIZ

    #: mnemonics that read their operands without writing the first one
    NON_WRITING = ("cmp", "test", "push", "call", "jmp", "bt", "nop", "ret")

    delta = 0
    for ins in md.disasm(buf[start:end], buf_va + start):
        ops = ins.operands
        if ins.mnemonic in ("add", "sub") and len(ops) == 2 \
                and ops[0].type == X86_OP_REG \
                and ins.reg_name(ops[0].reg) == reg \
                and ops[1].type == X86_OP_IMM:
            step = ops[1].imm
            delta += step if ins.mnemonic == "add" else -step
            continue
        if ins.mnemonic == "lea" and len(ops) == 2 \
                and ops[0].type == X86_OP_REG \
                and ins.reg_name(ops[0].reg) == reg \
                and ops[1].type == X86_OP_MEM \
                and ops[1].mem.base \
                and ins.reg_name(ops[1].mem.base) == reg \
                and ops[1].mem.index in (X86_REG_INVALID, X86_REG_RIZ):
            delta += ops[1].mem.disp
            continue
        writes = ops and ops[0].type == X86_OP_REG \
            and ins.reg_name(ops[0].reg) == reg
        if writes and ins.mnemonic not in NON_WRITING:
            return delta, False
    return delta, True


def _jmp_sites(buf: bytes) -> list[tuple[int, str, int]]:
    """(offset, register-name, opcode-kind) for every register-indirect jmp.

    Byte-pattern based, therefore immune to disassembler desynchronisation.
    Overlapping matches at +1 offsets (a `FF E0` inside a `41 FF E0`) are collapsed
    by keeping the match that starts earliest.

    NOTE: this covers only the **register** form. Use `indirect_sites` when the
    memory-indirect forms matter -- measured on the reference sample, this pattern
    finds 8,964 sites while the full population is 28,382.
    """
    regs = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
            "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
    found: dict[int, tuple[str, int]] = {}
    for i in range(8):
        for pat, base_idx, kind in ((bytes([0xFF, 0xE0 + i]), i, 0),
                                    (bytes([0x41, 0xFF, 0xE0 + i]), 8 + i, 1)):
            start = 0
            while True:
                k = buf.find(pat, start)
                if k < 0:
                    break
                # collapse a no-REX match that is actually the tail of a REX match
                if not (kind == 0 and k > 0 and buf[k - 1] == 0x41):
                    found.setdefault(k, (regs[base_idx], kind))
                start = k + 1
    return [(off, reg, kind) for off, (reg, kind) in sorted(found.items())]


def indirect_sites(buf: bytes) -> dict[str, list[int]]:
    """Census of *all* indirect call/jmp encodings, register AND memory.

    `call r/m64` is `FF /2` and `jmp r/m64` is `FF /4`; the ModRM *reg* field picks
    which. `_jmp_sites` only recognises mod=11 (the register forms), which on the
    reference sample enumerates 8,964 sites out of a true population of 28,382 --
    the missing 19,418 are memory-indirect forms (`jmp [rip+d]`, `call [reg+8]`,
    `jmp [reg+idx*8]`). Those are exactly the shapes a dispatch table or a loaded
    function pointer uses, so enumerating only the register form silently excludes
    most of the interesting control flow.

    Byte-wise, so it is immune to disassembler desynchronisation.

    Returns offsets *within* `buf`, keyed `jmp_reg` / `jmp_mem` / `call_reg` /
    `call_mem`.
    """
    out: dict[str, list[int]] = {"jmp_reg": [], "jmp_mem": [],
                                 "call_reg": [], "call_mem": []}
    n = len(buf)
    i = 0
    while i < n - 1:
        p = i
        if 0x40 <= buf[p] <= 0x4F and p + 2 < n and buf[p + 1] == 0xFF:
            p += 1
        elif buf[p] == 0xFF and p > 0 and 0x40 <= buf[p - 1] <= 0x4F:
            # This FF is the tail of a REX-prefixed form already reported at p-1.
            # Without this the same instruction is counted twice (e.g. `41 FF E1`
            # would appear as both `41 FF E1` and `FF E1`), inflating the census.
            i += 1
            continue
        if buf[p] == 0xFF and p + 1 < n:
            modrm = buf[p + 1]
            reg = (modrm >> 3) & 7
            mod = (modrm >> 6) & 3
            if reg == 4:
                out["jmp_reg" if mod == 3 else "jmp_mem"].append(i)
            elif reg == 2:
                out["call_reg" if mod == 3 else "call_mem"].append(i)
        i += 1
    return out


def resolve_hidden_branches(buf: bytes, buf_va: int, vm_va_lo: int, vm_va_hi: int,
                            forms: tuple[str, ...] = ("selfpc", "selfpc0",
                                                      "riplea", "imm64")
                            ) -> DeobfResult:
    """Resolve obfuscated direct branches in `buf` (all addresses in VA space)."""
    from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    from capstone.x86 import (X86_OP_IMM, X86_OP_MEM, X86_OP_REG, X86_REG_INVALID,
                              X86_REG_RIZ)

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    res = DeobfResult()
    sites = _jmp_sites(buf)
    res.jmp_reg_sites = len(sites)
    #: sites the forward pass already reached a verdict on (resolved OR deliberately
    #: rejected), so the backward pass cannot double-count them as "unresolved".
    handled_sites: set[int] = set()

    # ---------- forward-anchored form: selfpc ----------
    # The real shape, confirmed by byte inspection, is
    #     call $+5 ; <junk> ; pop reg ; add reg, imm ; jmp reg
    # The shift is applied with `add reg, imm32` (49 81 c0 ..), NOT with `lea`.
    # A first cut only modelled `lea`, so every site silently kept delta = 0 and
    # resolved to `p+5` -- the address of the `pop`, i.e. an apparent infinite loop.
    # That was wrong on ~1,900 sites and is exactly the kind of failure this
    # forward-anchored scan is supposed to make impossible.
    if "selfpc" in forms or "selfpc0" in forms:
        start = 0
        while True:
            p = buf.find(CALL_SELF, start)
            if p < 0:
                break
            start = p + 1
            chunk = buf[p:p + 64]
            if len(chunk) < 8:
                continue
            call_va = buf_va + p
            reg = None
            delta = 0
            jmp_va = None
            jmp_len = 0
            shift_site = 0
            shift_len = 0
            pop_site = 0
            for n, ins in enumerate(md.disasm(chunk[5:], call_va + 5)):
                if n > 12:
                    break
                if ins.mnemonic == "pop" and ins.operands \
                        and ins.operands[0].type == X86_OP_REG:
                    if reg is None:
                        reg = ins.reg_name(ins.operands[0].reg)
                        pop_site = ins.address
                    continue
                # `add reg, imm` / `sub reg, imm` on the tracked register
                if ins.mnemonic in ("add", "sub") and reg is not None \
                        and len(ins.operands) == 2 \
                        and ins.operands[0].type == X86_OP_REG \
                        and ins.reg_name(ins.operands[0].reg) == reg \
                        and ins.operands[1].type == X86_OP_IMM:
                    step = ins.operands[1].imm
                    delta += step if ins.mnemonic == "add" else -step
                    if shift_site == 0:
                        shift_site = ins.address
                        shift_len = ins.size
                    else:
                        shift_len = (ins.address + ins.size) - shift_site
                    continue
                if ins.mnemonic == "lea" and len(ins.operands) == 2 \
                        and ins.operands[0].type == X86_OP_REG \
                        and ins.operands[1].type == X86_OP_MEM \
                        and reg is not None \
                        and ins.reg_name(ins.operands[0].reg) == reg \
                        and ins.operands[1].mem.base \
                        and ins.reg_name(ins.operands[1].mem.base) == reg \
                        and ins.operands[1].mem.index in (X86_REG_INVALID, X86_REG_RIZ):
                    delta += ins.operands[1].mem.disp
                    if shift_site == 0:
                        shift_site = ins.address
                        shift_len = ins.size
                    else:
                        shift_len = (ins.address + ins.size) - shift_site
                    continue
                if ins.mnemonic == "jmp" and ins.operands \
                        and ins.operands[0].type == X86_OP_REG \
                        and reg is not None \
                        and ins.reg_name(ins.operands[0].reg) == reg:
                    jmp_va = ins.address
                    jmp_len = ins.size
                    break
                if ins.mnemonic in ("jmp", "call", "ret"):
                    break
            if jmp_va is None or reg is None:
                continue
            # A raw `pop reg; jmp reg` with no observed shift walks the stack: the
            # target is whatever was pushed earlier, not a fixed address. Refuse to
            # call that resolved -- and mark the site handled so the backward pass
            # below does not count it a second time.
            if delta == 0:
                res.unresolved += 1
                handled_sites.add(jmp_va - buf_va)
                continue
            if "selfpc" not in forms:
                continue
            form = "selfpc"
            target = call_va + 5 + delta
            res.branches.append(HiddenBranch(
                jmp_site=jmp_va, register=reg, target=target,
                form=form, anchor=call_va, confidence="high", delta=delta,
                jmp_len=jmp_len, fallthrough=(target == jmp_va + jmp_len),
                shift_site=shift_site, shift_len=shift_len, pop_site=pop_site))
            handled_sites.add(jmp_va - buf_va)

    # ---------- backward-anchored forms: riplea / imm64 ----------
    #
    # These forms search BACKWARD from the jmp for a byte pattern that materialises the
    # target. Unlike the forward `selfpc` form there is no structural guarantee that the
    # match is real: `49 B8 ..` can occur inside unrelated code, and the "immediate" is
    # then just the following garbage. Measured on the reference samples, every imm64
    # match without decode validation produced a target far outside the image
    # (e.g. 0xE2FF4166F5C166FB) and was still being counted as a real edge.
    #
    # So both forms now DECODE the candidate and require it to be the instruction they
    # claim, writing the SAME register the jmp uses, and require the resulting target to
    # be a plausible in-image address. Refusing is correct: an unresolved site is honest,
    # a fabricated target is not.
    img_lo, img_hi = buf_va, buf_va + (1 << 32)   # any VA in the low 4 GB of the image

    from capstone.x86 import (X86_OP_IMM, X86_OP_MEM, X86_OP_REG, X86_REG_INVALID,
                              X86_REG_RIP, X86_REG_RIZ)

    def _decode_at(offset: int):
        raw = buf[offset:offset + 16]
        if len(raw) < 8:
            return None
        return next(md.disasm(raw, buf_va + offset, count=1), None)

    for off, reg, _kind in sites:
        if off in handled_sites:
            continue
        lo = max(0, off - BACKWARD_WINDOW)
        window = buf[lo:off]
        best = None

        if "riplea" in forms:
            for pat in LEA_RIP:
                j = window.rfind(pat)
                if j < 0:
                    continue
                gap = len(window) - (j + 7)
                if gap > MAX_LEA_GAP:
                    continue
                ins = _decode_at(lo + j)
                if ins is None or ins.mnemonic != "lea" or len(ins.operands) != 2:
                    continue
                if ins.reg_name(ins.operands[0].reg) != reg:
                    continue
                mem = ins.operands[1]
                if mem.type != X86_OP_MEM or mem.mem.base != X86_REG_RIP:
                    continue
                tgt = ins.address + ins.size + mem.mem.disp
                if not (img_lo <= tgt < img_hi):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, tgt, "riplea", ins.address)

        if best is None and "imm64" in forms:
            for pat in MOV_IMM64:
                j = window.rfind(pat)
                if j < 0:
                    continue
                gap = len(window) - (j + 10)
                if gap > MAX_IMM_GAP:
                    continue
                ins = _decode_at(lo + j)
                # capstone spells the 64-bit-immediate form `movabs`, not `mov`; checking
                # only for "mov" silently rejected every real instance of this idiom.
                if ins is None or ins.mnemonic not in ("mov", "movabs") \
                        or len(ins.operands) != 2:
                    continue
                if ins.reg_name(ins.operands[0].reg) != reg:
                    continue
                if ins.operands[1].type != X86_OP_IMM:
                    continue
                # The engine shifts the loaded constant with a flags-neutral
                # `add reg, imm32` before jumping, so the raw immediate is the
                # PRE-shift base. Without this the target is wrong yet may still land
                # inside the section, which is how the error stays invisible.
                shift, ok = _accumulate_shift(buf, buf_va, lo + j + ins.size, off, reg, md)
                if not ok:
                    continue          # register redefined: refuse rather than guess
                tgt = ins.operands[1].imm + shift
                if not (img_lo <= tgt < img_hi):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, tgt, "imm64", ins.address)
        if best is None:
            res.unresolved += 1
            continue
        _gap, tgt, form, anchor = best
        res.branches.append(HiddenBranch(
            jmp_site=buf_va + off, register=reg, target=tgt, form=form,
            anchor=anchor, confidence="medium"))

    res.internal_edges = sum(1 for b in res.branches if vm_va_lo <= b.target < vm_va_hi)
    return res


def resolve_section(pm: PEMap, section_name: str, **kw) -> DeobfResult:
    sec = pm.section_named(section_name)
    if sec is None:
        return DeobfResult()
    buf = pm.section_bytes(sec)
    va = pm.base + sec.va
    return resolve_hidden_branches(buf, va, va, va + sec.vsize, **kw)
