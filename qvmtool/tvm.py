"""TVM (`.tvm0`) engine-specific analysis.

TVM is a *threaded interpreter*, so most of `deobf.py` does not apply to it. The
model implemented here is the one measured on the ACE corpus and cross-checked
against symbolic execution (`tvm-devirt trace ... --dispatches`):

  * `.text` entry is a bare `E9 rel32` trampoline into a per-function VM entry block
    -- never a `call`, and frequently **outside** the Exception-directory function
    map, so an anchored-decode-only pass loses about a quarter of them.
  * the VM context base is **RBP**; every handler reloads the VIP from a memory slot
    in that frame (`mov r9,[rbp+60h]`) and dereferences it as a **16-bit** word
    (`mov r8w,[r9]`), advancing by 2 (`add r8,2`).
  * there is still **no handler table**: each handler ends in an inline dispatch
    epilogue whose target is a per-site constant, hidden behind a flags-neutral
    `pushfq ; add reg,imm32 ; popfq` sandwich::

        push r8
        mov  r8, 1401C1F82h
        pushfq
        add  r8, -2E0E1h
        popfq
        jmp  r8                 -> 0x140193EA1

  * everything measured here is anchored on a byte pattern or on a `jmp reg` site,
    never on a linear sweep from a section start.

Two things `deobf.py` gets wrong on this engine, and which this module exists to
avoid, are worth stating explicitly because they are *silent* failures:

  * `deobf`'s backward-anchored `imm64` form matches `mov reg,imm64 ; ... ; jmp reg`
    and ignores the `add reg,imm32` in between, so it reports the pre-shift base. On
    ACE-GAME that is 1142 sites with a target that is wrong by a per-site constant --
    and the wrong target still lands inside `.tvm0`, so an "is it in the section"
    check passes.
  * `deobf`'s `riplea` form fires on the *computed* dispatch base (`lea r8,[rip+d]`
    feeding an MBA'd index), which is a base for an index, not a target.
"""

from __future__ import annotations

import collections
import struct
from dataclasses import dataclass, field
from typing import Optional

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG

from .model import Edge, Function, StateCarrier
from .pe import PEMap

#: The slot the reference report found in every module that carries the interpreter
#: core. It is *verified*, not assumed: `discover_vip_slot` re-derives it per image.
VIP_SLOT_KNOWN = 0x60
BYTECODE_UNIT = 2

#: Minimum "loaded and then dereferenced as a 16-bit word" count before the VIP slot
#: is treated as the interpreter core rather than noise. Three orders of magnitude of
#: margin in practice: core-carrying modules are 300-15,000, `ACE-Safe.dll` is below
#: 32, and the runner-up slot in a core-carrying module is a handful.
CORE_MIN_DEREFS = 32

GPR64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
         "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

PUSHFQ = 0x9C
POPFQ = 0x9D

#: The dispatch epilogue's last three bytes: the shift is applied with
#: `add reg, imm32` (REX.W 81 /0) between the pushfq/popfq pair.
_ADD_IMM32 = {0x48: 0xC0, 0x49: 0xC0}


def _md() -> Cs:
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    return md


# --------------------------------------------------------------------- anchors


def jmp_reg_sites(buf: bytes) -> list[tuple[int, str, int]]:
    """(offset, register, rex_kind) for every register-indirect `jmp`, byte-pattern
    based so it cannot be defeated by desynchronisation."""
    found: dict[int, tuple[str, int]] = {}
    for i in range(8):
        for pat, base_idx, kind in ((bytes([0xFF, 0xE0 + i]), i, 0),
                                    (bytes([0x41, 0xFF, 0xE0 + i]), 8 + i, 1)):
            start = 0
            while True:
                k = buf.find(pat, start)
                if k < 0:
                    break
                if not (kind == 0 and k > 0 and buf[k - 1] == 0x41):
                    found.setdefault(k, (GPR64[base_idx], kind))
                start = k + 1
    return [(off, reg, kind) for off, (reg, kind) in sorted(found.items())]


def epilogue_sites(buf: bytes) -> list[tuple[int, str]]:
    """`popfq ; jmp reg` sites -- the threading anchor for this engine.

    Every handler block in the analysed modules terminates this way, so the anchor
    is exact and cheap: 1149 sites on ACE-GAME.sys against the 1147 the reference
    report obtained from an independent byte-pattern census.
    """
    return [(off, reg) for off, reg, _k in jmp_reg_sites(buf) if buf[off - 1] == POPFQ]


# ------------------------------------------------------- VIP slot discovery


def _rbp_slot_loads(buf: bytes):
    """Yield (offset, reg_index, slot) for every `mov r64,[rbp+disp]`.

    Byte-pattern based on a complete instruction encoding, so every hit is a real
    instruction boundary; no decode order is assumed.
    """
    for reg in range(16):
        rex = 0x4C if reg >= 8 else 0x48
        low = reg & 7
        for modbase, ln in ((0x45, 4), (0x85, 7)):     # disp8 | disp32
            pat = bytes([rex, 0x8B, modbase | (low << 3)])
            start = 0
            while True:
                i = buf.find(pat, start)
                if i < 0:
                    break
                start = i + 1
                if ln == 4:
                    disp = int.from_bytes(buf[i + 3:i + 4], "little", signed=True)
                else:
                    disp = int.from_bytes(buf[i + 3:i + 7], "little", signed=True)
                yield i, reg, disp, ln


def _rbp_slot_stores(buf: bytes):
    """Yield (offset, reg_index, slot) for every `mov [rbp+disp],r64`."""
    for reg in range(16):
        rex = 0x4C if reg >= 8 else 0x48
        low = reg & 7
        for modbase, ln in ((0x45, 4), (0x85, 7)):
            pat = bytes([rex, 0x89, modbase | (low << 3)])
            start = 0
            while True:
                i = buf.find(pat, start)
                if i < 0:
                    break
                start = i + 1
                if ln == 4:
                    disp = int.from_bytes(buf[i + 3:i + 4], "little", signed=True)
                else:
                    disp = int.from_bytes(buf[i + 3:i + 7], "little", signed=True)
                yield i, reg, disp, ln


def _has_16bit_deref(buf: bytes, start: int, reg: int, limit: int = 64) -> bool:
    """Is `reg` used as the base of a 16-bit memory operand within `limit` bytes?

    Pattern level (not decode order): `66 [REX] 8B modrm` with `mod != 11`,
    `rm == reg & 7` and `REX.B` matching `reg >= 8`.
    """
    end = min(len(buf), start + limit)
    i = start
    while i < end - 2:
        j = buf.find(b"\x66", i, end)
        if j < 0:
            return False
        i = j + 1
        k = j + 1
        rex = 0
        if k < end and buf[k] in (0x41, 0x44, 0x45, 0x4D):
            rex = buf[k]
            k += 1
        if k + 1 >= end or buf[k] != 0x8B:
            continue
        modrm = buf[k + 1]
        if (modrm >> 6) == 3:
            continue
        if (modrm & 7) != (reg & 7):
            continue
        if bool(rex & 0x01) != (reg >= 8):
            continue
        return True
    return False


def discover_vip_slot(buf: bytes, top: int = 4) -> list[dict]:
    """Rank RBP-relative slots by "loaded, then dereferenced as a 16-bit word".

    This is the disassembler-free version of what `tvm-devirt`'s
    `Emulator::discover_vip_slot` does: rather than hard-coding `[rbp+0x60]`, find
    the slot that is reloaded and immediately treated as a pointer to the bytecode
    stream. Returns a ranked list so a caller can see the margin, not just a winner.
    """
    loads: collections.Counter = collections.Counter()
    stores: collections.Counter = collections.Counter()
    deref: collections.Counter = collections.Counter()
    for off, reg, slot, _ln in _rbp_slot_loads(buf):
        loads[slot] += 1
        if _has_16bit_deref(buf, off + 4, reg):
            deref[slot] += 1
    for _off, _reg, slot, _ln in _rbp_slot_stores(buf):
        stores[slot] += 1
    ranked = []
    for slot, n in deref.most_common(top):
        ranked.append({"slot": slot, "slot_hex": f"0x{slot & 0xFFFFFFFF:X}",
                       "loads": loads[slot], "stores": stores[slot],
                       "load_then_16bit_deref": n})
    return ranked


# ------------------------------------------------------- fetch idiom census


def slot_load_patterns(slot: int) -> tuple[bytes, ...]:
    """`mov r64,[rbp+slot]` for all 16 GPRs (disp8 form)."""
    if not 0 <= slot <= 0x7F:
        return ()
    out = []
    for reg in range(16):
        rex = 0x4C if reg >= 8 else 0x48
        out.append(bytes([rex, 0x8B, 0x45 | ((reg & 7) << 3), slot]))
    return tuple(out)


def slot_store_patterns(slot: int) -> tuple[bytes, ...]:
    """`mov [rbp+slot],r64` for all 16 GPRs (disp8 form)."""
    if not 0 <= slot <= 0x7F:
        return ()
    out = []
    for reg in range(16):
        rex = 0x4C if reg >= 8 else 0x48
        out.append(bytes([rex, 0x89, 0x45 | ((reg & 7) << 3), slot]))
    return tuple(out)


def advance_patterns(unit: int = BYTECODE_UNIT) -> tuple[bytes, ...]:
    """`add r64, unit` for all 16 GPRs, both imm8 and imm32 encodings."""
    out = []
    for reg in range(16):
        rex = 0x4C if reg >= 8 else 0x48
        low = reg & 7
        out.append(bytes([rex, 0x83, 0xC0 | low, unit & 0xFF]))
        out.append(bytes([rex, 0x81, 0xC0 | low]) + unit.to_bytes(4, "little"))
    return tuple(out)


def count_16bit_dereferences(buf: bytes) -> int:
    """Every `66 [REX] 8B modrm` with a memory operand, i.e. a 16-bit load."""
    n = 0
    i = 0
    while True:
        j = buf.find(b"\x66", i)
        if j < 0:
            return n
        i = j + 1
        k = j + 1
        if k < len(buf) and buf[k] in (0x41, 0x44, 0x45, 0x4D):
            k += 1
        if k + 1 < len(buf) and buf[k] == 0x8B and (buf[k + 1] >> 6) != 3:
            n += 1


def code_marker_patterns(slot: int = VIP_SLOT_KNOWN) -> tuple[bytes, ...]:
    """The pattern set that proves a block is plaintext TVM code.

    Generated over all 16 GPRs on purpose. The reference report's census hard-coded
    `mov r9,[rbp+60h]`, which reports 0 for SGuardAgent64.dll even though that module
    has 481 VIP reloads and 388 16-bit dereferences -- its scratch register differs.
    A code test that responds to the wrong register is worse than no test, because its
    zero looks like a finding.
    """
    return (slot_load_patterns(slot) + slot_store_patterns(slot)
            + advance_patterns())


def fetch_idiom_census(buf: bytes, slot: int = VIP_SLOT_KNOWN) -> dict:
    """Byte-level census of the bytecode-fetch idiom (desync-immune).

    The canonical pair is `mov r9,[rbp+60h]` then `mov r8w,[r9]`, and the advance is
    `add r8,2`. Counted per slot so a module whose interpreter lives at a different
    offset is not silently reported as "no VM core".
    """
    found = {}
    for pat in slot_load_patterns(slot):
        reg = ((pat[2] >> 3) & 7) | (8 if pat[0] == 0x4C else 0)
        found[f"vip_load_{GPR64[reg]}"] = buf.count(pat)
    total_loads = sum(found.values())

    fetches = count_16bit_dereferences(buf)
    advances = sum(buf.count(p) for p in advance_patterns())

    by_reg = {k: v for k, v in sorted(found.items(), key=lambda kv: -kv[1]) if v}
    return {
        "slot": slot,
        "slot_hex": f"0x{slot:X}",
        "vip_loads_by_register": by_reg,
        "vip_loads": total_loads,
        "vip_publishes": sum(1 for _o, _r, d, _l in _rbp_slot_stores(buf)
                             if d == slot),
        "mem16_dereferences": fetches,
        f"advance_by_{BYTECODE_UNIT}": advances,
        "canonical_pair_present": bool(total_loads and fetches),
    }


# ---------------------------------------------------- dispatch epilogues


@dataclass
class DispatchSite:
    site: int               # VA of the `jmp reg`
    register: str
    target: int             # resolved VA of the next handler
    form: str               # threaded-imm64 | threaded-riplea
    delta: int              # the flags-neutral shift
    anchor: int             # VA of the instruction that materialised the base
    shift_site: int = 0
    jmp_len: int = 0
    fallthrough: bool = False
    #: target == fall-through + 1, i.e. the branch skips a single junk byte that a
    #: disassembler reads as the start of a longer instruction. See `null_branch`.
    alignment_skip: bool = False

    @property
    def null_branch(self) -> bool:
        """Does this branch stay in the linear successor region?

        Two shapes occur, and both are CFG-neutral:

        * `target == site + jmp_len` -- the plain fall-through, the QVM "self-PC
          padding" shape;
        * `target == site + jmp_len + 1` -- the target skips exactly one byte. That
          byte is a real instruction *prefix* in the thread and a decoy to a
          disassembler: on ACE-GAME at 0x1400D0144 the bytes are
          `41 FF E0 | E8 41 58 48 91`, so a decoder reads `call rel32` at 0x1400D0147
          while the branch actually lands on 0x1400D0148, where `41 58` decodes as
          `pop r8` -- consuming the operand the epilogue pushed. Counting the second
          shape as an edge would add 151 fake edges to a single module.
        """
        return self.fallthrough or self.alignment_skip

    @property
    def patch_span(self) -> tuple[int, int]:
        """The bytes that actually decide the branch: the shift plus the jmp.

        The `mov reg,imm64` and the `push` before the `pushfq` must stay: the push
        supplies the next handler's operand. Rewriting only the tail keeps the
        stack balanced.
        """
        start = self.shift_site or self.anchor
        return start, self.site + self.jmp_len

    def to_dict(self) -> dict:
        return {"site": hex(self.site), "register": self.register,
                "target": hex(self.target), "form": self.form, "delta": self.delta,
                "anchor": hex(self.anchor), "fallthrough": self.fallthrough,
                "alignment_skip": self.alignment_skip,
                "null_branch": self.null_branch}


@dataclass
class DispatchResult:
    branches: list[DispatchSite] = field(default_factory=list)
    epilogue_candidates: int = 0
    jmp_reg_sites: int = 0
    unresolved: int = 0
    internal_edges: int = 0
    tvm_va_lo: int = 0
    tvm_va_hi: int = 0

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
        null = sum(1 for b in self.branches if b.null_branch)
        ft = sum(1 for b in self.branches if b.fallthrough)
        skip = sum(1 for b in self.branches if b.alignment_skip)
        return {
            "engine": "tvm",
            "jmp_reg_sites": self.jmp_reg_sites,
            "threaded_epilogue_candidates": self.epilogue_candidates,
            "resolved_branches": self.resolved,
            "resolution_rate": round(self.resolved / self.epilogue_candidates, 4)
                               if self.epilogue_candidates else 0.0,
            "forms": self.by_form(),
            "fallthrough_only": ft,
            "alignment_skip_only": skip,
            "null_branches": null,
            "real_edges": self.resolved - null,
            "unresolved_sites": self.unresolved,
            "internal_edges": self.internal_edges,
            "distinct_targets": len(self.targets()),
        }


def _sext32(v: int) -> int:
    """Normalise an imm32 that x86-64 sign-extends, whichever way capstone reports it."""
    return ((v + 0x80000000) % 0x100000000) - 0x80000000


def _materialisation(buf: bytes, before: int, reg: str, window: int = 24):
    """Closest preceding `mov reg,imm64` / `lea reg,[rip+d]` that ends at `before`.

    Byte-pattern and anchored backwards from the `pushfq` boundary on purpose:
    decoding forward from an offset 64 bytes earlier starts mid-instruction, so a
    perfectly regular epilogue would look unresolvable. This is the same desync trap
    the README's first section warns about, in its most tempting form.
    """
    if reg not in GPR64:
        return None
    r = GPR64.index(reg)
    hi, low = r >= 8, r & 7
    mov = bytes([0x49 if hi else 0x48, 0xB8 + low])
    lea = bytes([0x4C if hi else 0x48, 0x8D, 0x05 | (low << 3)])
    best = None
    for pat, kind in ((mov, "mov_imm64"), (lea, "lea_rip")):
        j = buf.rfind(pat, max(0, before - 64), before)
        if j < 0:
            continue
        ln = 10 if kind == "mov_imm64" else 7
        if j + ln > before:
            continue
        gap = before - (j + ln)
        if gap > window:
            continue
        if best is None or gap < best[0]:
            best = (gap, j, kind, ln)
    return best


def resolve_epilogue(buf: bytes, off: int, reg: str, sec_va: int
                     ) -> Optional[tuple[int, str, int, int, int]]:
    """Resolve one threaded epilogue ending at the `jmp reg` at `off`.

    Returns ``(target_va, form, anchor_off, delta, shift_off)`` or None. The shift is
    found by requiring a single instruction that closes exactly on the `popfq`, which
    makes a coincidental match very unlikely: it has to carry the right register AND
    sit directly after a `0x9C`.
    """
    if off < 6 or buf[off - 1] != POPFQ:
        return None
    md = _md()
    shift = None
    for s in range(off - 16, off - 1):
        if s < 1:
            continue
        ins = None
        for ins in md.disasm(buf[s:off - 1], 0):
            break
        if ins is None or ins.address != 0 or ins.address + ins.size != (off - 1 - s):
            continue
        if ins.mnemonic not in ("add", "sub", "lea") or len(ins.operands) < 2:
            continue
        if ins.operands[0].type != X86_OP_REG \
                or ins.reg_name(ins.operands[0].reg) != reg:
            continue
        if buf[s - 1] != PUSHFQ:
            continue
        shift = (s, ins)
        break
    if shift is None:
        return None
    s, shift_ins = shift
    delta = 0
    if shift_ins.operands[1].type == X86_OP_IMM:
        delta = _sext32(shift_ins.operands[1].imm)
        if shift_ins.mnemonic == "sub":
            delta = -delta
    elif shift_ins.mnemonic == "lea" and shift_ins.operands[1].type == X86_OP_MEM:
        delta = shift_ins.operands[1].mem.disp
    mat = _materialisation(buf, s - 1, reg)
    if mat is None:
        return None
    _gap, j, kind, _ln = mat
    if kind == "mov_imm64":
        base = struct.unpack_from("<Q", buf, j + 2)[0]
        form = "threaded-imm64"
    else:
        disp = struct.unpack_from("<i", buf, j + 3)[0]
        base = sec_va + j + 7 + disp
        form = "threaded-riplea"
    return base + delta, form, j, delta, s


def resolve_dispatch(buf: bytes, sec_va: int, end_va: int | None = None
                     ) -> DispatchResult:
    """Resolve every threaded dispatch epilogue in `buf`."""
    sites = jmp_reg_sites(buf)
    res = DispatchResult(jmp_reg_sites=len(sites), tvm_va_lo=sec_va,
                         tvm_va_hi=end_va if end_va is not None else sec_va + len(buf))
    for off, reg, _kind in sites:
        if buf[off - 1] != POPFQ:
            continue
        res.epilogue_candidates += 1
        got = resolve_epilogue(buf, off, reg, sec_va)
        if got is None:
            res.unresolved += 1
            continue
        target, form, anchor, delta, shift_off = got
        site_va = sec_va + off
        jmp_len = 3 if _kind else 2
        succ = site_va + jmp_len
        res.branches.append(DispatchSite(
            site=site_va, register=reg, target=target, form=form, delta=delta,
            anchor=sec_va + anchor, shift_site=sec_va + shift_off, jmp_len=jmp_len,
            fallthrough=(target == succ), alignment_skip=(target == succ + 1)))
    res.internal_edges = sum(1 for b in res.branches
                             if res.tvm_va_lo <= b.target < res.tvm_va_hi)
    return res


def resolve_section(pm: PEMap, section_name: str = ".tvm0") -> DispatchResult:
    sec = pm.section_named(section_name)
    if sec is None:
        return DispatchResult()
    buf = pm.section_bytes(sec)
    va = pm.base + sec.va
    return resolve_dispatch(buf, va, va + sec.vsize)


def dispatch_edges(res: DispatchResult, base: int = 0) -> list[Edge]:
    """Real (non-null) dispatch edges as `Edge` records.

    `Edge` is documented as RVA-keyed (`xrefs`, `ghidra`, `cfg.to_tsv` all add
    `pm.base` back when emitting), so pass the image base. Do NOT mix VA-keyed and
    RVA-keyed edges in one list: `cfg.validate_edges` bounds targets with a
    VA-based range, so an RVA-keyed edge is silently dropped as out-of-section --
    which is exactly how 91 correct trampoline edges disappeared from the first cut
    of `edges` on this engine.
    """
    return [Edge(caller=0, site=b.site - base, kind="tvm-dispatch",
                 target=b.target - base)
            for b in res.branches if not b.null_branch]


# ------------------------------------------------- native E9 trampolines


def native_trampoline_edges(pm: PEMap, vm_section: str = ".tvm0",
                            rejects_mid_instruction: bool = True
                            ) -> tuple[list[Edge], dict]:
    """Every `E9 rel32` in an executable non-VM section that lands in the VM section.

    Why a byte pattern and not the anchored decoder: the vendor places these
    trampolines in `.text` **gaps that no RUNTIME_FUNCTION covers** (22 of 93 on
    ACE-GAME.sys, 19/118 on ACE-ATS64, 16/78 on ACE-CSI64). An anchored-only pass
    therefore loses ~24 % of the real entries into the VM, which is a correctness
    problem rather than a rounding error.

    This is not a linear sweep: a 5-byte pattern with a validated `rel32` that lands
    inside the VM section cannot occur by accident, and the check is decode-free.
    Where the site is also a registered function entry it is reported as such, so a
    consumer can prefer the anchored subset. A site that falls *inside* a registered
    function is accepted only if decoding from that function's own entry lands exactly
    on it -- otherwise it is a decoy operand byte and is rejected.
    """
    vm = pm.section_named(vm_section)
    if vm is None:
        return [], {}
    lo, hi = vm.va, vm.va + vm.vsize
    funcs = _recover(pm)
    edges: list[Edge] = []
    stats = {"raw_sites": 0, "at_function_entry": 0, "outside_function_map": 0,
             "mid_body_confirmed_boundary": 0, "rejected_mid_instruction": 0}
    for s in pm.exec_sections():
        if s.name == vm_section:
            continue
        buf = pm.section_bytes(s)
        for i in range(0, len(buf) - 5):
            if buf[i] != 0xE9:
                continue
            rel = int.from_bytes(buf[i + 1:i + 5], "little", signed=True)
            tgt = s.va + i + 5 + rel
            if not (lo <= tgt < hi):
                continue
            site = s.va + i
            stats["raw_sites"] += 1
            owner = _owner(funcs, site)
            if owner is not None and owner.begin == site:
                stats["at_function_entry"] += 1
            elif owner is None:
                stats["outside_function_map"] += 1
            elif rejects_mid_instruction:
                if is_instruction_boundary(pm, owner.begin, site):
                    stats["mid_body_confirmed_boundary"] += 1
                else:
                    stats["rejected_mid_instruction"] += 1
                    continue
            edges.append(Edge(caller=owner.begin if owner else site, site=site,
                              kind="e9-trampoline", target=tgt))
    edges.sort(key=lambda e: e.target)
    return edges, stats


def is_instruction_boundary(pm: PEMap, begin: int, site: int,
                            max_resyncs: int = 64) -> bool:
    """Does a decode that starts at `begin` land exactly on `site`?

    Same resynchronising walk `xrefs._iter_insns` uses: capstone stops at the first
    byte it cannot decode, so a stall is expected on mutated bodies and must not be
    read as "boundary".
    """
    if site <= begin:
        return False
    raw = pm.read_rva(begin, site - begin)
    if not raw:
        return False
    md = _md()
    off = 0
    resyncs = 0
    n = len(raw)
    while off < n:
        progressed = False
        for ins in md.disasm(raw[off:off + 64], 0):
            off += ins.size
            progressed = True
            if off >= n:
                break
        if off == n:
            return True
        if off > n:
            return False
        if not progressed:
            if resyncs >= max_resyncs:
                return False
            nxt = None
            for cand in range(off + 1, min(off + 33, n)):
                got = 0
                total = 0
                for ins in md.disasm(raw[cand:cand + 48], 0):
                    got += 1
                    total += ins.size
                    if got >= 3 and total >= 6:
                        break
                if got >= 3 and total >= 6:
                    nxt = cand
                    break
            if nxt is None:
                return False
            resyncs += 1
            off = nxt
    return off == n


def _recover(pm: PEMap):
    from .functions import recover_functions
    return recover_functions(pm)


def _owner(funcs: list[Function], rva: int) -> Optional[Function]:
    for f in funcs:
        if f.begin <= rva < f.end:
            return f
    return None


# ------------------------------------------------------- context / carrier


def context_frame_census(buf: bytes, limit_offsets: int = 0x100) -> dict:
    """RBP-relative slot histogram: which constant offsets carry the context.

    TVM addresses its context with **constant** offsets (`[rbp+0x08]`=RAX,
    `[rbp+0x18]`=RBX, `[rbp+0x30]`=RDX, `[rbp+0x50]`=R9, `[rbp+0x60]`=VIP, ...), the
    opposite of the QVM finding that virtual register numbers are folded into the
    addressing mode. So the QVM `assess()` decisive signal can never fire here, and a
    verdict has to be built from the slot layout instead.
    """
    loads: collections.Counter = collections.Counter()
    stores: collections.Counter = collections.Counter()
    for _off, _reg, slot, _ln in _rbp_slot_loads(buf):
        if 0 <= slot < limit_offsets:
            loads[slot] += 1
    for _off, _reg, slot, _ln in _rbp_slot_stores(buf):
        if 0 <= slot < limit_offsets:
            stores[slot] += 1
    #: a stride-8 ladder of written slots is the guest register image
    written = sorted(s for s in stores if s % 8 == 0)
    ladder = 0
    run = 0
    for a, b in zip(written, written[1:]):
        run = run + 1 if b - a == 8 else 0
        ladder = max(ladder, run + 1)
    return {
        "loads": dict(sorted(loads.items(), key=lambda kv: -kv[1])[:16]),
        "stores": dict(sorted(stores.items(), key=lambda kv: -kv[1])[:16]),
        "distinct_slots_loaded": len(loads),
        "distinct_slots_stored": len(stores),
        "largest_stride8_write_ladder": ladder,
        "slot_0x60_loads": loads.get(VIP_SLOT_KNOWN, 0),
        "slot_0x60_stores": stores.get(VIP_SLOT_KNOWN, 0),
    }


def assess_carrier(section: str, vip_ranked: list[dict], census: dict,
                   carrier_hint: dict | None = None) -> StateCarrier:
    """Engine-correct carrier verdict for TVM, with the counters behind it."""
    sc = StateCarrier()
    hint = carrier_hint or {}
    sc.stack_slot_refs = hint.get("stack_slot_refs", 0)
    sc.obfuscated_index_refs = hint.get("stack_indexed_refs", 0)
    sc.heap_context_refs = hint.get("pointer_base_refs", 0)
    rip = hint.get("rip_relative_refs", 0)
    sc.notes.append(f"rip-relative refs excluded from the contest: {rip:,}")
    sc.notes.append(
        f"RBP-relative slots: {census.get('distinct_slots_loaded', 0)} loaded / "
        f"{census.get('distinct_slots_stored', 0)} stored, "
        f"largest stride-8 write ladder {census.get('largest_stride8_write_ladder', 0)}")

    vip = vip_ranked[0] if vip_ranked else None
    if vip is not None and vip.get("load_then_16bit_deref", 0) < CORE_MIN_DEREFS:
        # same bar `detect` uses, so a verdict and a carrier answer cannot disagree
        vip = None
    if vip is None:
        sc.carrier = "unknown"
        sc.notes.append("no `mov reg,[rbp+d]` + 16-bit dereference pair found: "
                        "the TVM interpreter core is not present in this section")
        return sc

    margin = ""
    if len(vip_ranked) > 1:
        margin = (f" (next slot 0x{vip_ranked[1]['slot']:X}: "
                  f"{vip_ranked[1]['load_then_16bit_deref']} vs {vip['load_then_16bit_deref']})")
    sc.carrier = "stack"
    sc.notes.append(
        f"DECISIVE: the VM context base is RBP -- slot 0x{vip['slot']:X} is reloaded "
        f"{vip['loads']:,}x and dereferenced as a 16-bit word at least "
        f"{vip['load_then_16bit_deref']:,}x{margin}")
    sc.notes.append(
        f"Verdict: stack (RBP-relative VM context frame); VIP = the qword at "
        f"[rbp+0x{vip['slot']:X}], not a register and not a heap struct. "
        f"Note this is NOT the QVM mechanism: the frame is addressed with CONSTANT "
        f"offsets, so the QVM `indexed frame refs` test cannot fire on TVM.")
    return sc
