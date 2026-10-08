"""Backward slicing with MBA folding: resolve the remaining indirect-branch sites.

The problem
-----------
`deobf` resolves the sites whose target is materialised by a *self-PC* idiom. The
rest materialise a **per-site base** and then add a computed offset, so resolving
them needs real value tracking, not pattern matching:

    lea   rax, [rip + 0x2A2000]      ; per-site base baked into the code
    ...
    add   rdx, rax                   ; + (MBA-folded offset)
    jmp   rdx

The method
----------
1. Collect *trustworthy instruction boundaries* in the VM section. These are the
   anchors: `call $+5` sites (always boundaries), `lea reg,[rip+d]` ends, resolved
   hidden-branch targets, and RUNTIME_FUNCTION entries.
2. For each unresolved jmp site, pick the nearest anchor before it and decode
   forward. Keep the decode only if it lands **exactly** on the jmp site, so the
   instruction stream used for the slice is genuine rather than accidentally
   aligned.
3. Run a bounded abstract interpretation over that stream with the MBA simplifier,
   then classify the jmp register's value:
     * a constant            -> resolved target
     * `base + unknown`      -> range-constrained (base recovered, extent unknown)
     * anything else         -> opaque

Step 3 is why `mba.Unknown` exists: it keeps "this is a real base plus a genuinely
data-dependent offset" distinguishable from "we failed to parse it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import (
    X86_OP_IMM, X86_OP_MEM, X86_OP_REG, X86_REG_INVALID, X86_REG_RIP, X86_REG_RIZ,
)

from . import mba
from .mba import Binary, Const, Expr, Unknown, Unary, Var, describe, simplify
from .pe import PEMap

CALL_SELF = b"\xe8\x00\x00\x00\x00"
LEA_RIP_PREFIXES = [bytes([0x48, 0x8D]), bytes([0x4C, 0x8D]),
                    bytes([0x49, 0x8D]), bytes([0x4D, 0x8D])]
JMP_PATTERNS = [bytes([0xFF, 0xE0 + i]) for i in range(8)] + \
               [bytes([0x41, 0xFF, 0xE0 + i]) for i in range(8)]

REGS = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

SUBREG_PARENT = {
    "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
    "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
    "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
    "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
    "esp": "rsp", "sp": "rsp", "spl": "rsp",
    "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
    "esi": "rsi", "si": "rsi", "sil": "rsi",
    "edi": "rdi", "di": "rdi", "dil": "rdi",
}
for _i in range(8, 16):
    SUBREG_PARENT[f"r{_i}d"] = f"r{_i}"
    SUBREG_PARENT[f"r{_i}w"] = f"r{_i}"
    SUBREG_PARENT[f"r{_i}b"] = f"r{_i}"

SUBREG_BITS = {}
for _n in ("al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"):
    SUBREG_BITS[_n] = 8
for _i in range(8, 16):
    SUBREG_BITS[f"r{_i}b"] = 8
    SUBREG_BITS[f"r{_i}w"] = 16
    SUBREG_BITS[f"r{_i}d"] = 32
for _n in ("ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "ah", "ch", "dh", "bh"):
    SUBREG_BITS[_n] = 16
for _n in ("eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"):
    SUBREG_BITS[_n] = 32


@dataclass
class SiteResolution:
    jmp_site: int                 # VA
    register: str
    kind: str                     # resolved | range | opaque
    anchor: int = 0               # VA of the anchor the slice started from
    instrs: int = 0
    expr: str = ""
    target: Optional[int] = None
    base: Optional[int] = None
    confidence: str = "low"

    def to_dict(self) -> dict:
        d = {"jmp_site": hex(self.jmp_site), "register": self.register,
             "kind": self.kind, "anchor": hex(self.anchor), "instrs": self.instrs,
             "expr": self.expr, "confidence": self.confidence}
        if self.target is not None:
            d["target"] = hex(self.target)
        if self.base is not None:
            d["base"] = hex(self.base)
        return d


@dataclass
class SliceReport:
    sites: list[SiteResolution] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sites:
            out[s.kind] = out.get(s.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        res = [s for s in self.sites if s.kind == "resolved"]
        rng = [s for s in self.sites if s.kind == "range"]
        return {
            "sites_analysed": len(self.sites),
            "kinds": self.counts(),
            "resolved": len(res),
            "range_constrained": len(rng),
            "distinct_bases": len({s.base for s in rng if s.base}),
            "distinct_edges": len({s.target for s in res if s.target is not None}),
        }


# ------------------------------------------------------------------ anchors

def collect_anchors(pm: PEMap, section_name: str, extra: set[int] | None = None
                    ) -> list[int]:
    """VAs that are certainly instruction boundaries inside the VM section."""
    sec = pm.section_named(section_name)
    if sec is None:
        return []
    buf = pm.section_bytes(sec)
    va = pm.base + sec.va
    anchors: set[int] = set(extra or ())

    start = 0
    while True:
        k = buf.find(CALL_SELF, start)
        if k < 0:
            break
        anchors.add(va + k)
        start = k + 1

    for pref in LEA_RIP_PREFIXES:
        start = 0
        while True:
            k = buf.find(pref, start)
            if k < 0:
                break
            # modrm must be mod=00, rm=101 (rip-relative), i.e. (b & 0xC7) == 0x05
            if k + 2 < len(buf) and (buf[k + 2] & 0xC7) == 0x05:
                anchors.add(va + k)
                anchors.add(va + k + 7)
            start = k + 1

    return sorted(anchors)


def _decode_to(buf: bytes, buf_va: int, start_off: int, end_off: int, md: Cs,
               max_instrs: int = 512):
    """Decode [start_off, end_off) exactly; None unless it lands on end_off.

    `max_instrs` bounds the cost of one slice. Without it a far anchor forces a
    multi-thousand-instruction decode per site, which is what makes a naive
    anchor-chaining implementation unusably slow on a 4.7 MB function.
    """
    span = end_off - start_off
    if span <= 0 or span > 0x10000:
        return None
    chunk = buf[start_off:end_off]
    out = []
    off = 0
    for ins in md.disasm(chunk, buf_va + start_off):
        out.append(ins)
        off += ins.size
        if len(out) > max_instrs:
            return None
    if off != len(chunk):
        return None
    return out


# ------------------------------------------------------- abstract evaluation

def evaluate(instrs) -> dict[str, Expr]:
    """Bounded forward abstract interpretation; returns register -> Expr.

    Registers that the slice never writes are still reported (as `Unknown`), because
    a caller wants a value for every register. Use `evaluate_defined` when the
    difference between "defined to an unknown value" and "never defined here"
    matters -- it does, because the latter means the slice started too late.

    Two models beyond plain register tracking are essential for this engine:

    * `call $+5` pushes a **known constant** (the return address). Without this, the
      whole `call $+5; pop reg; lea reg,[reg+d]` family degrades to `?stack + d` and
      every such site is misreported as opaque.

    * constant-displacement frame slots, so a value spilled and reloaded through
      `[rsp+d]` keeps its identity instead of collapsing to `?load`.
    """
    env, _defined = _evaluate_inner(instrs)
    return env


def evaluate_defined(instrs) -> tuple[dict[str, Expr], set[str]]:
    """`evaluate`, plus the set of registers the slice actually wrote."""
    return _evaluate_inner(instrs)


def _evaluate_inner(instrs) -> tuple[dict[str, Expr], set[str]]:
    env: dict[str, Expr] = {}
    defined: set[str] = set()
    stack: list[Expr] = []
    frame: dict[tuple[str, int], Expr] = {}
    rsp_delta = 0
    flags = Unknown("flags")

    def get(name: str) -> Expr:
        if name in env:
            return env[name]
        parent = SUBREG_PARENT.get(name)
        if parent and parent in env:
            return env[parent]
        return Unknown(f"reg:{name}")

    def put(name: str, e: Expr) -> None:
        parent = SUBREG_PARENT.get(name)
        bits = SUBREG_BITS.get(name)
        if parent and bits and bits < 64:
            defined.add(parent)
            if bits == 32:
                # 32-bit writes zero-extend to 64 bits on x86-64
                env[parent] = Unary("zext", e, 64, 4)
                return
            env[parent] = Unknown(f"subreg:{name}")
            return
        defined.add(name)
        env[name] = e

    def slot_key(ins, op):
        """Constant frame slot key for a memory operand, or None."""
        if op.type != X86_OP_MEM:
            return None
        mem = op.mem
        if mem.index not in (X86_REG_INVALID, X86_REG_RIZ, 0):
            return None
        if mem.base == X86_REG_INVALID or mem.base == 0:
            return None
        base = ins.reg_name(mem.base)
        if base not in ("rsp", "rbp"):
            return None
        key = ("rsp", mem.disp + rsp_delta) if base == "rsp" else ("rbp", mem.disp)
        return key

    for ins in instrs:
        m = ins.mnemonic
        ops = ins.operands
        try:
            # -------- calls push a known return address --------
            if m == "call":
                ret = ins.address + ins.size
                if ops and ops[0].type == X86_OP_IMM:
                    pushed = mba.Const(ret, 64)
                else:
                    pushed = Unknown("call-ret")
                stack.append(pushed)
                rsp_delta -= 8
                continue

            if m in ("mov", "movabs") and len(ops) == 2:
                if ops[0].type == X86_OP_REG:
                    if ops[1].type == X86_OP_MEM:
                        key = slot_key(ins, ops[1])
                        if key is not None:
                            put(ins.reg_name(ops[0].reg),
                                frame.get(key, Unknown("frame")))
                            continue
                    put(ins.reg_name(ops[0].reg), _src(ins, ops[1], get))
                elif ops[0].type == X86_OP_MEM:
                    key = slot_key(ins, ops[0])
                    if key is not None and ops[1].type == X86_OP_REG:
                        frame[key] = get(ins.reg_name(ops[1].reg))
                    elif key is not None and ops[1].type == X86_OP_IMM:
                        frame[key] = mba.Const(ops[1].imm, 64)
                continue

            if m == "lea" and len(ops) == 2 and ops[0].type == X86_OP_REG:
                put(ins.reg_name(ops[0].reg), _mem_expr(ins, ops[1], get))
                continue

            if m in ("movzx", "movsx", "movsxd") and len(ops) == 2:
                src = _src(ins, ops[1], get, small=True)
                ext = "zext" if m == "movzx" else "sext"
                arg = (SUBREG_BITS.get(ins.reg_name(ops[1].reg), 8) // 8
                       if ops[1].type == X86_OP_REG else 8)
                put(ins.reg_name(ops[0].reg), Unary(ext, src, 64, arg))
                continue

            if m in ("add", "sub", "xor", "and", "or", "imul", "shl", "shr", "sar",
                     "rol", "ror") and len(ops) == 2 and ops[0].type == X86_OP_REG:
                d = ins.reg_name(ops[0].reg)
                if d == "rsp":
                    # track the frame pointer so [rsp+d] keys stay valid
                    if m in ("add", "sub") and ops[1].type == X86_OP_IMM:
                        rsp_delta += ops[1].imm if m == "add" else -ops[1].imm
                    else:
                        rsp_delta = 0
                b = _src(ins, ops[1], get)
                op = {"imul": "mul"}.get(m, m)
                put(d, Binary(op, get(d), b))
                continue

            if m in ("inc", "dec") and ops and ops[0].type == X86_OP_REG:
                d = ins.reg_name(ops[0].reg)
                put(d, Binary("add" if m == "inc" else "sub", get(d), mba.Const(1, 64)))
                continue
            if m == "not" and ops and ops[0].type == X86_OP_REG:
                d = ins.reg_name(ops[0].reg)
                put(d, Unary("not", get(d)))
                continue
            if m == "neg" and ops and ops[0].type == X86_OP_REG:
                d = ins.reg_name(ops[0].reg)
                put(d, Unary("neg", get(d)))
                continue
            if m == "bswap" and ops and ops[0].type == X86_OP_REG:
                d = ins.reg_name(ops[0].reg)
                put(d, Unary("bswap", get(d)))
                continue
            if m == "xchg" and len(ops) == 2 and ops[0].type == X86_OP_REG \
                    and ops[1].type == X86_OP_REG:
                a = ins.reg_name(ops[0].reg)
                b = ins.reg_name(ops[1].reg)
                va_, vb_ = get(a), get(b)
                put(a, vb_)
                put(b, va_)
                continue
            if m == "push" and ops:
                stack.append(_src(ins, ops[0], get))
                rsp_delta -= 8
                continue
            if m == "pop" and ops and ops[0].type == X86_OP_REG:
                put(ins.reg_name(ops[0].reg),
                    stack.pop() if stack else Unknown("stack"))
                rsp_delta += 8
                continue
            if m == "pushfq":
                stack.append(flags)
                rsp_delta -= 8
                continue
            if m == "popfq":
                if stack:
                    stack.pop()
                rsp_delta += 8
                continue
            if m in ("lahf", "sahf", "cdqe", "cwde", "cqo", "cwd", "nop", "int3"):
                continue
        except Exception:
            # a malformed decode must degrade to "unknown", never crash the pass
            continue
    return env, defined


def _src(ins, op, get, small: bool = False) -> Expr:
    if op.type == X86_OP_IMM:
        return mba.Const(op.imm, 64)
    if op.type == X86_OP_REG:
        return get(ins.reg_name(op.reg))
    if op.type == X86_OP_MEM:
        return _mem_expr(ins, op, get)
    return Unknown("operand")


def _mem_expr(ins, op, get) -> Expr:
    if op.type != X86_OP_MEM:
        return Unknown("mem")
    mem = op.mem
    parts: list[Expr] = []
    if mem.base == X86_REG_RIP:
        return mba.Const(ins.address + ins.size + mem.disp, 64)
    if mem.base not in (X86_REG_INVALID, 0):
        parts.append(get(ins.reg_name(mem.base)))
    if mem.index not in (X86_REG_INVALID, X86_REG_RIZ, 0):
        idx = get(ins.reg_name(mem.index))
        if mem.scale and mem.scale != 1:
            idx = Binary("mul", idx, mba.Const(mem.scale, 64))
        parts.append(idx)
    if not parts:
        return Unknown("abs-mem")
    expr = parts[0]
    for p in parts[1:]:
        expr = Binary("add", expr, p)
    if mem.disp:
        expr = Binary("add", expr, mba.Const(mem.disp & mba.MASK64, 64))
    # A memory operand is a LOAD; unless the addressing is constant it is data.
    if isinstance(simplify(expr), Const):
        return Unknown("load")
    return Unknown("load-dyn")


# ------------------------------------------------------------------- driver

def _jmp_register(buf: bytes, off: int) -> Optional[str]:
    for i, pat in enumerate(JMP_PATTERNS):
        if buf[off:off + len(pat)] == pat:
            return REGS[i] if i < 16 else None
    return None


def analyse_sites(pm: PEMap, section_name: str, jmp_site_vas: list[int],
                  anchors: list[int], window: int = 1024,
                  functions=None, max_func_slice: int = 0x8000,
                  max_anchors: int = 4, max_instrs: int = 384) -> SliceReport:
    """Slice each unresolved jmp site and classify its target expression.

    Two strategies, in order of correctness:

    1. **Whole-function slice.** If the site's owning RUNTIME_FUNCTION body is small
       enough, interpret from the function entry. This is the honest model: the entry
       is the only point where the incoming state is genuinely undefined, and every
       base materialisation inside the body is then observed in order.

    2. **Chained anchors.** Inside the 4.7 MB mega-function a whole-function slice is
       impossible, so walk anchors from the nearest one backwards. A later anchor is
       preferred, but if its decode yields a non-constant we retry from an earlier one
       up to `max_anchors` times, because the base is often materialised a few
       hundred bytes before the branch.
    """
    sec = pm.section_named(section_name)
    if sec is None:
        return SliceReport()
    buf = pm.section_bytes(sec)
    buf_va = pm.base + sec.va
    lo_va = buf_va
    hi_va = buf_va + sec.vsize
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    import bisect
    rep = SliceReport()
    owners = {}
    if functions:
        for f in functions:
            owners[f.begin] = f

    def owning_func(site_va: int):
        rva = site_va - pm.base
        best = None
        for f in (functions or ()):
            if f.begin <= rva < f.end:
                if best is None or f.size < best.size:
                    best = f
        return best

    def classify(val: Expr, reg: str, site_va: int, anchor: int, n: int
                 ) -> SiteResolution:
        const = mba.as_constant(val)
        if const is not None:
            if lo_va <= const < hi_va:
                return SiteResolution(jmp_site=site_va, register=reg, kind="resolved",
                                      anchor=anchor, instrs=n, expr=describe(val),
                                      target=const, confidence="high")
            return SiteResolution(jmp_site=site_va, register=reg, kind="opaque",
                                  anchor=anchor, instrs=n, expr=describe(val),
                                  confidence="low")
        aff = mba.affine_parts(val)
        if aff is not None and aff[1]:
            base = aff[0] if aff[0] else None
            if base is not None and not (lo_va <= base < hi_va):
                base = None
            return SiteResolution(jmp_site=site_va, register=reg, kind="range",
                                  anchor=anchor, instrs=n, expr=describe(val),
                                  base=base, confidence="medium")
        return SiteResolution(jmp_site=site_va, register=reg, kind="opaque",
                              anchor=anchor, instrs=n, expr=describe(val),
                              confidence="low")

    for site_va in jmp_site_vas:
        off = site_va - buf_va
        if off < 0 or off >= len(buf):
            continue
        reg = _jmp_register(buf, off)
        if reg is None:
            continue
        result = None

        # ---- strategy 1: whole-function slice ----
        f = owning_func(site_va)
        if f is not None and f.size <= max_func_slice and f.begin < site_va - buf_va:
            instrs = _decode_to(buf, buf_va, f.begin, off, md)
            if instrs:
                env, defined = evaluate_defined(instrs)
                if reg in defined:
                    cand = classify(env[reg], reg, site_va, buf_va + f.begin,
                                    len(instrs))
                    if cand.kind == "resolved":
                        result = cand

        # ---- strategy 2: chained anchors ----
        # Only accept an anchor that actually DEFINES the branch register. A slice
        # that merely reads it proves nothing about where the base came from, so
        # stopping there would freeze a wrong verdict; instead we keep walking back
        # until the register is written, or the anchor supply runs out.
        if result is None:
            lo = max(lo_va, site_va - window)
            cands = anchors[bisect.bisect_left(anchors, lo):
                            bisect.bisect_left(anchors, site_va)]
            tried = 0
            best = None
            for a in reversed(cands):
                if tried >= max_anchors:
                    break
                instrs = _decode_to(buf, buf_va, a - buf_va, off, md)
                if not instrs:
                    continue
                env, defined = evaluate_defined(instrs)
                if reg not in defined:
                    # this window starts too late; keep walking back
                    continue
                val = simplify(env[reg])
                # Even when `reg` was written inside the window, the write may have
                # been `add reg, <junk>` on top of an unknown incoming value. If the
                # expression still refers to Unknown("reg:<reg>") the real base came
                # from before this anchor, so this candidate proves nothing.
                if mba.contains_unknown(val, f"reg:{reg}"):
                    continue
                tried += 1
                cand = classify(val, reg, site_va, a, len(instrs))
                if cand.kind == "resolved":
                    result = cand
                    break
                if best is None:
                    best = cand
            if result is None:
                result = best

        if result is None:
            result = SiteResolution(jmp_site=site_va, register=reg, kind="opaque",
                                    expr="no-anchored-decode", confidence="low")
        rep.sites.append(result)

    return rep
