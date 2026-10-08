"""Obfuscation idiom census, stride test, and MBA pattern classification.

Three questions, three independent measurements:

  1. *Census* -- how dense are the injector's signature idioms? (desync-immune,
     pure byte counting, so it is trustworthy on bytes no disassembler can follow)

  2. *Stride* -- are the idioms laid out on a fixed grid? A bytecode engine that
     encodes one virtual instruction per cell would show a spike in the gap
     histogram. A flat distribution means ordinary, if heavily mutated, machine
     code -- which is what settles "is this a bytecode VM at all".

  3. *MBA* -- Mixed Boolean-Arithmetic rewriting is the mutation engine's
     fingerprint. We look for the canonical fold-away shapes (x-x, x^x, neg/not/rol
     chains on the same register) rather than trying to evaluate them.
"""

from __future__ import annotations

import collections
import math
from typing import Iterable

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_MEM, X86_OP_REG

from .detect import JCC_6, idiom_census

CALL_5 = b"\xe8\x00\x00\x00\x00"
JMP_5 = b"\xe9\x00\x00\x00\x00"


def positions(buf: bytes, pattern: bytes, limit: int = 4_000_000) -> list[int]:
    out: list[int] = []
    start = 0
    while len(out) < limit:
        i = buf.find(pattern, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def stride_profile(buf: bytes, pattern: bytes, top: int = 14) -> dict:
    """Gap histogram + gap-distribution entropy for one idiom."""
    pos = positions(buf, pattern)
    gaps = collections.Counter(b - a for a, b in zip(pos, pos[1:]))
    total = sum(gaps.values()) or 1
    h = -sum((c / total) * math.log2(c / total) for c in gaps.values())
    return {
        "sites": len(pos),
        "gap_entropy_bits": round(h, 3),
        "distinct_gaps": len(gaps),
        "top_gaps": gaps.most_common(top),
        "max_gap_share": round((gaps.most_common(1)[0][1] / total) if gaps else 0.0, 4),
    }


def full_stride_profile(buf: bytes) -> dict:
    return {
        "call_self_pc": stride_profile(buf, CALL_5),
        "jmp_self_pc": stride_profile(buf, JMP_5),
    }


def is_fixed_stride(profile: dict, dominance: float = 0.25) -> bool:
    """A cell-encoded stream would have one gap dominating the histogram."""
    return profile.get("max_gap_share", 0.0) >= dominance


# ---------------------------------------------------------------- MBA shapes

def _md() -> Cs:
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    return md


def mba_metrics(buf: bytes, anchors: Iterable[int], max_instructions: int = 400_000
                ) -> dict:
    """Sample the section by decoding from anchors; count arithmetic-redox shapes.

    `anchors` must be offsets *within* `buf`. Idiom sites make good anchors because
    they are genuine instruction boundaries, so the sample is not polluted by
    mid-instruction garbage the way a blind linear sweep would be.
    """
    md = _md()
    counts = collections.Counter()
    stack_slot = 0
    stack_indexed = 0
    heap_style = 0
    rip_relative = 0
    total = 0

    def classify_mem(ins, mem) -> None:
        """Bucket one memory operand by where its base points."""
        nonlocal stack_slot, stack_indexed, heap_style, rip_relative
        base = ins.reg_name(mem.base) if mem.base else None
        idx = ins.reg_name(mem.index) if mem.index else None
        if base == "rip":
            rip_relative += 1
        elif base in ("rsp", "rbp") and idx is None:
            stack_slot += 1
        elif base in ("rsp", "rbp"):
            stack_indexed += 1
        elif base is not None or idx is not None:
            heap_style += 1

    for anchor in anchors:
        if total >= max_instructions:
            break
        chunk = buf[anchor:anchor + 4096]
        if len(chunk) < 16:
            continue
        regs: dict[str, int] = {}
        for ins in md.disasm(chunk, anchor):
            total += 1
            if total >= max_instructions:
                break
            m = ins.mnemonic
            if m in ("neg", "not", "rol", "ror", "bswap", "adc", "sbb", "xchg"):
                counts[m] += 1
            if m in ("pushfq", "popfq", "lahf", "sahf"):
                counts[m] += 1
            if m in ("sub", "xor", "add") and len(ins.operands) == 2:
                a, b = ins.operands
                # same-register fold: sub/xor reg,reg collapses to a constant
                if a.type == X86_OP_REG and b.type == X86_OP_REG \
                        and ins.reg_name(a.reg) == ins.reg_name(b.reg):
                    counts[m + "_self"] += 1
            if m in ("mov", "lea", "add", "sub", "and", "or", "xor") and len(ins.operands) == 2:
                src = ins.operands[1]
                if src.type == X86_OP_MEM:
                    classify_mem(ins, src.mem)
            if ins.operands and ins.operands[0].type == X86_OP_MEM:
                classify_mem(ins, ins.operands[0].mem)
            # record register-name usage as a crude pressure proxy
            for op in ins.operands:
                if op.type == X86_OP_REG and op.reg:
                    nm = ins.reg_name(op.reg)
                    regs[nm] = regs.get(nm, 0) + 1

    return {
        "instructions_sampled": total,
        "arith_redox": {k: v for k, v in counts.items()},
        "stack_slot_refs": stack_slot,
        "stack_indexed_refs": stack_indexed,
        "pointer_base_refs": heap_style,
        "rip_relative_refs": rip_relative,
    }


def idiom_summary(buf: bytes) -> dict:
    cen = idiom_census(buf)
    total_self = cen["call_self_pc"] + cen["jmp_self_pc"] + cen["jcc_self_pc"]
    per_mb = total_self / max(1, len(buf)) * (1 << 20)
    return {
        "counts": cen,
        "self_pc_total": total_self,
        "self_pc_per_mb": round(per_mb, 1),
        "bytes": len(buf),
    }
