"""Where does the engine keep its architectural state (the "VIP" question)?

Three candidate carriers, each with a distinct, measurable signature:

  STACK          the engine spills the whole native register file to the frame and
                 then addresses virtual registers as ``[rsp + f(reg)]`` -- note the
                 *indexed* stack reference, which is the giveaway, because a normal
                 compiler emits constant slot offsets.
  REGISTER       state lives in a fixed subset of the native register file for the
                 whole body; we would then see high register pressure with almost no
                 stack traffic and no context pointer.
  CONTEXT_HEAP   an allocated structure is threaded through the body, giving memory
                 references based on a pointer register with large constant offsets.

The verdict is decided on relative magnitudes plus the entry-prologue shape, and the
raw counters are always reported so the judgement can be checked.
"""

from __future__ import annotations

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_MEM, X86_OP_REG

from .model import StateCarrier
from .pe import PEMap

CALLEE_SAVED = {"rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15"}


def prologue_shape(pm: PEMap, entry_rva: int, lookahead: int = 60) -> dict:
    """Characterise the opening of a VM entry: how much state does it spill?"""
    raw = pm.read_rva(entry_rva, lookahead * 16)
    if not raw:
        return {"instructions": 0, "pushes": 0, "callee_saved_pushes": 0,
                "frame_writes": 0, "regs_pushed": []}
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    pushes: list[str] = []
    frame_writes = 0
    n = 0
    for ins in md.disasm(raw, pm.base + entry_rva):
        n += 1
        if n > lookahead:
            break
        if ins.mnemonic == "push" and ins.operands:
            if ins.operands[0].type == X86_OP_REG:
                pushes.append(ins.reg_name(ins.operands[0].reg))
            else:
                pushes.append("imm")
        if ins.mnemonic in ("mov", "lea") and len(ins.operands) == 2 \
                and ins.operands[0].type == X86_OP_MEM:
            base = ins.reg_name(ins.operands[0].mem.base) if ins.operands[0].mem.base else None
            if base in ("rsp", "rbp"):
                frame_writes += 1
    return {
        "instructions": n,
        "pushes": len(pushes),
        "callee_saved_pushes": sum(1 for r in pushes if r in CALLEE_SAVED),
        "frame_writes": frame_writes,
        "regs_pushed": pushes,
    }


def assess(carrier_hint: dict, prologue: dict | None = None) -> StateCarrier:
    """Turn the MBA sample counters into a carrier verdict with reasoning.

    RIP-relative operands are excluded from the contest: they address image
    globals/constants, so counting them as "context struct traffic" would make any
    obfuscated body look context-based.
    """
    sc = StateCarrier()
    sc.stack_slot_refs = carrier_hint.get("stack_slot_refs", 0)
    sc.obfuscated_index_refs = carrier_hint.get("stack_indexed_refs", 0)
    sc.heap_context_refs = carrier_hint.get("pointer_base_refs", 0)
    rip = carrier_hint.get("rip_relative_refs", 0)
    if prologue:
        sc.prologue_pushes = prologue.get("pushes", 0)
        callee = prologue.get("callee_saved_pushes", 0)
        sc.notes.append(
            f"entry prologue: {prologue.get('pushes', 0)} pushes "
            f"({callee} callee-saved), {prologue.get('frame_writes', 0)} frame writes"
        )
        if callee >= 4:
            sc.notes.append("entry spills most of the callee-saved register file")

    stack_total = sc.stack_slot_refs + sc.obfuscated_index_refs
    contested = stack_total + sc.heap_context_refs
    sc.notes.append(f"rip-relative refs excluded from the contest: {rip:,}")
    if contested == 0:
        sc.carrier = "unknown"
        sc.notes.append("no frame/pointer memory sample; carrier undecidable")
        return sc

    stack_share = stack_total / contested
    heap_share = sc.heap_context_refs / contested
    sc.notes.append(
        f"frame-relative={stack_total:,} ({stack_share:.1%}) vs "
        f"pointer-register-relative={sc.heap_context_refs:,} ({heap_share:.1%})"
    )

    # The decisive signal: a compiler emits CONSTANT frame offsets. A computed
    # index into the frame means virtual register numbers are being folded into the
    # addressing mode, i.e. the virtual register file lives on the stack.
    indexed_dominates = (sc.obfuscated_index_refs > sc.stack_slot_refs
                         and sc.obfuscated_index_refs >= 100)
    if indexed_dominates:
        sc.carrier = "stack"
        sc.notes.append(
            f"DECISIVE: {sc.obfuscated_index_refs:,} indexed vs "
            f"{sc.stack_slot_refs:,} constant-offset frame references -- the virtual "
            f"register file is addressed on the stack by a computed index"
        )
        return sc

    if stack_share >= 0.6:
        sc.carrier = "stack"
    elif heap_share >= 0.6:
        sc.carrier = "context_heap"
    elif stack_share >= 0.35 and heap_share >= 0.35:
        sc.carrier = "mixed"
        sc.notes.append("no carrier dominates this sample")
    else:
        sc.carrier = "register"
    return sc
