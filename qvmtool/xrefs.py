"""Anchored cross-reference discovery: native code -> VM section.

Why anchored rather than linearly swept
--------------------------------------
x86-64 is variable length. A linear sweep from a section start WILL desynchronise
on obfuscated or padded bytes and then silently drop real branches -- on these
images it desynchronises within a few hundred bytes, which is how a naive scan
reports zero references into the VM section even though dozens exist.

So this module decodes each native function **from its own RUNTIME_FUNCTION
entry**, bounded by that entry's end. Every reported branch is therefore a
genuinely decoded instruction, and the result is a vetted edge list rather than a
superset of coincidences.
"""

from __future__ import annotations

from typing import Iterable, Optional

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_REG_RIP

from .model import Edge, Function
from .pe import PEMap

#: Bound on how much of a function we decode. The giant VM "function" is excluded
#: by construction (we only walk native callers), but one caller can still be large.
MAX_FUNCTION_DECODE = 0x20000


def _md() -> Cs:
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    return md


def _next_stable_offset(raw: bytes, start: int, md: Cs, want: int = 3,
                        limit: int = 32) -> int | None:
    """Smallest offset >= start that decodes into a stable run of instructions.

    Used to fast-forward over undecodable bytes. Requiring several consecutive
    valid instructions (rather than one) keeps a lucky single-byte decode from
    dragging the cursor into the middle of real code.
    """
    end = min(start + limit, len(raw))
    for cand in range(start, end):
        got = 0
        total = 0
        for ins in md.disasm(raw[cand:cand + 48], 0):
            got += 1
            total += ins.size
            if got >= want and total >= 6:
                return cand
        if got >= want and total >= 6:
            return cand
    return None


def _iter_insns(pm: PEMap, begin: int, length: int, resync: bool = True,
                max_resyncs: int = 64):
    """Yield ``(instruction, via_resync)`` across [begin, begin+length).

    Why resynchronisation is necessary
    ----------------------------------
    ``capstone.disasm()`` is a generator that STOPS at the first byte it cannot
    decode. On a mutated body that silently truncates the scan, and any real branch
    living after the bad bytes is never observed -- which is exactly how a genuine
    ``.text`` -> VM edge gets missed (observed: a function whose only setup block
    sits mid-function, behind undecodable padding).

    So when decoding stalls we fast-forward to the next offset that decodes into a
    stable instruction run and carry on, up to ``max_resyncs`` times per function.
    Edges found only this way are flagged ``via_resync`` so a consumer can apply a
    stricter standard.
    """
    raw = pm.read_rva(begin, length)
    if not raw:
        return
    md = _md()
    off = 0
    resyncs = 0
    while off < len(raw):
        progressed = False
        for ins in md.disasm(raw[off:off + 64], pm.base + begin + off):
            yield ins, resyncs > 0
            off += ins.size
            progressed = True
        if progressed:
            continue
        if not resync or resyncs >= max_resyncs:
            return
        nxt = _next_stable_offset(raw, off + 1, md)
        if nxt is None:
            return
        resyncs += 1
        off = nxt


def native_to_vm_edges(pm: PEMap, funcs: list[Function], vm_section: str,
                       resync: bool = True) -> list[Edge]:
    """Every decoded branch from a non-VM executable section into `vm_section`."""
    vm = pm.section_named(vm_section)
    if vm is None:
        return []
    lo, hi = vm.va, vm.va + vm.vsize

    edges: list[Edge] = []
    for f in funcs:
        if f.section == vm_section or f.section is None:
            continue
        sec = pm.section_named(f.section)
        if sec is None or not sec.is_exec:
            continue
        length = min(f.size, MAX_FUNCTION_DECODE)
        for ins, via_resync in _iter_insns(pm, f.begin, length, resync=resync):
            site = ins.address - pm.base
            ops = ins.operands
            if not ops:
                continue

            # direct rel32 / rel8 branch
            if ins.mnemonic in ("call", "jmp") and ops[0].type == X86_OP_IMM:
                tgt = ops[0].imm - pm.base
                if lo <= tgt < hi:
                    edges.append(Edge(f.begin, site, ins.mnemonic, tgt, via_resync))
                continue

            # rip-relative memory branch: call/jmp qword ptr [rip+d]
            if ins.mnemonic in ("call", "jmp") and ops[0].type == X86_OP_MEM \
                    and ops[0].mem.base == X86_REG_RIP:
                tgt = (ins.address + ins.size + ops[0].mem.disp) - pm.base
                if lo <= tgt < hi:
                    edges.append(Edge(f.begin, site, ins.mnemonic + "_ripmem", tgt,
                                      via_resync))
                continue

            # rip-relative lea: address-of a VM entry taken as data
            if ins.mnemonic == "lea" and len(ops) > 1 and ops[1].type == X86_OP_MEM \
                    and ops[1].mem.base == X86_REG_RIP:
                tgt = (ins.address + ins.size + ops[1].mem.disp) - pm.base
                if lo <= tgt < hi:
                    edges.append(Edge(f.begin, site, "lea", tgt, via_resync))

    edges.sort(key=lambda e: e.target)
    return edges


def vm_entry_clusters(edges: list[Edge], minimum: int = 1) -> list[tuple[int, int]]:
    """(vm entry RVA, incoming edge count), most-shared first.

    A "shared dispatcher" in this family is simply a VM entry address that many
    distinct native call sites funnel into, because dispatch is a static direct
    call rather than a table lookup.
    """
    counts: dict[int, int] = {}
    for e in edges:
        counts[e.target] = counts.get(e.target, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(t, c) for t, c in ranked if c >= minimum]


def vm_stubs(pm: PEMap, funcs: list[Function], vm_section: str,
             resync: bool = True) -> list[tuple[int, Edge]]:
    """Native functions whose whole body is effectively a single call into the VM.

    These are the thin trampolines the vendor leaves behind after moving a
    function body into the VM payload. Requiring the branch to be the FIRST
    decoded instruction keeps comment/data false positives out.
    """
    first: dict[int, Edge] = {}
    for e in native_to_vm_edges(pm, funcs, vm_section, resync=resync):
        if e.caller not in first or e.site < first[e.caller].site:
            first[e.caller] = e
    out: list[tuple[int, Edge]] = []
    for f in funcs:
        e = first.get(f.begin)
        if e is None or e.site != f.begin:
            continue
        out.append((f.begin, e))
    return out


def find_absolute_pointers(pm: PEMap, vm_section: str, width: int = 8,
                           aligned_only: bool = True) -> list[tuple[str, int, int]]:
    """(section, holder RVA, target RVA) for data-resident pointers into the VM.

    This is the disassembly-independent test for a handler/dispatch table: a
    bytecode VM must store handler addresses (or offsets) somewhere.

    `aligned_only` matters a great deal. Scanning every byte offset turns any
    8-byte-aligned-looking repeating pattern into a burst of "pointers" that all
    share one target at odd offsets -- which is exactly what a table is *not*. A
    genuine pointer array is naturally aligned, so unaligned windows are noise and
    are dropped by default.

    Widening to 4-byte RVAs is available but callers must respect the UTF-16 trap
    documented in `utf16_false_positive_span`.
    """
    vm = pm.section_named(vm_section)
    if vm is None:
        return []
    lo, hi = vm.va, vm.va + vm.vsize
    out: list[tuple[str, int, int]] = []
    for s in pm.sections:
        if s.is_exec or s.name == ".reloc":
            continue
        raw = pm.section_bytes(s)
        step = width if aligned_only else 1
        for off in range(0, len(raw) - width, step):
            v = int.from_bytes(raw[off:off + width], "little")
            tgt = (v - pm.base) if width == 8 else v
            if lo <= tgt < hi:
                out.append((s.name, s.va + off, tgt))
    return out


def consecutive_runs(hits: list[tuple[str, int, int]], width: int = 8,
                     minimum: int = 8) -> list[tuple[str, int, int, int]]:
    """(section, start RVA, entries, span) for runs of adjacent pointer slots.

    This is the shape that actually distinguishes a table from incidental
    pointers: a handler table is many *consecutive* slots, all pointing into the
    VM section. Isolated hits are just globals that happen to hold a VM address.
    """
    runs: list[tuple[str, int, int, int]] = []
    for sec_name in {h[0] for h in hits}:
        slots = sorted(h[1] for h in hits if h[0] == sec_name)
        if not slots:
            continue
        start = prev = slots[0]
        count = 1
        for s in slots[1:]:
            if s - prev == width:
                prev = s
                count += 1
            else:
                if count >= minimum:
                    runs.append((sec_name, start, count, count * width))
                start = prev = s
                count = 1
        if count >= minimum:
            runs.append((sec_name, start, count, count * width))
    runs.sort(key=lambda r: -r[2])
    return runs


def utf16_false_positive_span(vm_lo: int, vm_hi: int) -> int:
    """Overlap between the VM RVA range and the UTF-16 ASCII dword space.

    Printable UTF-16LE ASCII produces dwords of the form 0x00XX00YY, i.e. roughly
    0x00200020..0x007E007E. When that range overlaps the VM section's RVA range,
    ANY "dword RVA points into the VM section" scan is dominated by string data and
    must not be reported as a table without an independent structural check.
    """
    return max(0, min(vm_hi, 0x007E007E) - max(vm_lo, 0x00200020))
