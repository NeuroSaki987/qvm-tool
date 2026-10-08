"""Dynamic edge recovery: emulate the VM and observe where its indirect jumps go.

Why this is the right attack on the remaining sites
--------------------------------------------------
Static slicing leaves a large class unresolved: the target is
``Const(base) + MBA(<load from the obfuscated stack region>)``. The offset genuinely
depends on runtime state, so no amount of static folding will produce it. But the
*state itself* is produced by the VM's own prologue -- it is not attacker input --
so simply running the VM and watching produces the answers directly.

This is the complement to `slice.py`: the slicer proves what is statically constant,
the emulator observes what is dynamically taken. Merging both maximises coverage, and
where they disagree the disagreement is informative (a dynamic target that the slicer
called opaque is exactly the case that motivated this module).

Efficiency note
---------------
Decoding every executed instruction with capstone would dominate the runtime. Register
indirect `jmp`/`call` always begin with ``FF`` (or ``41 FF`` when the target is
r8-r15), so the tracer pre-filters on the first byte and only decodes candidates.
"""

from __future__ import annotations

import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .model import Edge
from .pe import PEMap

try:  # unicorn is optional: the static pipeline must work without it
    from unicorn import (UC_ARCH_X86, UC_HOOK_CODE, UC_HOOK_MEM_FETCH_UNMAPPED,
                         UC_HOOK_MEM_READ, UC_HOOK_MEM_READ_UNMAPPED,
                         UC_HOOK_MEM_WRITE, UC_HOOK_MEM_WRITE_UNMAPPED, UC_MEM_READ,
                         UC_MEM_WRITE, UC_MODE_64, Uc, UcError)
    from unicorn.x86_const import (UC_X86_REG_RBP, UC_X86_REG_RAX, UC_X86_REG_RIP,
                                   UC_X86_REG_RSP)
    HAVE_UNICORN = True
except Exception:  # pragma: no cover
    HAVE_UNICORN = False

STACK_BASE = 0x100000000
STACK_SIZE = 0x1000000
HEAP_BASE = 0x300000000
HEAP_SIZE = 0x800000
ARG_BASE = 0x400000000


@dataclass
class DynamicResult:
    #: (site VA, target VA) -> observation count
    edges: Counter = field(default_factory=Counter)
    site_regs: dict[int, str] = field(default_factory=dict)
    site_targets: dict[int, set[int]] = field(default_factory=dict)
    site_hits: Counter = field(default_factory=Counter)
    frame_slots: dict[int, Counter] = field(default_factory=dict)
    entries_run: int = 0
    entries_with_vm: int = 0
    instructions: int = 0
    vm_instructions: int = 0
    failed_entries: int = 0

    def site_rows(self):
        return [(s, self.site_regs.get(s, "?"), sorted(self.site_targets.get(s, ())))
                for s in self.site_targets]

    def as_edges(self, kind: str = "dynamic") -> list[Edge]:
        return [Edge(caller=0, site=s, target=t, kind=kind) for (s, t) in self.edges]

    def summary(self) -> dict:
        multi = sum(1 for s, ts in self.site_targets.items() if len(ts) > 1)
        return {
            "entries_run": self.entries_run,
            "entries_with_vm_instructions": self.entries_with_vm,
            "entries_failed": self.failed_entries,
            "instructions": self.instructions,
            "vm_instructions": self.vm_instructions,
            "indirect_sites_seen": len(self.site_targets),
            "dynamic_edges": len(self.edges),
            "distinct_sites": len({s for s, _ in self.edges}),
            "sites_with_multiple_targets": multi,
            "frame_slots_touched": len(self.frame_slots),
            "frame_slot_min": min(self.frame_slots) if self.frame_slots else None,
            "frame_slot_max": max(self.frame_slots) if self.frame_slots else None,
        }


#: capstone register name -> unicorn register constant, for the 16 GPRs. The two
#: libraries number registers differently, so mapping by NAME is the only safe way to
#: read a branch target out of the emulator.
def _gpr_map():
    if not HAVE_UNICORN:
        return {}
    from unicorn import x86_const as X
    return {
        "rax": X.UC_X86_REG_RAX, "rcx": X.UC_X86_REG_RCX, "rdx": X.UC_X86_REG_RDX,
        "rbx": X.UC_X86_REG_RBX, "rsp": X.UC_X86_REG_RSP, "rbp": X.UC_X86_REG_RBP,
        "rsi": X.UC_X86_REG_RSI, "rdi": X.UC_X86_REG_RDI, "r8": X.UC_X86_REG_R8,
        "r9": X.UC_X86_REG_R9, "r10": X.UC_X86_REG_R10, "r11": X.UC_X86_REG_R11,
        "r12": X.UC_X86_REG_R12, "r13": X.UC_X86_REG_R13, "r14": X.UC_X86_REG_R14,
        "r15": X.UC_X86_REG_R15,
    }


class _Tracer:
    """One emulation run, instrumented for indirect-branch observation."""

    def __init__(self, pm: PEMap, vm_section: str, allocators: Iterable[int] = ()):
        from unicorn import Uc as _Uc
        self.pm = pm
        self.sec = pm.section_named(vm_section)
        assert self.sec is not None
        self.qlo = pm.base + self.sec.va
        self.qhi = self.qlo + self.sec.vsize
        self.img_lo = pm.base
        self.img_hi = max(pm.base + s.va + max(s.vsize, s.rsize)
                          for s in pm.sections)
        uc = _Uc(UC_ARCH_X86, UC_MODE_64)
        for s in pm.sections:
            size = (max(s.vsize, s.rsize) + 0xFFF) & ~0xFFF
            if size == 0:
                continue
            try:
                uc.mem_map(pm.base + s.va, size)
            except UcError:
                continue
            uc.mem_write(pm.base + s.va, pm.data[s.raw:s.raw + s.rsize])
        uc.mem_map(STACK_BASE, STACK_SIZE)
        uc.mem_map(HEAP_BASE, HEAP_SIZE)
        uc.mem_map(ARG_BASE, 0x10000)
        self.uc = uc
        self.allocators = set(allocators)
        self.allocs = 0
        self.md = None  # lazily built capstone
        self.res = DynamicResult()
        self._frames: dict[int, Counter] = defaultdict(Counter)
        self._extra_pages: set[int] = set()
        self.mapped_extra = 0
        self._gprs = _gpr_map()

    # ---------------------------------------------------------------- hooks
    def _decode_jmp(self, address, raw):
        if self.md is None:
            from capstone import CS_ARCH_X86, CS_MODE_64, Cs
            self.md = Cs(CS_ARCH_X86, CS_MODE_64)
            self.md.detail = True
        return next(self.md.disasm(raw, address, count=1), None)

    def hook_code(self, uc, address, size, user):
        self.res.instructions += 1
        if address in self.allocators:
            self.allocs += 1
            blk = HEAP_BASE + 0x10000 + self.allocs * 0x1000
            try:
                uc.mem_write(blk + 0x1000 - 1, b"\x00")
                sp = uc.reg_read(UC_X86_REG_RSP)
                ret = struct.unpack("<Q", uc.mem_read(sp, 8))[0]
                uc.reg_write(UC_X86_REG_RSP, sp + 8)
                uc.reg_write(UC_X86_REG_RAX, blk)
                uc.reg_write(UC_X86_REG_RIP, ret)
            except UcError:
                pass
            return
        if not (self.img_lo <= address < self.img_hi):
            return
        in_vm = self.qlo <= address < self.qhi
        if in_vm:
            self.res.vm_instructions += 1
        if not in_vm:
            return
        # Cheap pre-filter. A register/memory-indirect jmp/call is `FF /2` or `FF /4`,
        # optionally behind ANY REX prefix -- so `48 FF E0` (jmp rax) is as common as
        # bare `FF E0`, and accepting only `FF`/`41 FF` silently loses most sites.
        try:
            head = bytes(uc.mem_read(address, 2))
        except UcError:
            return
        rex = 0x40 <= head[0] <= 0x4F
        if not (head[0] == 0xFF or (rex and len(head) > 1 and head[1] == 0xFF)):
            return
        try:
            raw = bytes(uc.mem_read(address, size))
        except UcError:
            return
        ins = self._decode_jmp(address, raw)
        if ins is None or ins.mnemonic not in ("jmp", "call") or not ins.operands:
            return
        from capstone.x86 import X86_OP_IMM
        op = ins.operands[0]
        if op.type == X86_OP_IMM:
            # A direct branch (`call rel32` / `jmp rel32`) is not what we are hunting.
            # NOTE: in capstone's x86 binding X86_OP_REG == 1 and X86_OP_IMM == 2, so a
            # bare `op.type == 1` tests REG, not IMM. That mistake silently discarded
            # every register-indirect branch and made an earlier sweep report zero
            # dynamic edges across 3M instructions. Use the named constant.
            return
        # The code hook fires BEFORE the instruction executes, so RIP still holds the
        # instruction's own address -- reading RIP here would record every edge as a
        # self-loop. The target has to be computed from the operand instead.
        target = self._indirect_target(uc, ins)
        if target is None:
            return
        self.res.site_hits[address] += 1
        self.res.site_regs.setdefault(address, ins.op_str)
        self.res.site_targets.setdefault(address, set()).add(target)
        self.res.edges[(address, target)] += 1

    def _indirect_target(self, uc, ins):
        """Value the indirect branch will jump to, evaluated pre-execution."""
        from capstone.x86 import X86_OP_MEM, X86_OP_REG
        op = ins.operands[0]
        gpr = self._gprs
        try:
            if op.type == X86_OP_REG:
                name = ins.reg_name(op.reg)
                rid = gpr.get(name)
                return None if rid is None else uc.reg_read(rid)
            if op.type == X86_OP_MEM:
                mem = op.mem
                addr = 0
                if mem.base:
                    rid = gpr.get(ins.reg_name(mem.base))
                    if rid is None:
                        return None
                    addr += uc.reg_read(rid)
                if mem.index:
                    rid = gpr.get(ins.reg_name(mem.index))
                    if rid is not None:
                        addr += uc.reg_read(rid) * (mem.scale or 1)
                addr += mem.disp
                raw = uc.mem_read(addr, 8)
                return int.from_bytes(bytes(raw), "little")
        except Exception:
            return None
        return None

    def hook_mem(self, uc, access, address, size, value, user):
        if STACK_BASE <= address < STACK_BASE + STACK_SIZE:
            d = address - uc.reg_read(UC_X86_REG_RSP)
            self._frames[d][size] += 1

    def hook_unmapped(self, uc, access, address, size, value, user):
        """Map a stray data page on demand; refuse to chase fetches off-image.

        A `call $+5`-style idiom can compute a data address that no static mapping
        covers. Mapping it keeps the run alive. But mapping *fetches* is how a tracer
        ends up executing a sea of zeros forever, so fetches are left to
        `hook_unmapped_fetch`, which only fakes import returns.
        """
        if self.mapped_extra >= 512:
            uc.emu_stop()
            return False
        page = address & ~0xFFF
        if page in self._extra_pages:
            return False
        try:
            uc.mem_map(page, 0x1000)
            self._extra_pages.add(page)
            self.mapped_extra += 1
        except UcError:
            return False
        return True

    def hook_unmapped_fetch(self, uc, access, address, size, value, user):
        """Reached an import thunk: fake the return so the run continues."""
        try:
            sp = uc.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack("<Q", uc.mem_read(sp, 8))[0]
            uc.reg_write(UC_X86_REG_RSP, sp + 8)
            uc.reg_write(UC_X86_REG_RAX, 0)
            uc.reg_write(UC_X86_REG_RIP, ret)
            return True
        except UcError:
            return False

    # ------------------------------------------------------------------ run
    def run(self, entry_va: int, budget: int) -> None:
        uc = self.uc
        sp = STACK_BASE + STACK_SIZE // 2
        uc.mem_write(sp, struct.pack("<Q", 0xDEADBEEFDEADBEEF))
        sp += 0x2000
        uc.reg_write(UC_X86_REG_RSP, sp)
        uc.reg_write(UC_X86_REG_RBP, sp)
        from unicorn.x86_const import (UC_X86_REG_RBX, UC_X86_REG_RCX, UC_X86_REG_RDX,
                                       UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_R8,
                                       UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
                                       UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14,
                                       UC_X86_REG_R15)
        for r in (UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RSI, UC_X86_REG_RDI,
                  UC_X86_REG_R10, UC_X86_REG_R11, UC_X86_REG_R12, UC_X86_REG_R13,
                  UC_X86_REG_R14, UC_X86_REG_R15):
            uc.reg_write(r, 0)
        # plausible pointer/flag arguments in the integer-argument registers
        for i, r in enumerate((UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_R8,
                               UC_X86_REG_R9)):
            uc.reg_write(r, ARG_BASE + i * 0x40)
        uc.mem_write(ARG_BASE, b"\x00" * 0x10000)

        uc.hook_add(UC_HOOK_CODE, self.hook_code)
        uc.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, self.hook_mem)
        uc.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED,
                    self.hook_unmapped)
        uc.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, self.hook_unmapped_fetch)
        try:
            uc.emu_start(entry_va, 0, count=budget)
        except UcError:
            self.res.failed_entries += 1
        finally:
            try:
                uc.emu_stop()
            except Exception:
                pass
            uc.hook_del_all() if hasattr(uc, "hook_del_all") else None

    def harvest(self, into: DynamicResult) -> None:
        into.edges.update(self.res.edges)
        into.site_hits.update(self.res.site_hits)
        into.instructions += self.res.instructions
        into.vm_instructions += self.res.vm_instructions
        into.failed_entries += self.res.failed_entries
        for s, ts in self.res.site_targets.items():
            into.site_targets.setdefault(s, set()).update(ts)
            into.site_regs.setdefault(s, self.res.site_regs.get(s, "?"))
        if self.res.vm_instructions:
            into.entries_with_vm += 1
        for d, widths in self._frames.items():
            into.frame_slots.setdefault(d, Counter()).update(widths)


def default_entries(pm: PEMap, vm_section: str, limit: int = 64) -> list[int]:
    """Native **caller** RVAs, most-shared-first.

    Entering a VM entry block directly is useless: the block is mid-computation and
    its incoming state is undefined, so the run faults within ~100 instructions.
    Starting at the native caller instead lets that function's own prologue establish
    the state the VM then consumes -- which is the difference between 100 and 34,000
    instructions of useful coverage.
    """
    from . import functions as func_mod, xrefs as xref_mod
    funcs = func_mod.recover_functions(pm)
    edges = xref_mod.native_to_vm_edges(pm, funcs, vm_section)
    counts: Counter = Counter(e.caller for e in edges)
    return [c for c, _ in counts.most_common(limit)]


def recover_dynamic_edges(pm: PEMap, vm_section: str = ".qvm0",
                          entry_rvas: Optional[list[int]] = None,
                          budget: int = 200_000, max_entries: int = 64,
                          allocator_rvas: Iterable[int] = (0x170648, 0x1709D8),
                          verbose: bool = False, progress=None) -> DynamicResult:
    """Emulate from each entry and merge the observed indirect-branch targets."""
    if not HAVE_UNICORN:
        raise RuntimeError("unicorn is required for dynamic edge recovery")
    entries = list(entry_rvas) if entry_rvas else default_entries(
        pm, vm_section, max_entries)
    entries = entries[:max_entries]
    allocs = [pm.base + r for r in allocator_rvas]

    total = DynamicResult()
    for i, rva in enumerate(entries):
        t = _Tracer(pm, vm_section, allocs)
        t.run(pm.base + rva, budget)
        t.harvest(total)
        total.entries_run += 1
        if verbose and (i % 8 == 0 or i == len(entries) - 1):
            msg = (f"entry {i+1}/{len(entries)} RVA 0x{rva:X}: "
                   f"edges_so_far={len(total.edges)} sites={len(total.site_targets)}")
            if progress:
                progress(msg)
            else:
                print("  " + msg)
    return total
