#!/usr/bin/env python3
"""Task 2 driver: emulate from REAL call contexts and recover the vreg slot map.

Why this differs from the earlier probe
---------------------------------------
The first probe started at `.qvm0`'s section entry with a zeroed register file, so
the observed path was a junk path and the run left the image after ~130 instructions.
Here we start where the module actually starts VM work:

  * `ord7` (RVA 0x0B0E30) whose FIRST instruction is `call 0x989357` into `.qvm0`;
  * `0x658F0`, the native stub that calls `.qvm0` 0x994437 (reachable from
    `CreateObjectUd`/`CreateObjectUdp`), driven with a valid UTF-16 string argument
    because that native body is a string-length walk.

We also emulate `CreateObject` itself (RVA 0x03BA70) to let the real prologue run,
with the allocator hooked so the object allocation succeeds.

The point is not to "run the anti-cheat" -- it is to observe, in the instructions
that are genuinely reached, which frame slots the VM uses and at what widths, which
is the vreg layout.
"""
from __future__ import annotations

import json
import struct
import os
import sys
import time
from collections import Counter, defaultdict

import pefile
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
from unicorn import *
from unicorn.x86_const import *

IMAGE_BUDGET = 0x20000000
STACK_BASE = 0x100000000
STACK_SIZE = 0x400000
HEAP_BASE = 0x200000000
HEAP_SIZE = 0x400000
STR_BASE = 0x300000000


class Harness:
    def __init__(self, path: str):
        self.data = open(path, "rb").read()
        self.pe = pefile.PE(data=self.data, fast_load=True)
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.secs = [(s.Name.rstrip(b"\0").decode("latin1"), self.base + s.VirtualAddress,
                      max(s.Misc_VirtualSize, s.SizeOfRawData), s.PointerToRawData,
                      s.SizeOfRawData) for s in self.pe.sections]
        q = next(s for s in self.secs if s[0] == ".qvm0")
        self.qlo, self.qhi = q[1], q[1] + q[2]
        self.img_lo = self.base
        self.img_hi = max(s[1] + s[2] for s in self.secs)

        self.uc = Uc(UC_ARCH_X86, UC_MODE_64)
        for name, va, size, raw, rsize in self.secs:
            sz = (size + 0xFFF) & ~0xFFF
            try:
                self.uc.mem_map(va, sz)
            except UcError:
                continue
            self.uc.mem_write(va, self.data[raw:raw + rsize])
        self.uc.mem_map(STACK_BASE, STACK_SIZE)
        self.uc.mem_map(HEAP_BASE, HEAP_SIZE)
        self.uc.mem_map(STR_BASE, 0x10000)

        self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.md.detail = True

        # instrumentation
        self.n = 0
        self.frame_reads: Counter = Counter()
        self.frame_writes: Counter = Counter()
        self.frame_access_sizes: dict[int, Counter] = defaultdict(Counter)
        self.rip_hist: Counter = Counter()
        self.left_image_at = None
        self.inside_vm_insns = 0
        self.vm_slots: Counter = Counter()
        self.trace: list[tuple[int, bytes, str, str]] = []
        self.depth_min = None
        self.depth_max = None
        #: RVA-less VAs whose call sites are replaced by a synthetic allocation.
        #: 0x170648 is the wrapper CreateObject calls to obtain its 0x58-byte object.
        self.allocators = {self.base + 0x170648, self.base + 0x1709D8}
        self.allocs = 0
        self.alloc_blocks: list[int] = []

    # ------------------------------------------------------------ hooks
    def hook_code(self, uc, address, size, user):
        self.n += 1
        # ---- synthetic allocator: let CreateObject's prologue actually finish ----
        if address in self.allocators:
            self.allocs += 1
            blk = HEAP_BASE + 0x10000 + self.allocs * 0x1000
            try:
                uc.mem_write(blk, b"\x00" * 0x1000)
                sp = uc.reg_read(UC_X86_REG_RSP)
                ret = struct.unpack("<Q", uc.mem_read(sp, 8))[0]
                uc.reg_write(UC_X86_REG_RSP, sp + 8)
                uc.reg_write(UC_X86_REG_RAX, blk)
                uc.reg_write(UC_X86_REG_RIP, ret)
                self.alloc_blocks.append(blk)
                return
            except UcError:
                return
        if self.qlo <= address < self.qhi:
            self.inside_vm_insns += 1
        if not (self.img_lo <= address < self.img_hi):
            if self.left_image_at is None:
                self.left_image_at = address
            return
        try:
            raw = bytes(uc.mem_read(address, size))
        except UcError:
            return
        ins = next(self.md.disasm(raw, address, count=1), None)
        if ins is None:
            return
        if len(self.trace) < 30000:
            self.trace.append((address, raw, ins.mnemonic, ins.op_str))
        self.rip_hist[address] += 1

    def hook_mem(self, uc, access, address, size, value, user):
        rsp = uc.reg_read(UC_X86_REG_RSP)
        rbp = uc.reg_read(UC_X86_REG_RBP)
        if STACK_BASE <= address < STACK_BASE + STACK_SIZE:
            d_rsp = address - rsp
            self.frame_access_sizes[d_rsp][size] += 1
            if access in (UC_MEM_READ, UC_MEM_READ_AFTER) or access == 16:
                self.frame_reads[d_rsp] += 1
            if access in (UC_MEM_WRITE, UC_MEM_WRITE_UNMAPPED, UC_MEM_WRITE_PROT):
                self.frame_writes[d_rsp] += 1
            # relative to the frame base (rbp) as well
            d_rbp = address - rbp
            self.vm_slots[d_rbp] += 1
            depth = STACK_BASE + STACK_SIZE // 2 + 0x1000 - address
            self.depth_min = depth if self.depth_min is None else min(self.depth_min, depth)
            self.depth_max = depth if self.depth_max is None else max(self.depth_max, depth)

    def hook_unmapped_fetch(self, uc, access, address, size, value, user):
        """A fetch outside the image means we reached an import thunk.

        Return a synthetic success: set rax=0 and emulate the return by popping the
        stack, so the caller continues instead of the run dying here.
        """
        try:
            sp = uc.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack("<Q", uc.mem_read(sp, 8))[0]
            uc.reg_write(UC_X86_REG_RSP, sp + 8)
            uc.reg_write(UC_X86_REG_RAX, 0)
            uc.reg_write(UC_X86_REG_RIP, ret)
            return True
        except UcError:
            return False

    def run(self, entry_rva: int, args: tuple[int, ...], max_insn: int):
        uc = self.uc
        sp = STACK_BASE + STACK_SIZE // 2
        # a plausible caller frame: terminator return address + shadow space
        uc.mem_write(sp, struct.pack("<Q", 0xDEADBEEFDEADBEEF))
        sp += 0x1000
        uc.reg_write(UC_X86_REG_RSP, sp)
        uc.reg_write(UC_X86_REG_RBP, sp)
        for r in (UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RSI, UC_X86_REG_RDI,
                  UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
                  UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15):
            uc.reg_write(r, 0)
        for reg, val in zip((UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_R8,
                             UC_X86_REG_R9), args):
            uc.reg_write(reg, val)

        uc.hook_add(UC_HOOK_CODE, self.hook_code)
        uc.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, self.hook_mem)
        uc.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, self.hook_unmapped_fetch)

        entry = self.base + entry_rva
        try:
            uc.emu_start(entry, 0, count=max_insn)
            res = "reached instruction cap"
        except UcError as e:
            res = f"stopped: {e}"
        finally:
            try:
                uc.emu_stop()
            except Exception:
                pass
        return res

    def report(self, label: str) -> dict:
        slots = sorted(self.frame_access_sizes.items())
        return {
            "label": label,
            "instructions": self.n,
            "instructions_inside_qvm0": self.inside_vm_insns,
            "left_image": (hex(self.left_image_at) if self.left_image_at else None),
            "distinct_frame_slots": len(slots),
            "frame_slot_min": slots[0][0] if slots else None,
            "frame_slot_max": slots[-1][0] if slots else None,
            "stack_bytes_touched": (self.depth_max - self.depth_min
                                    if self.depth_min is not None else 0),
            "top_rsp_slots": [
                {"d_rsp": d, "reads": self.frame_reads.get(d, 0),
                 "writes": self.frame_writes.get(d, 0),
                 "widths": dict(self.frame_access_sizes[d])}
                for d, _ in sorted(self.frame_access_sizes.items())[:1]
            ],
        }

    def slot_table(self, limit: int = 60):
        rows = []
        for d, widths in sorted(self.frame_access_sizes.items()):
            rows.append((d, sum(widths.values()), self.frame_reads.get(d, 0),
                         self.frame_writes.get(d, 0), dict(widths)))
        rows.sort(key=lambda r: -r[1])
        return rows[:limit]


def main(path: str, out_prefix: str):
    results = {}
    # ---- A: ord7 (first instruction calls into .qvm0) ----
    h = Harness(path)
    res = h.run(0xB0E30, (0, 0, 0, 0), 400000)
    results["ord7"] = {**h.report("ord7"), "result": res,
                       "slots": h.slot_table(40)}
    with open(out_prefix + "_ord7_slots.tsv", "w") as f:
        f.write("d_rsp\taccesses\treads\twrites\twidths\n")
        for d, acc, rd, wr, w in h.slot_table(400):
            f.write(f"{d}\t{acc}\t{rd}\t{wr}\t{json.dumps(w)}\n")

    # ---- B: 0x658F0 with a real UTF-16 string (its native body walks one) ----
    h2 = Harness(path)
    s = "ACE-PBC-Game64".encode("utf-16-le") + b"\x00\x00"
    h2.uc.mem_write(STR_BASE, s)
    res2 = h2.run(0x658F0, (STR_BASE, 0, 0, 0), 400000)
    results["stub658F0"] = {**h2.report("stub658F0"), "result": res2,
                            "slots": h2.slot_table(40)}
    with open(out_prefix + "_stub658F0_slots.tsv", "w") as f:
        f.write("d_rsp\taccesses\treads\twrites\twidths\n")
        for d, acc, rd, wr, w in h2.slot_table(400):
            f.write(f"{d}\t{acc}\t{rd}\t{wr}\t{json.dumps(w)}\n")

    # ---- C: CreateObject with a valid out-parameter ----
    h3 = Harness(path)
    out_ptr = HEAP_BASE + 0x1000
    h3.uc.mem_write(out_ptr, b"\x00" * 8)
    res3 = h3.run(0x3BA70, (out_ptr, 0, 0, 0), 200000)
    results["CreateObject"] = {**h3.report("CreateObject"), "result": res3,
                               "slots": h3.slot_table(40)}

    with open(out_prefix + ".json", "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(f"== {k}: {v['result']}")
        print(f"   instructions={v['instructions']} "
              f"inside_qvm0={v['instructions_inside_qvm0']} "
              f"left_image={v['left_image']}")
        print(f"   distinct frame slots={v['distinct_frame_slots']} "
              f"range=[{v['frame_slot_min']}, {v['frame_slot_max']}] "
              f"stack bytes={v['stack_bytes_touched']}")
        for row in v["slots"][:10]:
            d, acc, rd, wr, w = row
            print(f"     [rsp{d:+d}] accesses={acc} r={rd} w={wr} widths={w}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
