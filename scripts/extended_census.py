#!/usr/bin/env python3
"""Extend the indirect-branch census to memory-indirect forms, then re-measure.

Measured defect this addresses
------------------------------
`qvmtool.deobf._jmp_sites` matches only the **register-indirect** encodings
(`FF /4` with mod=11 -> `FF E0..E7`, and `41 FF E0..E7`). But a dynamic sweep found
**389 of 1,348 observed sites (28.9%) that are absent from that census**, which means
the census — and therefore the "5,725 opaque sites" denominator derived from it — is an
incomplete enumeration of the real dispatch population.

`FF /2` and `FF /4` also encode **memory-indirect** call/jmp: `call [rax+8]`,
`jmp [rip+disp32]`, `jmp [rax+rcx*8]`, i.e. every ModRM whose *reg* field is 2 or 4,
at any mod value. Those are precisely the forms a dispatch table or a loaded function
pointer would use, so missing them is a substantive gap, not a rounding error.

This script implements a byte-wise census over all such forms (immune to disassembler
desynchronisation, as the existing census is) and reports:
  * how many additional sites the extended census finds
  * how much of the dynamic-site set it now explains
"""
from __future__ import annotations

import collections
import os
import sys

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qvmtool import deobf
from qvmtool.pe import PEMap

REX = range(0x40, 0x50)


def extended_census(buf: bytes) -> dict[str, list[int]]:
    """Byte-wise census of indirect call/jmp, register AND memory forms.

    x86-64 encodes `call r/m64` as `FF /2` and `jmp r/m64` as `FF /4`; the ModRM
    *reg* field selects which. mod=11 gives the register forms already covered;
    mod=00/01/10 give the memory forms (`[reg]`, `[reg+disp8]`, `[reg+disp32]`,
    plus SIB and RIP-relative variants) that were missing.
    """
    out: dict[str, list[int]] = collections.defaultdict(list)
    n = len(buf)
    i = 0
    while i < n - 1:
        p = i
        if buf[p] in REX and p + 2 < n and buf[p + 1] == 0xFF:
            p += 1
        if buf[p] == 0xFF and p + 1 < n:
            modrm = buf[p + 1]
            reg = (modrm >> 3) & 7
            mod = (modrm >> 6) & 3
            if reg == 4:                      # jmp r/m64
                key = "jmp_reg" if mod == 3 else "jmp_mem"
                out[key].append(i)
            elif reg == 2:                    # call r/m64
                key = "call_reg" if mod == 3 else "call_mem"
                out[key].append(i)
        i += 1
    return {k: v for k, v in out.items()}


def load_dynamic_sites(path: str) -> set[int]:
    sites: set[int] = set()
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            try:
                sites.add(int(p[0], 16))
            except ValueError:
                continue
    return sites


def main(pe_path: str, dyn_path: str) -> None:
    pm = PEMap(pe_path)
    sec = pm.section_named(".qvm0")
    buf = pm.section_bytes(sec)
    va = pm.base + sec.va
    lo, hi = va, va + sec.vsize

    ext = extended_census(buf)
    old = {va + off for off, _r, _k in deobf._jmp_sites(buf)}
    new_reg = {va + o for o in ext.get("jmp_reg", [])}
    new_mem = {va + o for o in ext.get("jmp_mem", [])}
    call_mem = {va + o for o in ext.get("call_mem", [])}
    call_reg = {va + o for o in ext.get("call_reg", [])}
    extended = new_reg | new_mem | call_reg | call_mem

    print(f"== extended indirect-branch census in {sec.name}\n")
    print(f"   existing census (register jmp only) : {len(old)}")
    print(f"   extended census total               : {len(extended)}")
    print(f"     jmp  reg  (FF /4 mod=11)          : {len(new_reg)}")
    print(f"     jmp  mem  (FF /4 mod!=11)         : {len(new_mem)}")
    print(f"     call reg  (FF /2 mod=11)          : {len(call_reg)}")
    print(f"     call mem  (FF /2 mod!=11)         : {len(call_mem)}")
    print(f"   NEW sites the old census missed     : {len(extended - old)}")
    print(f"   old sites the extended one misses   : {len(old - extended)}")

    dyn = load_dynamic_sites(dyn_path)
    print(f"\n== reconciliation against {len(dyn)} dynamically observed sites")
    print(f"   explained by existing census        : {len(dyn & old)} "
          f"({len(dyn & old)/max(1,len(dyn)):.1%})")
    print(f"   explained by extended census        : {len(dyn & extended)} "
          f"({len(dyn & extended)/max(1,len(dyn)):.1%})")
    print(f"   still unexplained                   : {len(dyn - extended)} "
          f"({len(dyn - extended)/max(1,len(dyn)):.1%})")

    unexplained = sorted(dyn - extended)
    if unexplained:
        print("\n   sample of still-unexplained dynamic sites:")
        from capstone import CS_ARCH_X86, CS_MODE_64, Cs
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        md.detail = True
        for a in unexplained[:10]:
            raw = pm.read_rva(a - pm.base, 16)
            ins = next(md.disasm(raw, a, count=1), None) if raw else None
            print(f"      0x{a:X}  {raw[:8].hex(' '):<24} "
                  f"{ins.mnemonic + ' ' + ins.op_str if ins else '?'}")
        print("   (a site we observed executing an indirect branch at, but which no")
        print("    byte pattern accounts for -- these need the executed-RIP log, not")
        print("    a static census, to explain.)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
