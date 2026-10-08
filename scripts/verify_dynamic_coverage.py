#!/usr/bin/env python3
"""Decisive test: does the executed code ever overlap the jmp-reg sites?

The dynamic sweep executed 2.98M instructions inside `.qvm0` and observed zero
register-indirect jumps, while a byte-level census finds 8,964 `FF E0..E7` /
`41 FF E0..E7` sites in the same 7.6 MB. One of these must be wrong:

  H1  the executed code simply never reaches the jmp-reg sites (they live in
      regions the VM does not run, e.g. decoys) -> the census is right, the
      dynamic route is right, and there is nothing to find;
  H2  the executed code DOES overlap them and we still saw none -> the detector is
      broken and the dynamic result is meaningless.

The test is a page-level set intersection: collect the 4 KB pages actually executed
inside `.qvm0`, then ask how many census sites fall inside those pages.

It also reports a control: the same intersection for the *resolved* idiom sites
(sites we know are real `jmp reg` instructions from byte inspection).
"""
from __future__ import annotations

import os
import sys
from collections import Counter

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qvmtool import deobf, emulate as emu, functions as fn, xrefs as xr
from qvmtool.pe import PEMap

PAGE = 0x1000


class CoverageTracer(emu._Tracer):
    """_Tracer that also records which pages it executed inside the VM section."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pages: Counter = Counter()

    def hook_code(self, uc, address, size, user):
        if self.qlo <= address < self.qhi:
            self.pages[address & ~(PAGE - 1)] += 1
        return super().hook_code(uc, address, size, user)


def main(path: str) -> None:
    pm = PEMap(path)
    sec = pm.section_named(".qvm0")
    buf = pm.section_bytes(sec)
    va = pm.base + sec.va
    lo, hi = va, va + sec.vsize

    # ---- census population and the known-real idiom population ----
    census = [va + off for off, _r, _k in deobf._jmp_sites(buf)]
    resolved = deobf.resolve_hidden_branches(buf, lo, lo, hi)
    real_idiom = [b.jmp_site for b in resolved.branches]
    print(f"census jmp-reg sites      : {len(census)}")
    print(f"resolved idiom jmp sites  : {len(real_idiom)}")

    # ---- execute from code-like native callers, collecting page coverage ----
    funcs = fn.recover_functions(pm)
    edges = xr.native_to_vm_edges(pm, funcs, ".qvm0")

    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)

    def density(rva: int) -> int:
        raw = pm.read_rva(rva, 256)
        return 0 if not raw or len(raw) < 256 else sum(1 for _ in md.disasm(raw, pm.base + rva))

    code_targets = {t for t in {e.target for e in edges} if density(t) >= 40}
    callers = [c for c, _ in Counter(e.caller for e in edges
                                     if e.target in code_targets).most_common()][:40]

    pages: Counter = Counter()
    for rva in callers:
        t = CoverageTracer(pm, ".qvm0", [pm.base + 0x170648])
        t.run(pm.base + rva, 200_000)
        pages.update(t.pages)
    print(f"\nentries emulated          : {len(callers)}")
    print(f"distinct .qvm0 pages run  : {len(pages)} "
          f"({len(pages)*PAGE:,} bytes = {len(pages)*PAGE/sec.vsize:.1%} of the section)")

    # ---- set intersection ----
    def in_executed(sites):
        return [s for s in sites if (s & ~(PAGE - 1)) in pages]

    c_hit = in_executed(census)
    r_hit = in_executed(real_idiom)
    print(f"\ncensus sites inside executed pages     : {len(c_hit)} / {len(census)} "
          f"({len(c_hit)/max(1,len(census)):.2%})")
    print(f"resolved idiom sites inside exec pages : {len(r_hit)} / {len(real_idiom)} "
          f"({len(r_hit)/max(1,len(real_idiom)):.2%})")

    print("\n-- verdict --")
    if not c_hit:
        print("   H1: the executed code NEVER overlaps the census sites. The 8,964")
        print("   jmp-reg patterns live in regions the VM does not run from these")
        print("   entries, so the dynamic route cannot find them. Census stands,")
        print("   dynamic result stands, and the two are not in conflict.")
    elif not r_hit:
        print("   H2-ish: census sites are inside executed pages but NO *verified*")
        print("   real jmp-reg site is. So the census patterns inside executed code")
        print("   are not instruction starts -- the byte census over-counts, and the")
        print("   dynamic zero is explained by the executed code containing no real")
        print("   register-indirect jumps.")
    else:
        print("   H2: real jmp-reg instructions WERE inside executed pages yet the")
        print("   tracer recorded none -> the detector is broken; fix it before")
        print("   trusting any dynamic result.")

    if r_hit:
        print(f"\n   example verified real sites that were executed: "
              + ", ".join(f"0x{s:X}" for s in r_hit[:6]))
    if pages:
        p = sorted(pages)
        print(f"\n   executed page range inside .qvm0: "
              f"0x{p[0]-va:X}..0x{p[-1]-va:X} (section offsets)")


if __name__ == "__main__":
    main(sys.argv[1])
