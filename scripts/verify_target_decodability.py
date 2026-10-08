#!/usr/bin/env python3
"""Verify the claim: "native->VM stub targets are not executable code in the file".

My earlier conclusion rested on three spot checks (0x2A2000 good, 0x989357 and
0x994437 bad). Three points is not a measurement, and the claim is load-bearing --
it is what blocks the dynamic route -- so it needs:

  * a CONTROL: known-good native code, whose decode density tells us what "code"
    actually looks like in this binary;
  * the FULL population: every one of the recovered native->VM edge targets, not a
    hand-picked handful;
  * an explicit threshold, so the classification is reproducible.

Metric: instructions decoded per 256 bytes from the exact target. Real x86-64
averages ~4.5 bytes/instruction, i.e. ~55/256B; non-code bytes average far longer
because the decoder keeps absorbing garbage into long instructions.
"""
from __future__ import annotations

import collections
import statistics
import os
import sys

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs

from qvmtool import functions as fn, xrefs as xr
from qvmtool.pe import PEMap

WINDOW = 256
#: below this the bytes are almost certainly not x86-64 code (control median is ~57)
CODE_THRESHOLD = 40


def density(md, buf: bytes, va: int) -> int:
    return sum(1 for _ in md.disasm(buf, va))


def main(path: str) -> None:
    pm = PEMap(path)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    sec = pm.section_named(".qvm0")
    qlo, qhi = sec.va, sec.va + sec.vsize

    def sample_at(rva: int):
        raw = pm.read_rva(rva, WINDOW)
        return None if not raw or len(raw) < WINDOW else density(md, raw, pm.base + rva)

    funcs = fn.recover_functions(pm)
    edges = xr.native_to_vm_edges(pm, funcs, ".qvm0")
    targets = sorted({e.target for e in edges})

    # ---- control 1: known-good native code (.text function entries) ----
    text_funcs = [f for f in funcs if f.section == ".text" and f.size >= WINDOW]
    step = max(1, len(text_funcs) // 300)
    control = [d for d in (sample_at(f.begin) for f in text_funcs[::step]) if d]

    # ---- control 2: the .qvm0 section start (previously confirmed coherent) ----
    entry0 = sample_at(qlo)

    # ---- population: every native->VM edge target ----
    pop = {}
    for t in targets:
        d = sample_at(t)
        if d is not None:
            pop[t] = d

    vals = sorted(pop.values())
    good = [t for t, d in pop.items() if d >= CODE_THRESHOLD]
    bad = [t for t, d in pop.items() if d < CODE_THRESHOLD]

    print(f"== decode-density verification: {path}")
    print(f"   metric: instructions decoded in {WINDOW} bytes from the target")
    print(f"   threshold for 'is code': >= {CODE_THRESHOLD}\n")
    print("-- CONTROL: .text function entries (known-good native code) --")
    print(f"   n={len(control)}  min={min(control)}  median={statistics.median(control):.0f}"
          f"  max={max(control)}")
    print(f"   fraction >= {CODE_THRESHOLD}: "
          f"{sum(1 for d in control if d >= CODE_THRESHOLD)/len(control):.1%}")
    print(f"-- CONTROL: .qvm0 section start 0x{qlo:X}: {entry0} insns/{WINDOW}B --\n")

    print("-- POPULATION: native -> .qvm0 edge targets --")
    print(f"   edges={len(edges)}  distinct targets={len(targets)}  measured={len(pop)}")
    if vals:
        print(f"   min={vals[0]}  median={statistics.median(vals):.0f}  max={vals[-1]}")
    print(f"   >= {CODE_THRESHOLD} (code-like) : {len(good):4d}  "
          f"({len(good)/max(1,len(pop)):.1%})")
    print(f"   <  {CODE_THRESHOLD} (not code)  : {len(bad):4d}  "
          f"({len(bad)/max(1,len(pop)):.1%})")

    hist = collections.Counter((v // 10) * 10 for v in vals)
    print("\n   histogram (bucket: count)")
    for b in sorted(hist):
        bar = "#" * min(60, hist[b] // max(1, len(vals) // 60 or 1))
        print(f"     {b:3d}-{b+9:<3d} {hist[b]:5d}  {bar}")

    # Does target density correlate with which band it lands in?
    print("\n   sample of the worst targets (lowest density):")
    for t in sorted(bad, key=lambda x: pop[x])[:10]:
        print(f"     0x{t:08X}  {pop[t]:3d} insns/256B   in .qvm0 offset 0x{t-qlo:X}")

    print("\n   interpretation:")
    ctrl_med = statistics.median(control)
    pop_med = statistics.median(vals) if vals else 0
    print(f"     control median {ctrl_med:.0f} vs population median {pop_med:.0f}")
    if pop_med < CODE_THRESHOLD <= ctrl_med:
        print("     => the majority of native->VM targets are NOT decodable as code,")
        print("        while the control population is. The claim holds at population")
        print("        scale, not just for the spot-checked examples.")
    elif pop_med >= ctrl_med * 0.7:
        print("     => the targets decode comparably to real code. The earlier claim")
        print("        was WRONG and rested on unrepresentative examples.")
    else:
        print("     => intermediate: the population is mixed. Report the two")
        print("        sub-populations separately rather than generalising.")

    # ---- do the good targets cluster in a region? ----
    if good:
        offs = sorted(t - qlo for t in good)
        print(f"\n   code-like targets span .qvm0 offsets 0x{offs[0]:X}..0x{offs[-1]:X}")
        bands = collections.Counter((o // 0x100000) * 0x100000 for o in offs)
        print("   code-like targets per 1 MB band: " +
              ", ".join(f"0x{k:X}:{v}" for k, v in sorted(bands.items())))


if __name__ == "__main__":
    main(sys.argv[1])
