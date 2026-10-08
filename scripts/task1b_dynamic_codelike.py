#!/usr/bin/env python3
"""Task 1, retargeted: run dynamic edge recovery only from *code-like* VM entries.

Why the first attempt produced nothing
-------------------------------------
The first dynamic sweep entered from every native caller of the VM section and ran
3.14M instructions for zero observed indirect jumps. The measurement in
`verify_target_decodability.py` explains it: 182 of the 263 distinct native->VM
targets decode at a median of 20 instructions/256B while known-good code decodes at
67. Emulating those entries means executing non-code bytes, which yields junk paths
and no real control flow.

So restrict the sweep to the sub-population whose entry actually decodes as code
(>=40 insns/256B). If the dynamic route works at all, it works here.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs

from qvmtool import emulate as emu, functions as fn, xrefs as xr
from qvmtool.pe import PEMap

WINDOW = 256
CODE_THRESHOLD = 40


def main(path: str, out_json: str) -> None:
    pm = PEMap(path)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    sec = pm.section_named(".qvm0")

    def density(rva: int) -> int:
        raw = pm.read_rva(rva, WINDOW)
        if not raw or len(raw) < WINDOW:
            return 0
        return sum(1 for _ in md.disasm(raw, pm.base + rva))

    funcs = fn.recover_functions(pm)
    edges = xr.native_to_vm_edges(pm, funcs, ".qvm0")

    code_targets = {t for t in {e.target for e in edges} if density(t) >= CODE_THRESHOLD}
    # entry = the NATIVE CALLER, so the VM entry inherits a real context
    callers = [c for c, n in Counter(
        e.caller for e in edges if e.target in code_targets).most_common()]
    total_callers = len({e.caller for e in edges})
    print(f"code-like VM targets: {len(code_targets)} / "
          f"{len({e.target for e in edges})}")
    print(f"their native callers: {len(callers)} / {total_callers}")

    res = emu.recover_dynamic_edges(
        pm, ".qvm0", entry_rvas=callers[:120], budget=300_000, max_entries=120,
        verbose=False)
    summary = res.summary()
    print(json.dumps(summary, indent=2))

    with open(out_json, "w") as f:
        json.dump({"summary": summary,
                   "code_like_targets": sorted(code_targets),
                   "entries_used": callers[:120],
                   "edges": [{"site": hex(s), "target": hex(t), "hits": c}
                             for (s, t), c in sorted(res.edges.items())]}, f, indent=2)
    print(f"wrote {out_json}")

    if res.edges:
        print("\ntop observed dynamic edges:")
        for (s, t), c in res.edges.most_common(15):
            print(f"   0x{s:X} -> 0x{t:X}   x{c}")
    else:
        print("\nstill zero dynamic edges even from code-like entries.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
