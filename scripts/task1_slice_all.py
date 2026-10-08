#!/usr/bin/env python3
"""Task 1 driver: full backward-slice pass over the unresolved indirect branches.

Writes a per-site classification plus a partial CFG so the result is auditable
rather than a single headline number.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qvmtool import deobf, functions as fn, slice as sl
from qvmtool.pe import PEMap


def main(pe_path: str, out_prefix: str) -> None:
    pm = PEMap(pe_path)
    sec = pm.section_named(".qvm0")
    buf = pm.section_bytes(sec)
    lo, hi = pm.base + sec.va, pm.base + sec.va + sec.vsize
    funcs = fn.recover_functions(pm)

    t0 = time.time()
    d = deobf.resolve_hidden_branches(buf, lo, lo, hi)
    # A resolved `selfpc` idiom whose target is the instruction after its own jmp is
    # semantically null: it is anti-disassembly padding, NOT a control-flow edge.
    real_idiom = [b for b in d.branches if not b.fallthrough]
    padding = [b for b in d.branches if b.fallthrough]
    idiom_sites = {b.jmp_site for b in d.branches}
    all_sites = [lo + off for off, _r, _k in deobf._jmp_sites(buf)]
    unresolved = [s for s in all_sites if s not in idiom_sites]
    anchors = sl.collect_anchors(pm, ".qvm0", extra={b.target for b in real_idiom})
    print(f"jmp sites={len(all_sites)} idiom_resolved={len(d.branches)} "
          f"(real={len(real_idiom)} padding={len(padding)}) "
          f"unresolved={len(unresolved)} anchors={len(anchors)}", flush=True)
    print(f"[{time.time()-t0:.1f}s] slice setup done", flush=True)

    rep = sl.analyse_sites(pm, ".qvm0", unresolved, anchors, functions=funcs,
                           window=4096, max_anchors=8, max_instrs=512)
    print(f"[{time.time()-t0:.1f}s] slicing done", flush=True)

    # ---- combine both stages into one edge set (padding excluded) ----
    edges = []
    for b in real_idiom:
        edges.append({"site": b.jmp_site, "target": b.target, "stage": "idiom",
                      "form": b.form, "confidence": b.confidence})
    for s in rep.sites:
        if s.kind == "resolved" and s.target is not None:
            edges.append({"site": s.jmp_site, "target": s.target, "stage": "slice",
                          "form": "mba-fold", "confidence": s.confidence})

    kinds = Counter(s.kind for s in rep.sites)
    # base quality: is the recovered base a plausible local code address?
    base_deltas = []
    for s in rep.sites:
        if s.kind == "range" and s.base and lo <= s.base < hi:
            base_deltas.append(s.base - s.jmp_site)

    summary = {
        "file": pe_path,
        "vm_section": ".qvm0",
        "jmp_reg_sites": len(all_sites),
        "idiom_resolved": len(d.branches),
        "idiom_real_edges": len(real_idiom),
        "idiom_null_padding": len(padding),
        "slice_targets_analysed": len(rep.sites),
        "slice_kinds": dict(kinds),
        "total_resolved_edges": len(edges),
        "total_distinct_targets": len({e["target"] for e in edges}),
        "base_recovered_sites": len(base_deltas),
        "base_delta_min": min(base_deltas) if base_deltas else None,
        "base_delta_max": max(base_deltas) if base_deltas else None,
        "base_delta_median": (sorted(base_deltas)[len(base_deltas) // 2]
                              if base_deltas else None),
        "coverage": round(len(edges) / len(all_sites), 4) if all_sites else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_prefix + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_prefix + ".edges.tsv", "w") as f:
        f.write("site\ttarget\tstage\tform\tconfidence\n")
        for e in sorted(edges, key=lambda x: x["site"]):
            f.write(f"{e['site']:#x}\t{e['target']:#x}\t{e['stage']}\t"
                    f"{e['form']}\t{e['confidence']}\n")
    with open(out_prefix + ".sites.tsv", "w") as f:
        f.write("jmp_site\tregister\tkind\tbase\tanchor\tinstrs\tconfidence\texpr\n")
        for s in rep.sites:
            f.write(f"{s.jmp_site:#x}\t{s.register}\t{s.kind}\t"
                    f"{(hex(s.base) if s.base else '-')}\t{s.anchor:#x}\t{s.instrs}\t"
                    f"{s.confidence}\t{s.expr}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
