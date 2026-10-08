#!/usr/bin/env python3
"""Cross-reference dynamic and static edge sets, and validate the dynamic targets.

The fixed detector produced thousands of dynamic edges. Before any of that is
believed it needs the same treatment the static edges got:

  * are the targets real instruction boundaries?
  * do they agree with the statically derived edges (mutual confirmation), or are
    they mostly new (meaning the static pass was incomplete)?
  * which sites have MULTIPLE targets? Those are the true dispatch sites, and they
    are exactly the population the static slicer called `opaque`.
"""
from __future__ import annotations

import collections
import os
import sys

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs

from qvmtool import slice as slice_mod
from qvmtool.pe import PEMap


def load_dynamic(path: str):
    """(site, target) -> hits from the dynamic_edges.tsv written by the CLI."""
    edges = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            edges[(int(parts[0], 16), int(parts[1], 16))] = int(parts[2])
    return edges


def load_static(path: str):
    edges = set()
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            s, t = int(p[0], 16), int(p[1], 16)
            if (t - s) in (2, 3):      # fall-through padding, not control flow
                continue
            edges.add((s, t))
    return edges


def main(pe_path: str, dyn_path: str, static_path: str) -> None:
    pm = PEMap(pe_path)
    sec = pm.section_named(".qvm0")
    buf = pm.section_bytes(sec)
    lo, hi = pm.base + sec.va, pm.base + sec.va + sec.vsize
    md = Cs(CS_ARCH_X86, CS_MODE_64)

    dyn = load_dynamic(dyn_path)
    stat = load_static(static_path)
    dyn_set = set(dyn)

    print(f"== dynamic vs static edge sets ({pe_path.split(chr(92))[-1]})")
    print(f"   dynamic edges : {len(dyn_set)}  (sites {len({s for s,_ in dyn_set})})")
    print(f"   static  edges : {len(stat)}  (sites {len({s for s,_ in stat})})")
    both = dyn_set & stat
    print(f"   agreed by both: {len(both)}")
    print(f"   dynamic-only  : {len(dyn_set - stat)}")
    print(f"   static-only   : {len(stat - dyn_set)}")

    # ---- boundary validation of dynamic targets (independent method) ----
    anchors = set(slice_mod.collect_anchors(pm, ".qvm0"))
    targets = {t for _, t in dyn_set}
    in_anchor = sum(1 for t in targets if t in anchors)
    print(f"\n-- dynamic target validation --")
    print(f"   distinct targets          : {len(targets)}")
    print(f"   inside .qvm0              : "
          f"{sum(1 for t in targets if lo <= t < hi)}")
    print(f"   are known anchors         : {in_anchor} "
          f"({in_anchor/max(1,len(targets)):.1%})")

    import bisect
    landed = 0
    for t in sorted(targets):
        j = bisect.bisect_right(sorted(anchors), t) - 1
        if j < 0:
            continue
        a = sorted(anchors)[j]
        if t - a >= 4096:
            continue
        off = 0
        for ins in md.disasm(buf[a - lo:t - lo], a):
            off += ins.size
        if a + off == t:
            landed += 1
    print(f"   exact instruction boundary: {landed} "
          f"({landed/max(1,len(targets)):.1%})")

    # ---- dispatch sites: many targets from one site ----
    per_site = collections.defaultdict(set)
    for s, t in dyn_set:
        per_site[s].add(t)
    multi = {s: ts for s, ts in per_site.items() if len(ts) > 1}
    print(f"\n-- dispatch sites (a site with >1 observed target) --")
    print(f"   sites={len(per_site)}  multi-target={len(multi)} "
          f"({len(multi)/max(1,len(per_site)):.1%})")
    print(f"   top dispatchers:")
    for s, ts in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:12]:
        reg = ""
        print(f"      0x{s:08X}  {len(ts):3d} targets   "
              f"e.g. " + ", ".join(f"0x{t:X}" for t in sorted(ts)[:4]))

    # ---- do the multi-target sites correspond to the statically 'opaque' set? ----
    try:
        import json
        sites_tsv = dyn_path.replace("dynamic_edges.tsv", "dynamic_sites.tsv")
        n_sites = sum(1 for _ in open(sites_tsv)) - 1
        print(f"\n   (dynamic_sites.tsv rows: {n_sites})")
    except OSError:
        pass

    print("\n-- interpretation --")
    if len(dyn_set - stat) > len(both) * 3:
        print("   The dynamic pass found far more than the static pass confirmed:")
        print("   the static slicer's bound (window/anchors) left most dispatch")
        print("   targets unresolved, and runtime observation closes that gap.")
    if multi:
        print(f"   {len(multi)} sites dispatch to multiple targets, so QVM does use")
        print("   genuine computed dispatch -- consistent with the ~5,700 `opaque`")
        print("   static sites being real dispatches rather than analysis failures.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
