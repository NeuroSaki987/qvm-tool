#!/usr/bin/env python3
"""Coverage of the statically `opaque` sites by the dynamic pass.

Objective item 1 asks specifically about the ~5,725 sites the static slicer could not
resolve. The dynamic sweep is the attack on them, so the honest question is not "how
many edges did we get" but **"of the opaque sites, how many did runtime observation
actually reach and resolve?"** This computes exactly that, and separates the two ways
an opaque site can end up resolved:

  resolved     -- observed executing, with a target we captured
  multi-target -- observed dispatching to MORE THAN ONE target (a true dispatch site)
  unobserved   -- never executed from any of the emulated entries

A site that is `opaque` statically and single-target dynamically is worth noting: it
may simply have been taken only once under these inputs.
"""
from __future__ import annotations

import collections
import os
import sys


def load_static_sites(path: str) -> dict[int, str]:
    """jmp_site -> kind, from slice.sites.tsv."""
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_site = header.index("jmp_site")
            i_kind = header.index("kind")
        except ValueError:
            raise SystemExit(f"unexpected header in {path}: {header}")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(i_site, i_kind):
                continue
            try:
                out[int(p[i_site], 16)] = p[i_kind]
            except ValueError:
                continue
    return out


def load_dynamic(path: str) -> tuple[set[int], dict[int, set[int]], int]:
    sites: set[int] = set()
    per: dict[int, set[int]] = collections.defaultdict(set)
    n = 0
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            try:
                s, t = int(p[0], 16), int(p[1], 16)
            except ValueError:
                continue
            sites.add(s)
            per[s].add(t)
            n += 1
    return sites, per, n


def main(sites_tsv: str, dyn_tsv: str, dyn_label: str) -> None:
    kinds = load_static_sites(sites_tsv)
    by_kind = collections.Counter(kinds.values())
    dyn_sites, per_site, dyn_edges = load_dynamic(dyn_tsv)

    opaque = {s for s, k in kinds.items() if k == "opaque"}
    resolved_static = {s for s, k in kinds.items() if k == "resolved"}
    padded = {s for s, k in kinds.items() if k not in ("opaque", "resolved")}

    hit = opaque & dyn_sites
    multi = {s for s in hit if len(per_site[s]) > 1}

    print(f"== coverage of statically-opaque sites by dynamic observation")
    print(f"   dynamic source: {dyn_label}  ({dyn_edges} edges, "
          f"{len(dyn_sites)} sites)\n")
    print(f"-- static slice population --")
    for k, v in by_kind.most_common():
        print(f"   {k:<10} {v}")
    print(f"   total      {len(kinds)}\n")

    print(f"-- the {len(opaque)} opaque sites --")
    print(f"   observed executing at least once : {len(hit):>5}  "
          f"({len(hit)/max(1,len(opaque)):.1%})")
    print(f"     of those, multi-target dispatch: {len(multi):>5}  "
          f"({len(multi)/max(1,len(opaque)):.1%})")
    print(f"   never observed                   : {len(opaque)-len(hit):>5}  "
          f"({(len(opaque)-len(hit))/max(1,len(opaque)):.1%})")

    print(f"\n-- cross-check: were statically-resolved sites also observed? --")
    rs_hit = resolved_static & dyn_sites
    print(f"   static-resolved {len(resolved_static)}, also observed dynamically: "
          f"{len(rs_hit)} ({len(rs_hit)/max(1,len(resolved_static)):.1%})")

    print(f"\n-- sites observed dynamically that the static pass never listed --")
    unknown = dyn_sites - set(kinds)
    print(f"   {len(unknown)} of {len(dyn_sites)} dynamic sites "
          f"({len(unknown)/max(1,len(dyn_sites)):.1%}) are NOT in the static site list")
    print("   (the static pass only enumerates byte-pattern jmp-reg candidates; a")
    print("    dynamic site outside that list means the census itself is incomplete)")

    print(f"\n-- verdict --")
    cov = len(hit) / max(1, len(opaque))
    if cov >= 0.5:
        print(f"   Dynamic observation reached {cov:.0%} of the opaque sites, so the")
        print("   'opaque' class is largely a static-analysis bound, not a property of")
        print("   the code. Runtime is the effective resolver for this class.")
    elif cov > 0:
        print(f"   Dynamic observation reached only {cov:.0%} of the opaque sites.")
        print("   Coverage is the binding constraint: more entries / longer budgets")
        print("   would be needed before concluding anything about the rest.")
    else:
        print("   No opaque site was reached. Either they are unreachable from the")
        print("   emulated entries, or the entries/inputs are unrepresentative.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "dynamic")
