"""Edge-set assembly, validation and export.

Every stage of this toolkit produces control-flow edges of a different confidence,
so they are merged here under one schema and validated the same way:

  idiom    -- resolved from the `call $+5; …; add reg,imm; jmp reg` construct
  slice    -- recovered by backward slicing with MBA folding
  dynamic  -- observed by emulation (ground truth for the run, under that input)

Validation is not optional. A resolved *constant* is not the same thing as a real
*edge*, and this module exists mostly to keep those apart:

  * a target that equals the branch's own fall-through carries no control flow, so it
    is dropped (it is anti-disassembly padding);
  * a target that is not a decoded instruction boundary is suspect, so it is flagged
    rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import deobf, functions as func_mod, slice as slice_mod
from .model import Edge
from .pe import PEMap


@dataclass
class EdgeSet:
    edges: list[Edge] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def merged(self) -> list[Edge]:
        return self.edges

    def by_stage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.edges:
            out[e.kind] = out.get(e.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def discover_static_edges(pm: PEMap, vm_section: str = ".qvm0",
                          window: int = 4096, max_anchors: int = 8,
                          max_instrs: int = 512, progress=None
                          ) -> tuple[list[Edge], dict]:
    """Run the full static pipeline: idiom resolution, then bounded slicing."""
    sec = pm.section_named(vm_section)
    if sec is None:
        return [], {}
    buf = pm.section_bytes(sec)
    lo, hi = pm.base + sec.va, pm.base + sec.va + sec.vsize
    funcs = func_mod.recover_functions(pm)

    d = deobf.resolve_hidden_branches(buf, lo, lo, hi)
    idiom_real = [b for b in d.branches if not b.fallthrough]
    idiom_null = [b for b in d.branches if b.fallthrough]
    if progress:
        progress(f"idiom: resolved={len(d.branches)} real={len(idiom_real)} "
                 f"null={len(idiom_null)}")

    handled = {b.jmp_site for b in d.branches}
    all_sites = [lo + off for off, _r, _k in deobf._jmp_sites(buf)]
    unresolved = [s for s in all_sites if s not in handled]

    anchors = slice_mod.collect_anchors(pm, vm_section,
                                        extra={b.target for b in idiom_real})
    rep = slice_mod.analyse_sites(pm, vm_section, unresolved, anchors,
                                  functions=funcs, window=window,
                                  max_anchors=max_anchors, max_instrs=max_instrs)
    if progress:
        progress(f"slice: {rep.summary()['kinds']}")

    resolved = [s for s in rep.sites if s.kind == "resolved" and s.target is not None]
    # apply the same fall-through test the idiom stage uses
    real_slice = [s for s in resolved if (s.target - s.jmp_site) not in (2, 3)]
    slice_null = [s for s in resolved if (s.target - s.jmp_site) in (2, 3)]

    # deobf/slice work in VA space (they derive targets from `call $+5` return
    # addresses). `Edge` is RVA project-wide, so rebase here -- this is the single
    # point where the two conventions meet, and getting it wrong yields
    # double-based addresses that still *look* like plausible numbers.
    edges = [Edge(site=b.jmp_site - pm.base, target=b.target - pm.base,
                  caller=0, kind="idiom") for b in idiom_real]
    edges += [Edge(site=s.jmp_site - pm.base, target=s.target - pm.base,
                   caller=0, kind="slice") for s in real_slice]

    stats = {
        "vm_section": vm_section,
        "jmp_reg_sites": len(all_sites),
        "idiom_resolved": len(d.branches),
        "idiom_real": len(idiom_real),
        "idiom_null_padding": len(idiom_null),
        "slice_analysed": len(rep.sites),
        "slice_kinds": rep.summary()["kinds"],
        "slice_resolved": len(resolved),
        "slice_real": len(real_slice),
        "slice_null_padding": len(slice_null),
        "edges_total": len(edges),
        "distinct_targets": len({e.target for e in edges}),
    }
    return edges, stats


def validate_edges(pm: PEMap, edges: list[Edge], vm_section: str = ".qvm0"
                   ) -> tuple[list[Edge], dict]:
    """Drop fall-through no-ops and flag targets that are not instruction boundaries.

    Boundary confirmation uses two independent signals: the target is a known anchor
    (`call $+5` site / `lea rip` end / an already-resolved target), or decoding from
    the nearest preceding anchor lands exactly on it.
    """
    import bisect

    from capstone import CS_ARCH_X86, CS_MODE_64, Cs

    sec = pm.section_named(vm_section)
    if sec is None:
        return edges, {}
    buf = pm.section_bytes(sec)
    # RVAs throughout: `Edge` is RVA project-wide, so the containment test and the
    # anchor set are rebased here. `collect_anchors` returns VAs (it works off the
    # `call $+5` idiom), hence the subtraction.
    lo, hi = sec.va, sec.va + sec.vsize
    anchors = sorted(a - pm.base for a in slice_mod.collect_anchors(pm, vm_section))
    md = Cs(CS_ARCH_X86, CS_MODE_64)

    kept: list[Edge] = []
    confirmed = flagged = 0
    for e in edges:
        if not (lo <= e.target < hi):
            continue
        if (e.target - e.site) in (2, 3):
            continue                       # null branch, not control flow
        ok = False
        i = bisect.bisect_left(anchors, e.target)
        if i < len(anchors) and anchors[i] == e.target:
            ok = True
        else:
            j = bisect.bisect_right(anchors, e.target) - 1
            if j >= 0 and e.target - anchors[j] < 4096:
                a = anchors[j]
                off = 0
                for ins in md.disasm(buf[a - lo:e.target - lo], pm.base + a):
                    off += ins.size
                ok = (a + off == e.target)
        if ok:
            confirmed += 1
        else:
            flagged += 1
        kept.append(e)

    return kept, {
        "input_edges": len(edges),
        "kept": len(kept),
        "confirmed_boundaries": confirmed,
        "unconfirmed_boundaries": flagged,
    }


def merge(*edge_lists: Iterable[Edge]) -> list[Edge]:
    """De-duplicate on (site, target), keeping the first stage that found it."""
    seen: set[tuple[int, int]] = set()
    out: list[Edge] = []
    for lst in edge_lists:
        for e in lst:
            key = (e.site, e.target)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    out.sort(key=lambda e: (e.site, e.target))
    return out


def to_tsv(edges: list[Edge]) -> str:
    lines = ["site\ttarget\tkind\tvia_resync\tfallthrough"]
    for e in edges:
        lines.append(f"{e.site:#x}\t{e.target:#x}\t{e.kind}\t"
                     f"{int(e.via_resync)}\t{int((e.target - e.site) in (2, 3))}")
    return "\n".join(lines) + "\n"


def from_tsv(text: str, image_base: int | None = None) -> list[Edge]:
    """Parse an edges TSV. Values are RVAs; pass `image_base` to auto-rebase legacy
    VA-keyed files (see `load_dynamic_tsv` for why that trap matters)."""
    out: list[Edge] = []
    rows = []
    for line in text.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 3:
            continue
        try:
            rows.append((int(f[0], 16), int(f[1], 16), f[2]))
        except ValueError:
            continue
    rebase = _needs_rebase([(s, t) for s, t, _ in rows], image_base)
    for s, t, kind in rows:
        base = image_base if rebase else 0
        out.append(Edge(caller=0, site=s - base, target=t - base, kind=kind))
    return out


def load_dynamic_tsv(path: str, image_base: int | None = None) -> list[Edge]:
    """Load a `dynamic_edges.tsv` (site, target, hits) as edges of kind `dynamic`.

    Runtime observation is the one source that cannot be wrong about a target it
    actually took, so these edges are worth merging into the static set rather than
    being kept in a separate report.

    **Unit handling.** `Edge.site`/`Edge.target` are RVAs project-wide (that is what
    `pm.exports()`, `functions`, `xrefs` and the Ghidra/IDA emitters all use). The
    emulator, however, works in VAs because it computes targets from register values.
    Files written by older runs are therefore VA-keyed. Passing `image_base` enables
    auto-normalisation: values that cannot be RVAs (>= image_base) are rebased. This
    exists because shipping the wrong unit produced double-based addresses in the
    generated Ghidra script (`0x3002a221d` instead of `0x1802a221d`) -- an
    in-image-looking number that would silently create nothing.
    """
    out: list[Edge] = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if "target" not in header:
            raise ValueError(f"{path} does not look like a dynamic_edges.tsv")
        rows = []
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            try:
                rows.append((int(p[0], 16), int(p[1], 16)))
            except ValueError:
                continue
    rebase = _needs_rebase(rows, image_base)
    for s, t in rows:
        base = image_base if rebase else 0
        out.append(Edge(caller=0, site=s - base, target=t - base, kind="dynamic"))
    return out


def _needs_rebase(rows: list[tuple[int, int]], image_base: int | None) -> bool:
    """True if the values look like VAs rather than RVAs for this image base.

    The decision is taken on the **site** column only. A site is the address of an
    instruction that actually executed, so it is structurally guaranteed to be inside
    the image; targets are not -- junk paths in the emulator produce occasional wild
    targets below the image base, and a min-over-everything test is defeated by a
    single one of them (which is exactly what happened: one target of 0x156e1b2b6
    suppressed rebasing for the whole 3,454-edge file, and every dynamic edge was then
    silently discarded as out-of-section).
    """
    if not image_base or not rows:
        return False
    return min(site for site, _ in rows) >= image_base


def coverage_report(static_edges: list[Edge], dynamic_edges: list[Edge]) -> dict:
    """How much do the two independent sources agree, and what does each add?"""
    s = {(e.site, e.target) for e in static_edges}
    d = {(e.site, e.target) for e in dynamic_edges}
    return {
        "static_edges": len(s),
        "dynamic_edges": len(d),
        "agreed": len(s & d),
        "static_only": len(s - d),
        "dynamic_only": len(d - s),
        "merged": len(s | d),
        "static_sites": len({x for x, _ in s}),
        "dynamic_sites": len({x for x, _ in d}),
        "dynamically_dispatched_sites": len(
            {x for x in {a for a, _ in d}
             if len({b for a, b in d if a == x}) > 1}),
    }
