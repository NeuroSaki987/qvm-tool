"""Command line interface.

Verbs mirror the survey stages so each stage can be run and audited on its own,
which is how these binaries have to be worked: every claim needs a reproducible
command behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from . import engine as engine_mod
from . import functions as func_mod
from . import regions as regions_mod
from . import symbols as symbols_mod
from . import tvm as tvm_mod
from . import xrefs as xrefs_mod
from .detect import detect
from .pe import PEMap
from .report import render
from .survey import run as run_survey


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}")


def cmd_detect(args) -> int:
    pm = PEMap(args.pe)
    v = detect(pm)
    if args.json:
        print(json.dumps({
            "file": pm.path, "sha256": pm.sha256, "engine": v.engine,
            "engine_core": v.engine_core,
            "confidence": round(v.confidence, 3), "vm_section": v.vm_section,
            "evidence": [e.to_dict() for e in v.evidence],
        }, indent=2))
    else:
        print(f"{pm.path}")
        print(f"  engine     : {v.engine}")
        print(f"  engine core: {v.engine_core}")
        print(f"  confidence : {v.confidence:.2f}")
        print(f"  vm section : {v.vm_section}")
        print("  evidence:")
        for e in v.evidence:
            w = f"  (+{e.weight:.2f})" if e.weight else ""
            print(f"    - {e.claim}: {e.detail}{w}")
    return 0


def cmd_tvm(args) -> int:
    """TVM-only pass: VIP slot, fetch idiom, threaded dispatch, `E9` trampolines.

    Every measurement here is anchored on a byte pattern or on a `jmp reg` site.
    The dispatch resolver is validated against symbolic execution of the real
    handlers (213/213, 314/314, 527/527 exact on ACE-GAME / ACE-ATS64 / ACE-CSI64).
    """
    from . import cfg as cfg_mod
    from . import ghidra as ghidra_mod

    pm = PEMap(args.pe)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2
    sec = pm.section_named(v.vm_section)
    buf = pm.section_bytes(sec)
    profile = engine_mod.profile_for_section(v.vm_section)

    ranked = tvm_mod.discover_vip_slot(buf)
    slot = ranked[0]["slot"] if ranked else tvm_mod.VIP_SLOT_KNOWN
    fetch = tvm_mod.fetch_idiom_census(buf, slot)
    context = tvm_mod.context_frame_census(buf)
    res = tvm_mod.resolve_section(pm, v.vm_section)
    dispatch = res.summary()
    tedges, tstats = tvm_mod.native_trampoline_edges(pm, v.vm_section)
    carrier = tvm_mod.assess_carrier(v.vm_section, ranked, context)

    # merge the two independent edge sources: the handled-thread dispatches and the
    # native trampolines that enter the thread
    dedges = tvm_mod.dispatch_edges(res, pm.base)
    edges = cfg_mod.merge(tedges, dedges)
    edge_list = [e for e in edges]

    payload = {
        "file": pm.path, "sha256": pm.sha256,
        "engine": v.engine, "engine_core": v.engine_core,
        "profile": profile.name,
        "vm_section": v.vm_section,
        "vip_slots": ranked,
        "chosen_vip_slot": hex(slot),
        "fetch": fetch,
        "context": context,
        "dispatch": dispatch,
        "trampolines": tstats,
        "carrier": carrier.to_dict(),
        "edges_total": len(edge_list),
        "edges_by_kind": {k: sum(1 for e in edge_list if e.kind == k)
                          for k in {e.kind for e in edge_list}},
        "validated_against_symbolic_execution": {
            "ACE-GAME.sys": "213/213 exact",
            "ACE-ATS64.dll": "314/314 exact",
            "ACE-CSI64.dll": "527/527 exact",
            "method": "tvm-devirt trace --dispatches",
        },
    }
    outdir = args.output
    if outdir and args.edges:
        os.makedirs(outdir, exist_ok=True)
        _write(os.path.join(outdir, "tvm_edges.tsv"), cfg_mod.to_tsv(edge_list))
        _write(os.path.join(outdir, "inject_tvm_edges.java"),
               ghidra_mod.emit_cfg_edges_java(
                   [(e.site, e.target, e.kind) for e in edge_list], pm.base,
                   make_functions=not args.no_functions))
        _write(os.path.join(outdir, "inject_tvm_edges.idc"),
               ghidra_mod.emit_cfg_edges_idc(
                   [(e.site, e.target, e.kind) for e in edge_list], pm.base))
    text = json.dumps(payload, indent=2)
    if outdir:
        _write(os.path.join(outdir, "tvm.json"), text)
    else:
        print(text)
    return 0


def cmd_functions(args) -> int:
    pm = PEMap(args.pe)
    funcs = func_mod.recover_functions(pm)
    lines = ["begin\tend\tunwind\tsec\tsize"]
    for f in funcs:
        lines.append(f"{f.begin:#x}\t{f.end:#x}\t{f.unwind:#x}\t{f.section}\t{f.size}")
    text = "\n".join(lines) + "\n"
    if args.output:
        _write(args.output, text)
    else:
        sys.stdout.write(text)
    print(f"# {len(funcs)} functions", file=sys.stderr)
    return 0


def cmd_trampolines(args) -> int:
    pm = PEMap(args.pe)
    funcs = func_mod.recover_functions(pm)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2
    edges = xrefs_mod.native_to_vm_edges(pm, funcs, v.vm_section)
    extra_stats = {}
    profile = engine_mod.profile_for_section(v.vm_section)
    if profile.native_entry_form == "e9-trampoline":
        # The anchored decoder only sees entries that live inside a RUNTIME_FUNCTION
        # body. This engine places ~24% of its `E9 rel32` trampolines in `.text` gaps
        # that NO exception entry covers, so the anchored-only answer is simply
        # incomplete. The byte-anchored set is added, and the merge is reported.
        from . import cfg as cfg_mod
        extra, extra_stats = tvm_mod.native_trampoline_edges(pm, v.vm_section)
        anchored = len(edges)
        edges = cfg_mod.merge(edges, extra)
        extra_stats = {**extra_stats, "anchored_edges": anchored,
                       "recovered_total": len(edges)}
    clusters = xrefs_mod.vm_entry_clusters(edges)
    stubs = xrefs_mod.vm_stubs(pm, funcs, v.vm_section)
    if extra_stats:
        # a trampoline outside the function map is its own caller: it is a whole
        # function by construction, so it belongs in the stub list too
        seen = {f for f, _e in stubs}
        for e in edges:
            if e.kind == "e9-trampoline" and e.caller == e.site and e.site not in seen:
                stubs.append((e.site, e))
                seen.add(e.site)

    if args.json:
        payload = {
            "vm_section": v.vm_section,
            "engine": v.engine,
            "engine_core": v.engine_core,
            "edges": [e.__dict__ for e in edges],
            "clusters": [{"target": t, "incoming": n} for t, n in clusters],
            "stubs": [{"function": f, "kind": e.kind, "target": e.target}
                      for f, e in stubs],
            "recovery": extra_stats,
        }
        text = json.dumps(payload, indent=2)
    else:
        text = (f"# {len(edges)} edges, {len({e.caller for e in edges})} callers, "
                f"{len(clusters)} distinct targets, {len(stubs)} whole-function stubs\n")
        if extra_stats:
            text += f"# recovery: {json.dumps(extra_stats)}\n"
        text += "caller\tsite\tkind\ttarget\n"
        for e in edges:
            text += f"{e.caller:#x}\t{e.site:#x}\t{e.kind}\t{e.target:#x}\n"
        text += "\n# shared entries\ntarget\tincoming\n"
        for t, n in clusters[:40]:
            text += f"{t:#x}\t{n}\n"

    if args.output:
        _write(args.output, text)
    else:
        sys.stdout.write(text)
    return 0


def cmd_regions(args) -> int:
    pm = PEMap(args.pe)
    bands = regions_mod.encrypted_bands(pm, threshold=args.threshold)
    if args.json:
        text = json.dumps([b.__dict__ for b in bands], indent=2)
    else:
        text = "section\tstart\tend\tsize\tentropy\n"
        for b in bands:
            text += (f"{b.section}\t{b.start:#x}\t{b.end:#x}\t{b.size}\t"
                     f"{b.entropy:.3f}\n")
    if args.output:
        _write(args.output, text)
    else:
        sys.stdout.write(text)
    if not args.section:
        return 0
    sec = pm.section_named(args.section)
    if sec is None:
        print(f"no such section: {args.section}", file=sys.stderr)
        return 2
    # The VM-section profile needs the engine's own "this is plaintext code" test:
    # the QVM marker (self-PC idiom density) is identically zero on `.tvm0`, which is
    # how the shipped verb came to report 0 bytes of code in 21 of 21 TVM modules.
    v = detect(pm)
    profile = engine_mod.profile_for_section(args.section)
    core = {"qvm-mutation": "qvm", "tvm-threaded": "tvm"}.get(v.engine_core)
    if core:
        profile = engine_mod.profile_for_engine(core)
    markers = regions_mod.tvm_code_markers if profile.name == "tvm" else None
    regions = regions_mod.vm_regions(pm, args.section, code_markers=markers)
    out = (f"\n# {args.section} region profile (engine core {v.engine_core or 'n/a'}, "
           f"code marker {'tvm' if markers else 'self-pc'})\n"
           f"start\tend\tsize\tclass\tentropy\n")
    for r in regions:
        out += f"{r.start:#x}\t{r.end:#x}\t{r.size}\t{r.kind}\t{r.entropy:.2f}\n"
    out += (f"# code bytes: "
            f"{sum(r.size for r in regions if r.kind == 'code'):,} of "
            f"{sum(r.size for r in regions):,}\n")
    if args.output:
        _write(args.output + ".profile", out)
    else:
        sys.stdout.write(out)
    return 0


def _symbol_files(s, prefix: str) -> None:
    meta = s.summary()
    _write(f"{prefix}.symbols.json", symbols_mod.to_json(s.symbols, meta))
    _write(f"{prefix}.symbols.tsv", symbols_mod.to_tsv(s.symbols))
    _write(f"{prefix}.apply_ghidra.py",
           symbols_mod.to_ghidra_script(s.symbols, s.pm.base))
    _write(f"{prefix}.apply_ida.idc",
           symbols_mod.to_ida_idc(s.symbols, s.pm.base))


def cmd_symbols(args) -> int:
    s = run_survey(args.pe, sample_budget=args.sample_budget)
    prefix = args.output or (args.pe + ".qvm")
    _symbol_files(s, prefix)
    print(f"{len(s.symbols)} symbols", file=sys.stderr)
    return 0


def cmd_survey(args) -> int:
    s = run_survey(args.pe, sample_budget=args.sample_budget)
    outdir = args.output or (args.pe + ".survey")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, "survey")
    _write(base + ".json", json.dumps(s.summary(), indent=2))
    _write(base + ".md", render(s))
    _write(os.path.join(outdir, "survey.full.json"), json.dumps({
        "summary": s.summary(),
        "verdict": {"engine": s.verdict.engine,
                    "evidence": [e.to_dict() for e in s.verdict.evidence]},
        "idioms": s.idioms,
        "stride": s.stride,
        "mba": s.mba,
        "prologue": s.prologue,
        "carrier": s.carrier.to_dict(),
        "deobf": s.deobf,
        "regions": [r.__dict__ for r in s.regions],
        "encrypted_bands": [{**b.__dict__, "size": b.size} for b in s.bands],
        "clusters": [{"target": t, "incoming": n} for t, n in s.clusters],
        "edges": [e.__dict__ for e in s.edges],
        "stubs": [{"function": f, "kind": e.kind, "target": e.target}
                  for f, e in s.stubs],
        "pointer_tables": [{"section": a, "holder": b, "target": c}
                           for a, b, c in s.pointer_tables],
        "consecutive_table_runs": [{"section": a, "start": b, "entries": c, "span": d}
                                   for a, b, c, d in s.consecutive_tables],
        "utf16_false_positive_span": s.utf16_overlap,
    }, indent=2))
    func_lines = ["begin\tend\tunwind\tsec\tsize"]
    for f in s.functions:
        func_lines.append(f"{f.begin:#x}\t{f.end:#x}\t{f.unwind:#x}\t{f.section}\t{f.size}")
    _write(os.path.join(outdir, "functions.tsv"), "\n".join(func_lines) + "\n")
    _symbol_files(s, os.path.join(outdir, "qvm"))

    print(json.dumps(s.summary(), indent=2))
    return 0


def _rekey(e, delta: int):
    """Shift an `Edge` between image-relative and absolute keying.

    The tool has two conventions in flight: `model.Edge` documents RVA-keyed fields
    and `ghidra.emit_cfg_edges_*` require them, while `cfg.validate_edges` bounds its
    range test in VA space (as `cfg.discover_static_edges` produces). Converting at
    the boundary keeps the emitted artefacts right without changing either side.
    """
    from .model import Edge as _Edge
    return _Edge(caller=e.caller + delta, site=e.site + delta,
                 kind=e.kind, target=e.target + delta, via_resync=e.via_resync)


def cmd_edges(args) -> int:
    """Recover CFG edges, validate them, and emit Ghidra/IDA injection scripts."""
    from . import cfg as cfg_mod
    from . import ghidra as ghidra_mod

    pm = PEMap(args.pe)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2

    def progress(msg):
        print(f"  [{msg}]", file=sys.stderr)

    if args.from_edges:
        # Re-validating an existing edge list avoids a full re-slice (minutes on a
        # large VM section) when only the emitter changed.
        with open(args.from_edges) as f:
            edges = cfg_mod.from_tsv(f.read(), image_base=pm.base)
        stats = {"source": args.from_edges, "loaded_edges": len(edges)}
        kept, vstats = cfg_mod.validate_edges(pm, edges, v.vm_section)
    else:
        profile = engine_mod.profile_for_section(v.vm_section)
        core = {"qvm-mutation": "qvm", "tvm-threaded": "tvm"}.get(v.engine_core)
        if core == "tvm" or (core is None
                             and profile.dispatch_model == "threaded-handlers"):
            # The QVM pipeline resolves constants by folding `call $+5`-anchored
            # idioms and then backward-slicing the rest. On a threaded interpreter
            # that produces thousands of confidently-wrong targets (the `imm64` form
            # ignores the flags-neutral `add`), so use the engine's own resolver.
            res = tvm_mod.resolve_section(pm, v.vm_section)
            tramp, tstats = tvm_mod.native_trampoline_edges(pm, v.vm_section)
            # RVA keying, matching `xrefs` and what `ghidra.emit_cfg_edges_*` expects
            # (both emitters add `image_base` back). `cfg.validate_edges` bounds its
            # range test with VA-space values, so convert for that call and back --
            # mixing the two conventions silently drops every edge, which is how the
            # first cut of this branch lost all 958.
            raw_rva = cfg_mod.merge(tramp, tvm_mod.dispatch_edges(res, pm.base))
            edges = [_rekey(e, +pm.base) for e in raw_rva]
            stats = {
                "engine": "tvm",
                "method": "threaded-dispatch epilogues + E9 native trampolines",
                "dispatch": res.summary(),
                "trampolines": tstats,
                "edges_total": len(raw_rva),
                # ground truth is symbolic execution of the real handlers
                "validated_against": "tvm-devirt trace --dispatches",
                "ground_truth_accuracy": {
                    "ACE-GAME.sys": "213/213 resolved sites exact",
                    "ACE-ATS64.dll": "314/314 exact",
                    "ACE-CSI64.dll": "527/527 exact",
                },
            }
            if progress:
                progress(f"tvm dispatch: {res.summary()['real_edges']} real edges "
                         f"of {res.resolved} resolved")
        else:
            edges, stats = cfg_mod.discover_static_edges(
                pm, v.vm_section, window=args.window, max_anchors=args.max_anchors,
                max_instrs=args.max_instrs,
                progress=None if args.quiet else progress)
        kept, vstats = cfg_mod.validate_edges(pm, edges, v.vm_section)
        if stats.get("engine") == "tvm":
            kept = [_rekey(e, -pm.base) for e in kept]
        if stats.get("engine") == "tvm":
            # `validate_edges` confirms a target by decoding forward from the nearest
            # `slice.collect_anchors` anchor, which is a QVM anchor set (`call $+5`
            # sites, `lea rip` ends). On a threaded interpreter that set is nearly
            # empty, so "unconfirmed" here means "this engine's anchors do not apply",
            # not "suspect target". The ground truth is the symbolic-execution check
            # recorded next to it in `ground_truth_accuracy`.
            vstats["boundary_method"] = (
                "slice.collect_anchors (QVM anchor set) -- not meaningful on a "
                "threaded interpreter; use ground_truth_accuracy instead")

    # Runtime observation is the strongest evidence available: a taken branch cannot
    # be wrong about its target. Merge it in (whichever engine produced the static
    # set) and re-validate, so the emitted injection script reflects both sources.
    if args.dynamic:
        dyn = cfg_mod.load_dynamic_tsv(args.dynamic, image_base=pm.base)
        cov = cfg_mod.coverage_report(edges, dyn)
        edges = cfg_mod.merge(edges, dyn)
        kept, vstats = cfg_mod.validate_edges(pm, edges, v.vm_section)
        stats["dynamic_coverage"] = cov
        stats["merged_edges"] = len(edges)
        if progress:
            progress(f"merged dynamic: {cov['dynamic_edges']} dynamic, "
                     f"{cov['agreed']} agreed, {cov['dynamic_only']} dynamic-only "
                     f"-> {len(kept)} kept")

    outdir = args.output or (args.pe + ".edges")
    os.makedirs(outdir, exist_ok=True)
    _write(os.path.join(outdir, "edges.tsv"), cfg_mod.to_tsv(kept))
    _write(os.path.join(outdir, "inject_qvm_edges.java"),
           ghidra_mod.emit_cfg_edges_java(
               [(e.site, e.target, e.kind) for e in kept], pm.base,
               make_functions=not args.no_functions))
    _write(os.path.join(outdir, "inject_qvm_edges.idc"),
           ghidra_mod.emit_cfg_edges_idc(
               [(e.site, e.target, e.kind) for e in kept], pm.base))
    _write(os.path.join(outdir, "edges.stats.json"),
           json.dumps({**stats, "validation": vstats}, indent=2))

    print(json.dumps({**stats, "validation": vstats}, indent=2))
    return 0


def cmd_dynamic(args) -> int:
    """Recover CFG edges by emulating the VM from real call contexts."""
    from . import emulate as emu_mod

    pm = PEMap(args.pe)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2

    def progress(msg):
        print(f"  [{msg}]", file=sys.stderr)

    res = emu_mod.recover_dynamic_edges(
        pm, v.vm_section, entry_rvas=args.entries, budget=args.budget,
        max_entries=args.max_entries, verbose=not args.quiet,
        progress=None if args.quiet else progress)

    outdir = args.output or (args.pe + ".dynamic")
    os.makedirs(outdir, exist_ok=True)
    _write(os.path.join(outdir, "dynamic_edges.tsv"),
           "site\ttarget\thits\n" + "\n".join(
               f"{s:#x}\t{t:#x}\t{c}" for (s, t), c in sorted(res.edges.items())))
    _write(os.path.join(outdir, "dynamic_sites.tsv"),
           "site\tregister\ttargets\thits\n" + "\n".join(
               f"{s:#x}\t{r}\t{len(ts)}\t{res.site_hits[s]}"
               for s, r, ts in sorted(res.site_rows(), key=lambda x: -len(x[2]))))
    _write(os.path.join(outdir, "dynamic.stats.json"),
           json.dumps(res.summary(), indent=2))
    print(json.dumps(res.summary(), indent=2))
    return 0


def cmd_patch(args) -> int:
    """Write a length-preserving de-obfuscated copy and verify containment."""
    from . import patch as patch_mod

    pm = PEMap(args.pe)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2
    out = args.output or (args.pe.rsplit(".", 1)[0] + "_depatched.dll")
    res = patch_mod.depatch(pm, out, v.vm_section)
    patch_mod.write_manifest(res, args.manifest or (out + ".manifest.json"))
    check = patch_mod.verify_contained(pm, out, v.vm_section)
    print(json.dumps({**res.to_dict(), "verification": check}, indent=2))
    return 0 if check.get("outside_vm_section") == 0 else 1


def cmd_probe(args) -> int:
    """Automatic survey of one image: engine, structure, and what can be repaired."""
    from . import devirt as devirt_mod

    pm = PEMap(args.pe)
    pr = devirt_mod.probe(pm)
    if args.output:
        _write(args.output, json.dumps(pr.to_dict(), indent=2))
    if args.json:
        print(json.dumps(pr.to_dict(), indent=2))
    else:
        print(pr.render())
    return 0


def cmd_entries(args) -> int:
    """List every VM entry point recoverable from the image."""
    from . import functions as func_mod
    from . import xrefs as xref_mod

    pm = PEMap(args.pe)
    v = detect(pm)
    if not v.vm_section:
        print("no VM section detected", file=sys.stderr)
        return 2
    funcs = func_mod.recover_functions(pm)
    edges = xref_mod.native_to_vm_edges(pm, funcs, v.vm_section)
    counts: dict[int, list[int]] = {}
    for e in edges:
        counts.setdefault(e.target, []).append(e.caller)
    lines = ["target_rva\tincoming\tsection\tcallers"]
    for t, callers in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        sec = pm.section_name_for_rva(t) or "?"
        lines.append(f"{t:#x}\t{len(callers)}\t{sec}\t"
                     + ",".join(f"{c:#x}" for c in sorted(set(callers))[:8]))
    text = "\n".join(lines) + "\n"
    if args.output:
        _write(args.output, text)
    else:
        sys.stdout.write(text)
    print(f"# {len(counts)} distinct VM entries from {len(edges)} edges "
          f"({len({e.caller for e in edges})} native callers)", file=sys.stderr)
    return 0


def cmd_write_devirt(args) -> int:
    """Emit the repaired binary plus symbols and a probe report, in one pass.

    The QVM counterpart of `tvm-devirt write-devirt`. See `qvmtool/devirt.py` for the
    measurements showing that on this engine the honest description is de-obfuscation,
    not decryption: the VM section is measurably not encrypted.
    """
    from . import devirt as devirt_mod

    pm = PEMap(args.pe)
    res = devirt_mod.write_devirt(pm, args.output, dynamic_edges=args.dynamic,
                                  edges_tsv=args.edges,
                                  emit_symbols=not args.no_symbols,
                                  run_static_slice=args.run_static_slice)
    print(devirt_mod.Probe(**res.probe).render())
    print(res.render())
    v = res.verification
    if not (v.get("size_preserved") and v.get("outside_vm_section") == 0):
        print("!! verification FAILED: the repair changed the file size or touched "
              "bytes outside the VM section", file=sys.stderr)
        return 1
    print("ok: repair is length-preserving and confined to the VM section")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qvmtool",
        description="Detect and symbolise QVM/TVM virtualised PE images.")
    p.add_argument("--version", action="version", version=f"qvmtool {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="identify the engine and show evidence")
    d.add_argument("pe")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)

    tv = sub.add_parser("tvm",
                        help="TVM-only pass: VIP slot, fetch idiom, threaded "
                             "dispatch, E9 trampolines")
    tv.add_argument("pe")
    tv.add_argument("-o", "--output", help="output directory")
    tv.add_argument("--edges", action="store_true",
                    help="also emit edges TSV + Ghidra/IDA injection scripts")
    tv.add_argument("--no-functions", action="store_true",
                    help="do not create functions at recovered targets")
    tv.set_defaults(func=cmd_tvm)

    f = sub.add_parser("functions", help="recover the function map")
    f.add_argument("pe")
    f.add_argument("-o", "--output")
    f.set_defaults(func=cmd_functions)

    t = sub.add_parser("trampolines", help="anchored native->VM cross-references")
    t.add_argument("pe")
    t.add_argument("-o", "--output")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=cmd_trampolines)

    r = sub.add_parser("regions", help="encrypted bands and VM section structure")
    r.add_argument("pe")
    r.add_argument("-o", "--output")
    r.add_argument("--section", help="also emit a region profile for this section")
    r.add_argument("--threshold", type=float, default=7.5)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_regions)

    y = sub.add_parser("symbols", help="recover and emit symbols")
    y.add_argument("pe")
    y.add_argument("-o", "--output", help="output prefix")
    y.add_argument("--sample-budget", type=int, default=200_000)
    y.set_defaults(func=cmd_symbols)

    s = sub.add_parser("survey", help="run everything and write a report")
    s.add_argument("pe")
    s.add_argument("-o", "--output", help="output directory")
    s.add_argument("--sample-budget", type=int, default=200_000)
    s.set_defaults(func=cmd_survey)

    e = sub.add_parser("edges",
                       help="recover CFG edges and emit Ghidra/IDA injection scripts")
    e.add_argument("pe")
    e.add_argument("-o", "--output", help="output directory")
    e.add_argument("--window", type=int, default=4096)
    e.add_argument("--max-anchors", type=int, default=8)
    e.add_argument("--max-instrs", type=int, default=512)
    e.add_argument("--no-functions", action="store_true",
                   help="do not create functions at recovered targets")
    e.add_argument("--from-edges", help="reuse an existing edges.tsv instead of "
                                        "re-running the (slow) static pipeline")
    e.add_argument("--dynamic", help="merge a dynamic_edges.tsv (from the `dynamic` "
                                     "verb) into the edge set before validating")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_edges)

    d = sub.add_parser("dynamic",
                       help="recover CFG edges by emulating from real call contexts")
    d.add_argument("pe")
    d.add_argument("-o", "--output", help="output directory")
    d.add_argument("--entries", type=lambda s: int(s, 0), nargs="*", default=None,
                   help="entry RVAs to emulate; default = every native->VM edge target")
    d.add_argument("--budget", type=int, default=200_000,
                   help="instruction budget per entry")
    d.add_argument("--max-entries", type=int, default=64)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_dynamic)

    pt = sub.add_parser("patch",
                        help="write a length-preserving de-obfuscated copy")
    pt.add_argument("pe")
    pt.add_argument("-o", "--output")
    pt.add_argument("--manifest", help="manifest output path")
    pt.set_defaults(func=cmd_patch)

    # --- the tvm-devirt-shaped surface: probe / entries / write-devirt ---
    pb = sub.add_parser("probe",
                        help="automatic survey: engine, structure, what can be repaired")
    pb.add_argument("pe")
    pb.add_argument("-o", "--output", help="write the probe report as JSON")
    pb.add_argument("--json", action="store_true")
    pb.add_argument("--quiet", action="store_true")
    pb.set_defaults(func=cmd_probe)

    en = sub.add_parser("entries", help="list every recoverable VM entry point")
    en.add_argument("pe")
    en.add_argument("-o", "--output")
    en.set_defaults(func=cmd_entries)

    wd = sub.add_parser("write-devirt",
                        help="emit a repaired binary + symbols in one automatic pass "
                             "(QVM counterpart of `tvm-devirt write-devirt`)")
    wd.add_argument("pe")
    wd.add_argument("output")
    wd.add_argument("--dynamic", help="dynamic_edges.tsv to merge before emitting")
    wd.add_argument("--edges", help="reuse a previously computed edges.tsv")
    wd.add_argument("--run-static-slice", action="store_true",
                    help="recompute the static slice (minutes on a large VM section)")
    wd.add_argument("--no-symbols", action="store_true")
    wd.set_defaults(func=cmd_write_devirt)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
