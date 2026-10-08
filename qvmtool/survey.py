"""Pipeline orchestration: one call produces the whole survey.

Order matters and is deliberate:
  1. detect            -- cheap, decides whether the rest is meaningful
  2. recover functions -- anchored function map; everything downstream depends on it
  3. idioms            -- census + stride (settles "bytecode VM or not")
  4. trampolines       -- needs (2) for anchored decoding
  5. regions           -- structure + what needs unpacking
  6. state             -- VIP carrier, needs (3) sample counters
  7. symbols           -- needs (2), (4)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import detect as detect_mod
from . import deobf as deobf_mod
from . import engine as engine_mod
from . import functions as func_mod
from . import idiom as idiom_mod
from . import regions as regions_mod
from . import state as state_mod
from . import symbols as symbols_mod
from . import tvm as tvm_mod
from . import xrefs as xrefs_mod
from .model import Edge, Function, Region, StateCarrier, Verdict
from .pe import IMAGE_DIR_EXCEPTION, PEMap


def sample_anchors(buf: bytes, profile: engine_mod.EngineProfile) -> list[int]:
    """Frame-decode anchors appropriate to the engine.

    This is where the QVM assumption was fatal rather than merely wrong: `survey`
    anchored its MBA sample on `call $+5` sites, and a `.tvm0` module has 0-12 of
    them against 200,000 instructions of budget. The sample therefore ran on a few
    hundred instructions (or none) and the carrier verdict collapsed to `unknown`
    or `context_heap` for every TVM module in the corpus. TVM anchors on its own
    dispatch epilogues and VIP loads, which are equally genuine instruction
    boundaries.
    """
    out: list[int] = []
    if "call-self-pc" in profile.sample_anchors:
        out += idiom_mod.positions(buf, idiom_mod.CALL_5)
    if "threaded-epilogue" in profile.sample_anchors:
        out += [off for off, _reg in tvm_mod.epilogue_sites(buf)]
    if "vip-load" in profile.sample_anchors:
        for reg in range(16):
            rex = 0x4C if reg >= 8 else 0x48
            low = reg & 7
            pat = bytes([rex, 0x8B, 0x45 | (low << 3), tvm_mod.VIP_SLOT_KNOWN])
            start = 0
            while True:
                i = buf.find(pat, start)
                if i < 0:
                    break
                out.append(i)
                start = i + 1
    return sorted(set(out))


@dataclass
class Survey:
    pm: PEMap
    verdict: Verdict
    functions: list[Function] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    clusters: list[tuple[int, int]] = field(default_factory=list)
    stubs: list[tuple[int, Edge]] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    bands: list[Region] = field(default_factory=list)
    idioms: dict[str, Any] = field(default_factory=dict)
    stride: dict[str, Any] = field(default_factory=dict)
    mba: dict[str, Any] = field(default_factory=dict)
    prologue: dict[str, Any] = field(default_factory=dict)
    carrier: StateCarrier = field(default_factory=StateCarrier)
    pointer_tables: list[tuple[str, int, int]] = field(default_factory=list)
    consecutive_tables: list[tuple[str, int, int, int]] = field(default_factory=list)
    deobf: dict[str, Any] = field(default_factory=dict)
    utf16_overlap: int = 0
    symbols: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    #: engine profile used for this survey, and the TVM-only measurements
    profile: str = "unknown"
    anchors_used: int = 0
    anchor_kind: str = ""
    tvm: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------- convenience
    def functions_in(self, section: str, cap: Optional[int] = None) -> list[Function]:
        out = [f for f in self.functions if f.section == section]
        return out[:cap] if cap else out

    def by_section(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.functions:
            key = f.section or "?"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict[str, Any]:
        return {
            "file": self.pm.path,
            "sha256": self.pm.sha256,
            "image_base": self.pm.base,
            "entry_point": self.pm.ep,
            "engine": self.verdict.engine,
            "engine_core": self.verdict.engine_core,
            "engine_profile": self.profile,
            "confidence": round(self.verdict.confidence, 3),
            "vm_section": self.verdict.vm_section,
            "vm_range": [hex(x) for x in (self.verdict.vm_range or (0, 0))],
            "functions_total": len(self.functions),
            "functions_by_section": self.by_section(),
            "vm_edges": len(self.edges),
            "vm_edges_via_resync": sum(1 for e in self.edges if e.via_resync),
            "vm_distinct_targets": len(self.clusters),
            "vm_distinct_callers": len({e.caller for e in self.edges}),
            "vm_stubs": len(self.stubs),
            "encrypted_bytes": sum(b.size for b in self.bands),
            "state_carrier": self.carrier.carrier,
            "sample_anchors": self.anchors_used,
            "sample_anchor_kind": self.anchor_kind,
            "hidden_branches_resolved": self.deobf.get("resolved_branches", 0),
            "hidden_branch_resolution_rate": self.deobf.get("resolution_rate", 0.0),
            "pointer_table_hits": len(self.pointer_tables),
            "consecutive_table_runs": len(self.consecutive_tables),
            "handler_table_present": bool(self.consecutive_tables),
            "utf16_false_positive_span": self.utf16_overlap,
            "tvm_dispatch": self.tvm.get("dispatch", {}),
            "tvm_trampolines": self.tvm.get("trampolines", {}),
            "elapsed_s": round(self.elapsed, 2),
        }


def run(path: str, sample_budget: int = 200_000) -> Survey:
    t0 = time.time()
    pm = PEMap(path)
    verdict = detect_mod.detect(pm)
    s = Survey(pm=pm, verdict=verdict)

    s.functions = func_mod.recover_functions(pm)
    if verdict.vm_section is None:
        s.elapsed = time.time() - t0
        return s

    vm = pm.section_named(verdict.vm_section)
    assert vm is not None
    buf = pm.section_bytes(vm)
    profile = engine_mod.profile_for_section(verdict.vm_section)
    core_engine = {"qvm-mutation": "qvm", "tvm-threaded": "tvm"}.get(
        verdict.engine_core)
    if core_engine and core_engine != profile.name:
        # the measured core outranks the section name
        profile = engine_mod.profile_for_engine(core_engine)
    s.profile = profile.name

    s.idioms = idiom_mod.idiom_summary(buf)
    s.stride = idiom_mod.full_stride_profile(buf)

    # anchored decode sample: idiom sites are genuine instruction boundaries. WHICH
    # idiom is engine-specific -- anchoring on `call $+5` starves a TVM sample.
    anchors = sample_anchors(buf, profile)
    s.anchor_kind = "+".join(profile.sample_anchors)
    s.anchors_used = len(anchors)
    step = max(1, len(anchors) // sample_budget) if anchors else 1
    s.mba = idiom_mod.mba_metrics(buf, anchors[::step], max_instructions=sample_budget)

    s.edges = xrefs_mod.native_to_vm_edges(pm, s.functions, verdict.vm_section)
    if profile.native_entry_form == "e9-trampoline":
        # The anchored pass misses the trampolines the vendor placed in `.text` gaps
        # no RUNTIME_FUNCTION covers (~24% of them), so add the byte-anchored set.
        extra, tstats = tvm_mod.native_trampoline_edges(pm, verdict.vm_section)
        from . import cfg as cfg_mod
        s.edges = cfg_mod.merge(s.edges, extra)
        s.tvm["trampolines"] = tstats
    s.clusters = xrefs_mod.vm_entry_clusters(s.edges)
    s.stubs = xrefs_mod.vm_stubs(pm, s.functions, verdict.vm_section)

    markers = regions_mod.tvm_code_markers if profile.name == "tvm" else None
    s.regions = regions_mod.vm_regions(pm, verdict.vm_section, code_markers=markers)
    s.bands = regions_mod.encrypted_bands(pm)
    exc = pm.data_directory(IMAGE_DIR_EXCEPTION)
    if exc is not None and exc.VirtualAddress:
        s.regions = regions_mod.annotate_unwind_table(
            pm, s.regions, exc.VirtualAddress, exc.Size)

    entry = 0x2A2000 if verdict.engine == "qvm" else vm.va
    if s.clusters:
        entry = s.clusters[0][0]
    s.prologue = state_mod.prologue_shape(pm, min(entry, vm.va + vm.vsize - 1))

    if profile.name == "tvm":
        ranked = tvm_mod.discover_vip_slot(buf)
        slot = ranked[0]["slot"] if ranked else tvm_mod.VIP_SLOT_KNOWN
        s.tvm["vip_slots"] = ranked
        s.tvm["fetch"] = tvm_mod.fetch_idiom_census(buf, slot)
        s.tvm["context"] = tvm_mod.context_frame_census(buf)
        # the QVM `slice.py` pipeline resolves nothing on a threaded interpreter, so
        # the engine's own dispatch resolver is what produces real edges here.
        s.deobf = tvm_mod.resolve_section(pm, verdict.vm_section).summary()
        s.carrier = tvm_mod.assess_carrier(verdict.vm_section, ranked, s.tvm["context"],
                                           s.mba)
        s.tvm["dispatch"] = s.deobf
    else:
        s.carrier = state_mod.assess(s.mba, s.prologue)
        # resolve the engine's obfuscated-branch idioms inside the VM payload
        s.deobf = deobf_mod.resolve_section(pm, verdict.vm_section).summary()

    s.pointer_tables = xrefs_mod.find_absolute_pointers(pm, verdict.vm_section, width=8)
    s.consecutive_tables = xrefs_mod.consecutive_runs(s.pointer_tables, width=8, minimum=8)
    s.utf16_overlap = xrefs_mod.utf16_false_positive_span(vm.va, vm.va + vm.vsize)

    s.symbols = symbols_mod.build_symbols(pm, s.functions, s.edges, verdict, s.stubs)
    s.elapsed = time.time() - t0
    return s
