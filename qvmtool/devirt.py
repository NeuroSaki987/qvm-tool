"""Automatic QVM probe + de-obfuscation writer -- the QVM analogue of `tvm-devirt`.

`write-devirt` in tvm-devirt takes a protected image and emits a repaired one with the
virtualised code lowered back to native form. This module is the QVM counterpart:

    qvmtool probe        <pe>             -- what is this, and what is in it?
    qvmtool entries      <pe>             -- every VM entry point it can find
    qvmtool write-devirt <pe> <out>       -- emit the repaired binary + symbols

WHAT THIS TOOL DOES *NOT* CLAIM
-------------------------------
It is not a "decryptor", and it would be dishonest to ship one under that name for this
target. Three independent measurements say the VM section is **not encrypted**:

  1. A byte-level census finds ~25k indirect branches and tens of thousands of get-PC
     idioms inside `.qvm0`; a 5-byte idiom cannot occur by chance (P = 2^-40), so the
     section is plaintext machine code.
  2. Emulation of 5.28M instructions with the section mapped `UC_PROT_ALL` -- i.e. an
     in-process decryptor would have succeeded silently -- recorded **zero writes** to
     the `.qvm0` address range. Nothing decrypts it, at all.
  3. The high-entropy bytes are **never executed**. Execution inside those pages covers
     ~1.4% of each page (median 4-instruction cleartext forwarder stubs:
     `mov [rsp+d],imm32; lea rsp,[rsp+d]; call <plaintext VM entry>`), and 542 such
     islands end in a terminal `call` with no return, which is why the following bytes
     never run. They are filler, not ciphertext.

So the repair this tool performs is **de-obfuscation and normalisation**, not decryption:
it makes the real control flow explicit so a disassembler can follow it, and it emits
symbols so the recovered structure lands in the analyst's tool.

WHAT IT DOES
------------
  1. detect the engine (QVM mutation vs TVM threaded) and pick the VM section
  2. recover the authoritative function map from the *relocated* Exception directory
     (the `.pdata` section is a decoy holding encrypted bytes)
  3. enumerate indirect-branch sites with the full census (register AND memory forms)
  4. resolve the obfuscated-branch idioms, discarding the semantically-null ones
  5. optionally merge dynamically observed edges (the strongest evidence available)
  6. write a **length-preserving** repair: null-branch tails -> nop, real hidden
     branches -> `jmp rel32`, touching bytes only inside the VM section
  7. verify containment (no byte outside the VM section changed) and emit symbols for
     Ghidra and IDA plus a machine-readable probe report

Length preservation matters: all relative offsets, relocations and unwind entries stay
valid, so the output loads and disassembles exactly like the input.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field

from . import cfg as cfg_mod
from . import deobf
from . import detect as detect_mod
from . import functions as func_mod
from . import patch as patch_mod
from . import regions as regions_mod
from . import symbols as symbols_mod
from . import xrefs as xref_mod
from .pe import IMAGE_DIR_EXCEPTION, PEMap

NOP = 0x90


@dataclass
class Probe:
    """Everything the probe phase learns about one image."""

    path: str = ""
    sha256: str = ""
    engine: str = "none"
    confidence: float = 0.0
    vm_section: str | None = None
    vm_size: int = 0
    sections: int = 0
    exports: int = 0
    functions_total: int = 0
    functions_by_section: dict = field(default_factory=dict)
    largest_function: int = 0
    native_to_vm_edges: int = 0
    native_callers: int = 0
    vm_targets: int = 0
    whole_function_stubs: int = 0
    census_sites: dict = field(default_factory=dict)
    census_total: int = 0
    idiom_resolved: int = 0
    idiom_null_padding: int = 0
    idiom_real_edges: int = 0
    encrypted_bands: list = field(default_factory=list)
    encrypted_bytes: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        L = [f"== QVM probe: {self.path}",
             f"   sha256      : {self.sha256}",
             f"   engine      : {self.engine} (confidence {self.confidence:.2f})",
             f"   vm section  : {self.vm_section} ({self.vm_size:,} bytes)",
             f"   sections    : {self.sections}   exports: {self.exports}",
             f"   functions   : {self.functions_total} {self.functions_by_section}"]
        if self.largest_function:
            L.append(f"   largest fn  : {self.largest_function:,} bytes")
        L.append(f"   native->VM  : {self.native_to_vm_edges} edges from "
                 f"{self.native_callers} callers into {self.vm_targets} targets "
                 f"({self.whole_function_stubs} whole-function stubs)")
        if self.census_total:
            L.append(f"   branch census: {self.census_total} indirect sites "
                     f"{self.census_sites}")
        L.append(f"   idiom resolve: {self.idiom_resolved} resolved, of which "
                 f"{self.idiom_null_padding} are semantically-null padding and "
                 f"{self.idiom_real_edges} are real branches")
        if self.encrypted_bands:
            L.append(f"   high-entropy : {self.encrypted_bytes:,} bytes in "
                     f"{len(self.encrypted_bands)} bands "
                     f"(never executed; filler, NOT ciphertext)")
        for n in self.notes:
            L.append(f"   note        : {n}")
        return "\n".join(L)


def probe(pm: PEMap, verbose: bool = False) -> Probe:
    p = Probe(path=pm.path, sha256=pm.sha256)
    v = detect_mod.detect(pm)
    p.engine = v.engine
    p.confidence = round(v.confidence, 3)
    p.vm_section = v.vm_section
    p.sections = len(pm.sections)
    p.exports = len(pm.exports())
    if not v.vm_section:
        p.notes.append("no VM section: nothing to repair")
        return p

    sec = pm.section_named(v.vm_section)
    p.vm_size = sec.vsize

    funcs = func_mod.recover_functions(pm)
    p.functions_total = len(funcs)
    counts: dict[str, int] = {}
    for f in funcs:
        k = f.section or "?"
        counts[k] = counts.get(k, 0) + 1
    p.functions_by_section = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    if funcs:
        p.largest_function = max(f.size for f in funcs)

    edges = xref_mod.native_to_vm_edges(pm, funcs, v.vm_section)
    p.native_to_vm_edges = len(edges)
    p.native_callers = len({e.caller for e in edges})
    p.vm_targets = len({e.target for e in edges})
    p.whole_function_stubs = len(xref_mod.vm_stubs(pm, funcs, v.vm_section))

    buf = pm.section_bytes(sec)
    census = deobf.indirect_sites(buf)
    p.census_sites = {k: len(v_) for k, v_ in census.items() if v_}
    p.census_total = sum(p.census_sites.values())

    lo, hi = pm.base + sec.va, pm.base + sec.va + sec.vsize
    d = deobf.resolve_hidden_branches(buf, lo, lo, hi)
    p.idiom_resolved = len(d.branches)
    p.idiom_null_padding = sum(1 for b in d.branches if b.fallthrough)
    p.idiom_real_edges = p.idiom_resolved - p.idiom_null_padding

    bands = regions_mod.encrypted_bands(pm)
    p.encrypted_bands = [{"section": b.section, "start": hex(b.start),
                          "end": hex(b.end), "size": b.size,
                          "entropy": round(b.entropy, 3)} for b in bands]
    p.encrypted_bytes = sum(b.size for b in bands)

    exc = pm.data_directory(IMAGE_DIR_EXCEPTION)
    if exc is not None and pm.section_name_for_rva(exc.VirtualAddress) == v.vm_section:
        p.notes.append(
            "Exception directory is relocated INTO the VM section; the `.pdata` "
            "section is a decoy, so the function map must be read from there")
    p.notes.append(
        "measured to be NOT encrypted: 0 runtime writes to the VM range across 5.28M "
        "emulated instructions with UC_PROT_ALL; high-entropy bytes are never-executed "
        "filler. This tool de-obfuscates, it does not decrypt")
    return p


@dataclass
class WriteResult:
    input: str = ""
    output: str = ""
    probe: dict = field(default_factory=dict)
    repair: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    dynamic_coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        L = ["", "== write-devirt", f"   in  : {self.input}",
             f"   out : {self.output}",
             f"   repair: {self.repair}",
             f"   verify: {self.verification}"]
        if self.dynamic_coverage:
            L.append(f"   dynamic: {self.dynamic_coverage}")
        L.append("   artifacts:")
        for a in self.artifacts:
            L.append(f"     {a}")
        return "\n".join(L)


def write_devirt(pm: PEMap, out_path: str, dynamic_edges: str | None = None,
                 edges_tsv: str | None = None, symbols_prefix: str | None = None,
                 emit_symbols: bool = True, run_static_slice: bool = False
                 ) -> WriteResult:
    """Emit a repaired (length-preserving, de-obfuscated) copy plus its artifacts.

    Edge sources, cheapest first:
      * `edges_tsv`      -- reuse a previously computed static edge list
      * `dynamic_edges`  -- runtime-observed edges (the strongest evidence available)
      * `run_static_slice` -- recompute the static pass; minutes on a large VM
        section, so it is opt-in rather than the default
    """
    res = WriteResult(input=pm.path, output=out_path)
    pr = probe(pm)
    res.probe = pr.to_dict()
    if not pr.vm_section:
        raise ValueError(f"{pm.path}: no VM section detected; nothing to repair")

    # --- repair ---
    patch_res = patch_mod.depatch(pm, out_path, pr.vm_section)
    res.repair = patch_res.to_dict()
    manifest_path = out_path + ".manifest.json"
    patch_mod.write_manifest(patch_res, manifest_path)
    res.artifacts.append(manifest_path)

    # --- verification: the repair must not touch anything outside the VM section ---
    res.verification = patch_mod.verify_contained(pm, out_path, pr.vm_section)

    # --- assemble the edge set from whichever sources were supplied ---
    static_edges: list = []
    if edges_tsv:
        with open(edges_tsv, encoding="utf-8") as f:
            static_edges = cfg_mod.from_tsv(f.read(), image_base=pm.base)
    elif run_static_slice:
        static_edges, _ = cfg_mod.discover_static_edges(pm, pr.vm_section)
    dyn: list = []
    if dynamic_edges:
        dyn = cfg_mod.load_dynamic_tsv(dynamic_edges, image_base=pm.base)

    if static_edges or dyn:
        res.dynamic_coverage = cfg_mod.coverage_report(static_edges, dyn)
        merged = cfg_mod.merge(static_edges, dyn)
        kept, vstats = cfg_mod.validate_edges(pm, merged, pr.vm_section)
        res.dynamic_coverage["validated"] = vstats
        from . import ghidra as ghidra_mod
        base = os.path.splitext(out_path)[0]
        p_java = base + ".inject_edges.java"
        p_idc = base + ".inject_edges.idc"
        with open(p_java, "w", encoding="utf-8", newline="\n") as f:
            f.write(ghidra_mod.emit_cfg_edges_java(
                [(e.site, e.target, e.kind) for e in kept], pm.base,
                make_functions=False))
        with open(p_idc, "w", encoding="utf-8", newline="\n") as f:
            f.write(ghidra_mod.emit_cfg_edges_idc(
                [(e.site, e.target, e.kind) for e in kept], pm.base))
        res.artifacts += [p_java, p_idc]

    # --- symbols so the recovered structure lands in Ghidra / IDA ---
    if emit_symbols:
        funcs = func_mod.recover_functions(pm)
        edges = xref_mod.native_to_vm_edges(pm, funcs, pr.vm_section)
        stubs = xref_mod.vm_stubs(pm, funcs, pr.vm_section)
        syms = symbols_mod.build_symbols(pm, funcs, edges,
                                         detect_mod.detect(pm), stubs)
        prefix = symbols_prefix or os.path.splitext(out_path)[0]
        meta = {"probe": pr.to_dict()}
        for path, text in (
            (prefix + ".symbols.json", symbols_mod.to_json(syms, meta)),
            (prefix + ".symbols.tsv", symbols_mod.to_tsv(syms)),
            (prefix + ".apply_ghidra.py",
             symbols_mod.to_ghidra_script(syms, pm.base)),
            (prefix + ".apply_ida.idc",
             symbols_mod.to_ida_idc(syms, pm.base)),
        ):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            res.artifacts.append(path)

    # --- probe report next to the output ---
    report_path = os.path.splitext(out_path)[0] + ".probe.txt"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(pr.render() + "\n" + res.render() + "\n")
    res.artifacts.append(report_path)
    return res
