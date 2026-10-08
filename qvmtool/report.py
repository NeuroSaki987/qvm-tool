"""Markdown report rendering from a Survey."""

from __future__ import annotations

from .survey import Survey


def _hex(x: int) -> str:
    return f"0x{x:X}"


def render(s: Survey) -> str:
    pm = s.pm
    v = s.verdict
    L: list[str] = []
    add = L.append

    add(f"# QVM/TVM survey — {pm.path}")
    add("")
    add(f"- SHA-256: `{pm.sha256}`")
    add(f"- ImageBase `{_hex(pm.base)}`, EntryPoint `{_hex(pm.ep)}`, "
        f"DllCharacteristics `0x{pm.dll_chars:X}`")
    add(f"- Detected engine: **{v.engine}** (confidence {v.confidence:.2f})"
        + (f" via `{v.vm_section}`" if v.vm_section else ""))
    add(f"- Engine core located: **{v.engine_core}** "
        f"(profile `{s.profile}`; this, not the section name, is what discriminates "
        f"QVM from TVM)")
    add("")

    add("## Sections")
    add("")
    add("| section | VA | VSize | Raw | RSize | Chars | exec | write |")
    add("|---|---|---|---|---|---|---|---|")
    for sec in pm.sections:
        add(f"| `{sec.name}` | {_hex(sec.va)} | {_hex(sec.vsize)} | {_hex(sec.raw)} | "
            f"{_hex(sec.rsize)} | `0x{sec.chars:08X}` | {sec.is_exec} | {sec.is_write} |")
    add("")

    add("## Detection evidence")
    add("")
    for e in v.evidence:
        add(f"- **{e.claim}** — {e.detail}" + (f"  _(+{e.weight:.2f})_" if e.weight else ""))
    add("")

    add("## Function map (recovered from the Exception directory)")
    add("")
    add(f"- Functions recovered: **{len(s.functions)}**")
    for name, count in s.by_section().items():
        add(f"  - `{name}`: {count}")
    if s.functions:
        biggest = max(s.functions, key=lambda f: f.size)
        add(f"- Largest single entry: `{_hex(biggest.begin)}–{_hex(biggest.end)}` "
            f"({biggest.size:,} bytes) in `{biggest.section}`")
    add("")

    add("## VM section structure")
    add("")
    add("| start | end | size | class | entropy |")
    add("|---|---|---|---|---|")
    for r in s.regions:
        add(f"| {_hex(r.start)} | {_hex(r.end)} | {r.size:,} | {r.kind} | {r.entropy:.2f} |")
    add("")

    add("## Idiom census (byte level, desync-immune)")
    add("")
    c = s.idioms.get("counts", {})
    add("| idiom | count |")
    add("|---|---|")
    for k, val in c.items():
        add(f"| {k} | {val:,} |")
    add(f"| **self-PC total** | **{s.idioms.get('self_pc_total', 0):,}** "
        f"({s.idioms.get('self_pc_per_mb', 0):,.0f}/MB) |")
    add("")

    add("### Stride test — fixed-cell bytecode stream?")
    add("")
    add("| idiom | sites | gap entropy (bits) | distinct gaps | largest gap share |")
    add("|---|---|---|---|---|")
    for key, prof in s.stride.items():
        add(f"| {key} | {prof['sites']:,} | {prof['gap_entropy_bits']} | "
            f"{prof['distinct_gaps']:,} | {prof['max_gap_share']:.3f} |")
    add("")
    flat = all(p["max_gap_share"] < 0.25 for p in s.stride.values()) if s.stride else False
    add(f"Verdict: **{'FLAT — no fixed-cell encoding (native mutated code, not bytecode)' if flat else 'STRIDED — possible cell encoding'}**")
    add("")

    add("## Native → VM cross-references (anchored, not linearly swept)")
    add("")
    add(f"- Edges: **{len(s.edges)}**")
    add(f"- Distinct native callers: **{len({e.caller for e in s.edges})}**")
    add(f"- Distinct VM entries: **{len(s.clusters)}**")
    add(f"- Whole-function trampolines (VM stub is the entire function): **{len(s.stubs)}**")
    add("")
    if s.clusters:
        add("### Shared VM entries (the real \"dispatcher\" shape)")
        add("")
        add("| VM entry | incoming edges |")
        add("|---|---|")
        for t, n in s.clusters[:15]:
            add(f"| {_hex(t)} | {n} |")
        add("")

    add("## Handler / dispatch table test")
    add("")
    add(f"- Data-resident 8-byte pointers into the VM section (aligned): "
        f"**{len(s.pointer_tables)}**")
    add(f"- Consecutive pointer runs (>=8 adjacent slots) — the actual table shape: "
        f"**{len(s.consecutive_tables)}**")
    add(f"- UTF-16 false-positive overlap with the VM RVA range: "
        f"**{s.utf16_overlap:,}** values wide")
    add("")
    if s.consecutive_tables:
        add("| section | start | entries | span |")
        add("|---|---|---|---|")
        for sec, start, entries, span in s.consecutive_tables[:15]:
            add(f"| `{sec}` | {_hex(start)} | {entries} | {span} |")
        add("")
        add("A table-shaped run exists — inspect before concluding.")
    elif s.profile == "tvm":
        add("- No table-shaped run exists, and TVM does not need one: dispatch is "
            "**threaded**, so each handler ends in an inline epilogue that materialises "
            "its successor as a per-site immediate. But do NOT read that as QVM's "
            "conclusion: this engine **does** fetch bytecode — a 16-bit word at the "
            "VIP — and the fetch count is reported below.")
    else:
        add("- A bytecode dispatcher must store handler addresses in a **run of "
            "consecutive slots**. No such run exists, the isolated hits are globals "
            "that happen to hold a VM address, and every one of the edges above is a "
            "direct `call`/`jmp` to a fixed VM address. Dispatch is therefore "
            "**statically resolved: there is no handler table and no opcode fetch**.")
    add("")

    if s.profile == "tvm" and s.tvm:
        add("## TVM threaded-interpreter measurements")
        add("")
        ranked = s.tvm.get("vip_slots") or []
        if ranked:
            add("| RBP slot | loads | stores | loaded then 16-bit dereferenced |")
            add("|---|---|---|---|")
            for r in ranked:
                add(f"| `[rbp+0x{r['slot']:X}]` | {r['loads']:,} | {r['stores']:,} | "
                    f"{r['load_then_16bit_deref']:,} |")
            add("")
            add(f"The VIP slot is **discovered**, not hard-coded: the winner is the slot "
                f"that is reloaded and then dereferenced as a bytecode word. On every "
                f"ACE module measured that is `[rbp+0x{ranked[0]['slot']:X}]`.")
            add("")
        fetch = s.tvm.get("fetch") or {}
        if fetch:
            add(f"- Bytecode fetch: VIP slot published **{fetch.get('vip_publishes', 0):,}** "
                f"times, **{fetch.get('mem16_dereferences', 0):,}** 16-bit "
                f"dereferences, advance-by-{fetch.get('advance_by', 2)} sites "
                f"**{fetch.get('advance_by_2', 0):,}** — the VM instruction stream unit "
                f"is a 16-bit word.")
        ctx = s.tvm.get("context") or {}
        if ctx:
            add(f"- Context frame: {ctx.get('distinct_slots_loaded', 0)} distinct "
                f"RBP slots loaded, {ctx.get('distinct_slots_stored', 0)} stored, "
                f"largest stride-8 write ladder "
                f"{ctx.get('largest_stride8_write_ladder', 0)} slots — a guest "
                f"register image, not a computed-index frame.")
        d = s.tvm.get("dispatch") or {}
        if d:
            add(f"- Threaded dispatch: {d.get('resolved_branches', 0):,} of "
                f"{d.get('threaded_epilogue_candidates', 0):,} `popfq; jmp reg` "
                f"epilogues resolved ({d.get('resolution_rate', 0):.1%}), "
                f"{d.get('null_branches', 0):,} of them CFG-neutral "
                f"({d.get('alignment_skip_only', 0):,} by skipping a single "
                f"anti-disassembly prefix byte), leaving "
                f"**{d.get('real_edges', 0):,} real handler edges**.")
        tr = s.tvm.get("trampolines") or {}
        if tr:
            add(f"- Native entry: {tr.get('at_function_entry', 0)} `E9 rel32` "
                f"trampolines at registered function entries, "
                f"**{tr.get('outside_function_map', 0)} in `.text` gaps that no "
                f"RUNTIME_FUNCTION covers**, "
                f"{tr.get('rejected_mid_instruction', 0)} rejected as decoy operand "
                f"bytes. An anchored-only pass silently loses the middle group.")
        add("")

    add("## State carrier (VIP location)")
    add("")
    add(f"- Verdict: **{s.carrier.carrier}**")
    add(f"- plain constant-offset frame refs: {s.carrier.stack_slot_refs:,}")
    add(f"- *indexed* frame refs (computed virtual-register numbers): "
        f"{s.carrier.obfuscated_index_refs:,}")
    add(f"- pointer-register-relative (context struct) refs: "
        f"{s.carrier.heap_context_refs:,}")
    add(f"- entry prologue pushes: {s.carrier.prologue_pushes}")
    add(f"- MBA sample anchors: {s.anchors_used:,} ({s.anchor_kind}) — engine-specific, "
        f"because anchoring on `call $+5` starves a TVM sample")
    add("")
    for n in s.carrier.notes:
        add(f"  - {n}")
    add("")

    add("## Encrypted regions (unpack targets)")
    add("")
    if s.profile == "tvm":
        add("Read this table together with the region profile below. On this engine the "
            "entropy threshold is **not** a code/encryption test: `.tvm0` is plaintext "
            "MBA handler code plus bytecode, and it trips `H > 7.5` in several modules "
            "because it is dense, not because it is ciphered. The section is also "
            "`EXECUTE|READ` and **not writable**, so there is nowhere to write an "
            "unpacked form of it — the blocks it labels `encrypted` are not unpack "
            "targets.")
        add("")
    add("| section | start | end | size | mean entropy |")
    add("|---|---|---|---|---|")
    for b in s.bands:
        add(f"| `{b.section}` | {_hex(b.start)} | {_hex(b.end)} | {b.size:,} | {b.entropy:.3f} |")
    add("")
    add(f"Total high-entropy bytes: **{sum(b.size for b in s.bands):,}**")
    add("")

    add("## Symbols")
    add("")
    add(f"- Total symbols: **{len(s.symbols)}**")
    kinds: dict[str, int] = {}
    for sym in s.symbols:
        kinds[sym["kind"]] = kinds.get(sym["kind"], 0) + 1
    for k, n in sorted(kinds.items()):
        add(f"  - {k}: {n}")
    add("")
    add(f"_Survey completed in {s.elapsed:.2f}s._")
    return "\n".join(L) + "\n"
