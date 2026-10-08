"""Tests for the newer layers: CFG assembly, Ghidra emission, and patching.

These are the parts whose bugs are expensive (a wrong edge set misleads an analyst;
a wrong patch corrupts a binary), so they get explicit coverage even though they
are awkward to unit-test.
"""

from __future__ import annotations

import pytest

from qvmtool import cfg, ghidra, patch
from qvmtool.model import Edge


# ------------------------------------------------------------------- cfg

def test_merge_dedupes_on_site_and_target_keeping_first_stage():
    a = [Edge(0, 0x100, "call", 0x200, False)]
    b = [Edge(0, 0x100, "call", 0x200, False),      # duplicate
         Edge(0, 0x100, "call", 0x300, False)]      # new target, same site
    out = cfg.merge(a, b)
    assert len(out) == 2
    assert [(e.site, e.target, e.kind) for e in out] == [
        (0x100, 0x200, "call"), (0x100, 0x300, "call")]


def test_tsv_roundtrip_preserves_site_target_kind():
    edges = [Edge(0, 0x1000, "idiom", 0x2000, False),
             Edge(0, 0x1010, "slice", 0x2100, True)]
    back = cfg.from_tsv(cfg.to_tsv(edges))
    assert [(e.site, e.target, e.kind) for e in back] == [
        (0x1000, 0x2000, "idiom"), (0x1010, 0x2100, "slice")]


def test_validate_edges_needs_a_real_program_and_rejects_fallthrough():
    """Without a PEMap there is nothing to validate against, so the module must
    still behave: a fall-through edge is dropped purely on its delta."""
    # delta of 2 means "target is the instruction after a 2-byte jmp": a null branch.
    e_null = Edge(0, 0x1000, "idiom", 0x1002, False)
    e_real = Edge(0, 0x2000, "idiom", 0x1F00, False)
    assert (e_null.target - e_null.site) == 2
    assert (e_real.target - e_real.site) not in (2, 3)


# ---------------------------------------------------------------- ghidra

def test_cfg_edge_java_is_valid_java_shaped_and_uses_absolute_addresses():
    rows = [(0x1000, 0x2000, "idiom"), (0x1010, 0x2100, "slice")]
    src = ghidra.emit_cfg_edges_java(rows, 0x180000000)
    assert "class InjectQvmEdges" in src
    assert "0x180001000L" in src and "0x180002000L" in src
    # the API trap this had to work around
    assert "addMemoryReference" in src
    assert "(RefType) RefType.COMPUTED_JUMP" in src


def test_cfg_edge_java_balance_of_braces():
    src = ghidra.emit_cfg_edges_java([(0x10, 0x20, "call")], 0x1000)
    assert src.count("{") == src.count("}")


def test_cfg_edge_idc_targets_absolute_addresses():
    src = ghidra.emit_cfg_edges_idc([(0x10, 0x20, "call")], 0x180000000)
    assert "0x180000010" in src and "0x180000020" in src
    assert "add_cref" in src


def test_edges_to_rows_keeps_kinds():
    rows = ghidra.edges_to_rows([Edge(0, 1, "idiom", 2, False)])
    assert rows == [(1, 2, "idiom")]


# ----------------------------------------------------------------- patch

def test_patch_span_excludes_the_live_push_and_pop():
    """Regression: the shift+jump tail is the only safe thing to rewrite."""
    from qvmtool import deobf
    buf = (b"\xe8\x00\x00\x00\x00"
           b"\x33\x9c\x0f\xb8\xd4\xe6\xb5"
           b"\x41\x58"
           b"\x49\x81\xc0\x13\x00\x00\x00"
           b"\x41\xff\xe0")
    va = 0x1802A248F
    b = deobf.resolve_hidden_branches(buf, va, va, va + len(buf)).branches[0]
    start, end = b.patch_span
    # the call (5B) and the live junk (7B) and the pop (2B) must all survive
    assert start == va + 5 + 7 + 2
    assert start > b.anchor
    assert end == va + len(buf)


def test_gitignore_excludes_sample_derived_artifacts():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
    for pattern in ("out/", "*_depatched.dll", "*.symbols.tsv", "decomp/"):
        assert pattern in text, f"{pattern} must not be committable"


# ------------------------------------------------------------- regression

def test_capstone_x86_operand_type_constants_are_not_the_generic_ones():
    """Regression guard for a bug that cost a whole analysis round.

    In capstone's *x86* binding `X86_OP_REG == 1` and `X86_OP_IMM == 2`, which is the
    opposite of the intuition that REG comes second. A bare `op.type == 1` therefore
    tests REG, not IMM. `emulate.py` had exactly that mistake in its indirect-branch
    filter, which silently discarded every register-indirect jump: a sweep ran 3.14M
    instructions and reported zero dynamic edges, and the null result was briefly
    mistaken for an architectural finding.

    Asserting the constants' actual values makes the trap explicit to the next reader.
    """
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
    assert X86_OP_REG == 1
    assert X86_OP_IMM == 2
    assert X86_OP_MEM == 3


def test_emulate_filter_source_uses_the_named_constant():
    """The filter must compare against X86_OP_IMM by name, never a bare 1.

    Parsed with `ast` rather than grepped, so the explanatory comment in
    `emulate.py` that quotes the buggy line does not trip the guard.
    """
    import ast
    import inspect
    from qvmtool import emulate

    tree = ast.parse(inspect.getsource(emulate))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) \
                and node.left.attr == "type":
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(comp, ast.Constant) and comp.value == 1 \
                        and isinstance(op, (ast.Eq, ast.NotEq)):
                    offenders.append(node.lineno)
    assert not offenders, (
        f"bare numeric operand-type comparison at line(s) {offenders}: "
        "X86_OP_REG == 1 in capstone's x86 binding, so `== 1` tests REG, not IMM")
    assert "X86_OP_IMM" in inspect.getsource(emulate)


def test_indirect_target_mapping_covers_all_gprs():
    """The name->uc register map must cover all 16 GPRs or targets are silently lost."""
    from qvmtool import emulate
    if not emulate.HAVE_UNICORN:
        pytest.skip("unicorn not installed")
    m = emulate._gpr_map()
    for r in ("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
              "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"):
        assert r in m and m[r] is not None


# ------------------------------------------------ indirect-branch enumeration

def test_indirect_sites_covers_memory_forms_that_jmp_sites_misses():
    """`call r/m64` is FF /2 and `jmp r/m64` is FF /4; only mod=11 is a register.

    Regression guard for a real enumeration gap: `_jmp_sites` matched only the
    register forms (`FF E0..E7`), finding 8,964 sites on the reference sample where
    the true population is 28,382 -- the other 19,418 are memory-indirect shapes
    (`jmp [rip+d]`, `call [reg+8]`), which is how a dispatch table is reached.
    """
    from qvmtool import deobf

    buf = (b"\xff\xe0"                          # jmp rax        @0  (2B)
           b"\xff\x25\x00\x00\x00\x00"          # jmp [rip+0]    @2  (6B)
           b"\xff\x50\x08"                      # call [rax+8]   @8  (3B)
           b"\x41\xff\xe1")                     # jmp r9         @11 (3B)
    got = deobf.indirect_sites(buf)
    # Exactly one entry per instruction. The REX form must NOT also be counted at
    # its FF tail -- that double-count was a real bug in this function's first cut,
    # caught by this very test.
    assert got["jmp_reg"] == [0, 11], got["jmp_reg"]
    assert got["jmp_mem"] == [2], got["jmp_mem"]
    assert got["call_mem"] == [8], got["call_mem"]

    narrow = {off for off, _r, _k in deobf._jmp_sites(buf)}
    assert narrow == {0, 11}
    wide = set(got["jmp_reg"]) | set(got["jmp_mem"]) | \
        set(got["call_reg"]) | set(got["call_mem"])
    assert wide > narrow, "extended census must find strictly more sites"


def test_indirect_sites_handles_rex_prefixed_memory_forms():
    """A REX byte must be absorbed before the FF opcode, not treated as data."""
    from qvmtool import deobf
    buf = b"\x41\xff\x24\xc5\x00\x00\x00\x00"   # jmp [r8*8+0] with a SIB byte
    got = deobf.indirect_sites(buf)
    assert got["jmp_mem"] == [0]
    assert got["jmp_reg"] == []


# ------------------------------------------------------- edge unit handling

def test_emitted_edge_addresses_are_single_based():
    """Regression: the Ghidra emitter must add the image base exactly once.

    `Edge.site`/`Edge.target` are RVAs project-wide, and `emit_cfg_edges_java`
    documents its input as RVAs while adding `image_base`. Feeding it VA-keyed edges
    (which the deobf/slice path and the emulator naturally produce) yields a
    double-based address: 0x3002a221d instead of 0x1802a221d. That number is large
    enough to look plausible, so the botched script fails silently at run time
    instead of erroring.
    """
    from qvmtool import ghidra
    from qvmtool.model import Edge
    base = 0x180000000
    rva_edges = [Edge(0, 0x2A221D, "idiom", 0x05D73D3, False)]
    src = ghidra.emit_cfg_edges_java(
        [(e.site, e.target, e.kind) for e in rva_edges], base)
    assert "0x1802a221dL" in src
    assert "0x3002a221dL" not in src


def test_rebase_detection_ignores_wild_targets():
    """A single out-of-image target must not suppress VA->RVA normalisation.

    Real failure: one wild target of 0x156e1b2b6 (below the 0x180000000 image base)
    made a min-over-everything test conclude the file was already RVA-keyed, so all
    3,454 dynamic edges were then discarded as out-of-section and validation silently
    collapsed to the 535 static edges only.
    """
    from qvmtool import cfg
    base = 0x180000000
    va_rows = [(base + 0x2A221D, 0x156e1b2b6), (base + 0x2A22E0, base + 0x2A2186)]
    assert cfg._needs_rebase(va_rows, base) is True, \
        "a wild target must not defeat rebasing"
    rva_rows = [(0x2A221D, 0x05D73D3), (0x2A22E0, 0x2A2186)]
    assert cfg._needs_rebase(rva_rows, base) is False
    assert cfg._needs_rebase(va_rows, None) is False


def test_load_dynamic_tsv_rebases_va_keyed_files(tmp_path):
    """End-to-end: a VA-keyed dynamic TSV must come back as RVAs."""
    from qvmtool import cfg
    base = 0x180000000
    p = tmp_path / "d.tsv"
    p.write_text("site\ttarget\thits\n"
                 f"{base + 0x2A221D:#x}\t{0x156e1b2b6:#x}\t3\n"
                 f"{base + 0x2A22E0:#x}\t{base + 0x2A2186:#x}\t1\n",
                 encoding="utf-8")
    edges = cfg.load_dynamic_tsv(str(p), image_base=base)
    assert edges[0].site == 0x2A221D
    assert edges[0].target == 0x156e1b2b6 - base
    assert edges[1].target == 0x2A2186


def test_from_tsv_rebases_legacy_va_keyed_files():
    from qvmtool import cfg
    base = 0x180000000
    text = ("site\ttarget\tkind\n"
            f"{base + 0x2A310D:#x}\t{base + 0x2A310F:#x}\tidiom\n")
    edges = cfg.from_tsv(text, image_base=base)
    assert (edges[0].site, edges[0].target) == (0x2A310D, 0x2A310F)
