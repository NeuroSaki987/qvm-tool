"""Unit tests for the pure / statistical helpers.

These deliberately avoid depending on a sample binary: they test the parts whose
answers can be proven from constructed inputs, which is where a tool like this
actually goes wrong.
"""

from __future__ import annotations

import pytest

from qvmtool import deobf, detect, idiom, mba, slice, symbols, xrefs
from qvmtool.model import Edge, Function, Section, Verdict
from qvmtool.regions import classify_block, entropy


# --------------------------------------------------------------- detection

def test_idiom_census_counts_each_signature_form():
    buf = (b"\xe8\x00\x00\x00\x00"      # call $+5
           b"\xe9\x00\x00\x00\x00"      # jmp $+5
           b"\x0f\x84\x00\x00\x00\x00"  # jcc $+6
           b"\x9c"                      # pushfq
           b"\x9d"                      # popfq
           b"\x87")                     # xchg
    c = detect.idiom_census(buf)
    assert c["call_self_pc"] == 1
    assert c["jmp_self_pc"] == 1
    assert c["jcc_self_pc"] == 1
    assert c["pushfq"] == 1
    assert c["popfq"] == 1
    assert c["xchg"] == 1


def test_idiom_census_does_not_count_nonzero_displacement_branches():
    """`call rel32` with a real displacement must NOT look like the $+5 idiom."""
    buf = b"\xe8\x10\x00\x00\x00"
    assert detect.idiom_census(buf)["call_self_pc"] == 0


def test_entropy_bounds():
    assert entropy(b"") == 0.0
    assert entropy(b"\x00" * 4096) == 0.0
    # a byte-aligned 0..255 ramp has maximal entropy for one byte
    assert abs(entropy(bytes(range(256)) * 16) - 8.0) < 1e-9


# ------------------------------------------------------------------ stride

def test_fixed_stride_is_detected_when_a_gap_dominates():
    """A cell-encoded stream would put idioms on a regular grid."""
    cell = 16
    buf = bytearray(b"\x90" * (cell * 400))
    for i in range(400):
        buf[i * cell:i * cell + 5] = b"\xe8\x00\x00\x00\x00"
    prof = idiom.stride_profile(bytes(buf), idiom.CALL_5)
    assert idiom.is_fixed_stride(prof)
    assert prof["sites"] == 400


def test_flat_layout_is_not_flagged_as_fixed_stride():
    """Pseudo-random idiom placement must not be reported as a cell encoding."""
    import random
    rng = random.Random(1234)
    buf = bytearray(b"\x90" * 40000)
    for _ in range(300):
        off = rng.randrange(0, 40000 - 5)
        buf[off:off + 5] = b"\xe8\x00\x00\x00\x00"
    prof = idiom.stride_profile(bytes(buf), idiom.CALL_5)
    assert not idiom.is_fixed_stride(prof)


def test_stride_profile_reports_entropy_and_gaps():
    buf = b"\xe8\x00\x00\x00\x00" + b"\x90" * 3 + b"\xe8\x00\x00\x00\x00"
    prof = idiom.stride_profile(buf, idiom.CALL_5)
    assert prof["sites"] == 2
    assert prof["distinct_gaps"] == 1
    assert prof["top_gaps"][0][0] == 8


# ---------------------------------------------------------------- functions

def test_function_size_and_owning_section_lookup():
    f = Function(begin=0x1000, end=0x1100, unwind=0x5000, section=".text")
    assert f.size == 0x100
    assert f.section == ".text"


def test_utf16_false_positive_span_is_zero_when_ranges_do_not_overlap():
    # a VM section far above the UTF-16 ASCII dword space
    assert xrefs.utf16_false_positive_span(0x10000000, 0x11000000) == 0


def test_utf16_false_positive_span_matches_reference_sample_case():
    """The reference sample's .qvm0 range overlaps 0x00200020..0x007E007E heavily."""
    span = xrefs.utf16_false_positive_span(0x2A2000, 0x9E145C)
    assert span == 0x007E007E - 0x2A2000


def test_utf16_string_produces_dwords_inside_the_vm_range():
    """Documents the trap: a UTF-16 string looks like a table of VM-pointing RVAs."""
    text = "api-ms-win-core".encode("utf-16-le")
    dwords = [int.from_bytes(text[i:i + 4], "little") for i in range(0, len(text) - 4, 4)]
    vm_lo, vm_hi = 0x2A2000, 0x9E145C
    inside = [d for d in dwords if vm_lo <= d < vm_hi]
    assert inside, "expected the api-ms-win string to alias into the VM RVA range"


def test_consecutive_runs_requires_adjacent_slots():
    """A real table is adjacent slots; scattered globals must not form a run."""
    hits = [("sec", 0x1000, 0x2A2000), ("sec", 0x1008, 0x2A2100),
            ("sec", 0x1010, 0x2A2200), ("sec", 0x1018, 0x2A2300)]
    assert xrefs.consecutive_runs(hits, width=8, minimum=4) == [("sec", 0x1000, 4, 32)]
    # breaking adjacency in the middle splits (and here drops) the run
    gapped = [("sec", 0x1000, 1), ("sec", 0x1008, 2), ("sec", 0x2000, 3), ("sec", 0x2008, 4)]
    assert xrefs.consecutive_runs(gapped, width=8, minimum=3) == []


def test_vm_stub_detection_requires_branch_at_function_entry():
    """A branch later in the body must not qualify a function as a thin stub."""
    f = Function(begin=0x1000, end=0x1100, unwind=0x0, section=".text")
    e1 = Edge(caller=0x1000, site=0x1000, kind="call", target=0x2A2000)
    e2 = Edge(caller=0x1000, site=0x1040, kind="call", target=0x2A2000)
    # emulate the selection rule used by xrefs.vm_stubs
    first: dict[int, Edge] = {}
    for e in (e2, e1):
        if e.caller not in first or e.site < first[e.caller].site:
            first[e.caller] = e
    assert first[f.begin].site == f.begin


def test_vm_entry_clusters_rank_by_incoming_count():
    edges = [Edge(0x1000, 0x1000, "call", 0x2A2000),
             Edge(0x2000, 0x2000, "call", 0x2A2000),
             Edge(0x3000, 0x3000, "jmp", 0x2B0000)]
    clusters = xrefs.vm_entry_clusters(edges)
    assert clusters[0] == (0x2A2000, 2)
    assert clusters[1] == (0x2B0000, 1)


# ---------------------------------------------------------------- regions

def test_classify_block_separates_code_from_encrypted():
    # dense self-PC idioms are code even though such a block is low entropy --
    # density must outrank the padding test, otherwise a cell-encoded run of
    # `call $+5` is misfiled as padding.
    code = b"\xe8\x00\x00\x00\x00" * 40
    assert classify_block(code)[0] == "code"

    # realistic mutated-code block: high entropy AND several idioms
    import random
    rng = random.Random(7)
    real = bytearray(rng.randrange(256) for _ in range(4096))
    for i in range(12):
        real[i * 300:i * 300 + 5] = b"\xe8\x00\x00\x00\x00"
    cls, h = classify_block(bytes(real))
    assert cls == "code", f"expected code, got {cls} (H={h:.2f})"

    # high entropy, few zeros -> encrypted
    enc = bytes((i * 37 + 11) % 256 for i in range(4096))
    assert classify_block(enc)[0] == "encrypted"

    # padding has no idioms at all
    assert classify_block(b"\x00" * 4096)[0] == "zero"
    assert classify_block(b"\xcc" * 4096)[0] == "zero"


# ----------------------------------------------------------------- symbols

def test_ghidra_script_uses_absolute_addresses_and_is_valid_python():
    syms = [{"address": 0x1000, "name": "ace_CreateObject", "kind": "export",
             "detail": "ordinal 8"}]
    src = symbols.to_ghidra_script(syms, 0x180000000)
    assert "0x180001000" in src
    assert "ace_CreateObject" in src
    compile(src, "<ghidra>", "exec")   # the generated script must at least parse


def test_ida_script_targets_absolute_addresses():
    syms = [{"address": 0x2000, "name": "qvm_entry_2000", "kind": "vm_entry",
             "detail": "3 callers"}]
    src = symbols.to_ida_idc(syms, 0x180000000)
    assert "MakeName(0x180002000" in src


def test_tsv_emitter_keeps_kinds_and_addresses():
    syms = [{"address": 0x10, "name": "a", "kind": "function", "detail": "d"}]
    line = symbols.to_tsv(syms).splitlines()[1]
    assert line.startswith("0x00000010\ta\tfunction")


# ------------------------------------------------------------------ verdict

def test_verdict_confidence_grows_with_weighted_evidence():
    v = Verdict()
    v.add("a", "detail", 0.4)
    v.add("b", "detail", 0.3)
    assert abs(v.confidence - 0.7) < 1e-9
    assert len(v.evidence) == 2


def test_verdict_confidence_is_capped_at_one():
    v = Verdict()
    for _ in range(10):
        v.add("x", "y", 0.5)
    assert v.confidence == 1.0


def test_section_flag_helpers():
    s = Section(".qvm0", 0x2A2000, 0x100, 0x1000, 0x200, 0x68000020)
    assert s.is_exec and not s.is_write
    assert s.span == 0x200


# ------------------------------------------------------------------- deobf

def test_resolves_selfpc_branch_with_lea_shift():
    """`call $+5; pop rsi; lea rsi,[rsi+0xb]; jmp rsi` -> target = call+5+0xb."""
    # exactly the bytes observed in the reference sample at RVA 0x2C45BC
    buf = (b"\xe8\x00\x00\x00\x00"          # call $+5
           b"\x5e"                          # pop rsi
           b"\x48\x8d\xb4\x26\x0b\x00\x00\x00"   # lea rsi,[rsi+riz+0xb]
           b"\xff\xe6")                     # jmp rsi
    va = 0x180000000
    r = deobf.resolve_hidden_branches(buf, va, va, va + len(buf))
    assert r.resolved == 1
    b = r.branches[0]
    assert b.form == "selfpc"
    assert b.delta == 0x0B
    assert b.register == "rsi"
    assert b.target == va + 5 + 0x0B
    assert b.confidence == "high"


def test_selfpc_idiom_uses_add_not_lea_and_is_flagged_null():
    """Superseded an earlier expectation: a bare `pop reg; jmp reg` is NOT resolved.

    It was initially treated as "target = p+5", which is the address of the `pop`
    itself -- an apparent infinite loop. The real engine always applies a shift, and
    the shift is applied with `add reg, imm32`. See
    `test_selfpc_idiom_resolves_and_uses_add_not_lea`.
    """
    buf = b"\xe8\x00\x00\x00\x00" b"\x58" b"\xff\xe0"
    va = 0x180000000
    r = deobf.resolve_hidden_branches(buf, va, va, va + len(buf))
    assert r.resolved == 0


def test_resolves_rip_lea_and_imm64_backward_forms():
    va = 0x180000000
    # lea rdx,[rip+0x20] ; jmp rdx  -> target = end_of_lea + 0x20
    lea_off = 0x40
    buf = bytearray(b"\x90" * 0x100)
    buf[lea_off:lea_off + 7] = b"\x48\x8d\x15\x20\x00\x00\x00"
    jmp_off = lea_off + 7
    buf[jmp_off:jmp_off + 2] = b"\xff\xe2"
    r = deobf.resolve_hidden_branches(bytes(buf), va, va, va + len(buf))
    got = [b for b in r.branches if b.form == "riplea"]
    assert len(got) == 1
    assert got[0].target == va + lea_off + 7 + 0x20

    # mov rax, imm64 ; jmp rax
    buf2 = bytearray(b"\x90" * 0x60)
    buf2[0x10:0x12] = b"\x48\xb8"
    buf2[0x12:0x1A] = (va + 0x1234).to_bytes(8, "little")
    buf2[0x1A:0x1C] = b"\xff\xe0"
    r2 = deobf.resolve_hidden_branches(bytes(buf2), va, va, va + len(buf2))
    got2 = [b for b in r2.branches if b.form == "imm64"]
    assert len(got2) == 1
    assert got2[0].target == va + 0x1234


def test_branch_forms_can_be_disabled():
    buf = (b"\xe8\x00\x00\x00\x00" b"\x5e"
           b"\x48\x8d\xb4\x26\x0b\x00\x00\x00" b"\xff\xe6")
    va = 0x180000000
    r = deobf.resolve_hidden_branches(buf, va, va + len(buf), va + len(buf),
                                      forms=("riplea",))
    assert r.resolved == 0


# ------------------------------------------------------------------- MBA

def test_mba_folds_the_identities_this_injector_emits():
    x = mba.Var("x")
    # x - x == 0 and x ^ x == 0 are how the engine hides a constant zero
    assert mba.as_constant(mba.Binary("sub", x, x)) == 0
    assert mba.as_constant(mba.Binary("xor", x, x)) == 0
    # not(not(x)) == x ; neg(neg(x)) == x
    assert mba.simplify(mba.Unary("not", mba.Unary("not", x))) == x
    assert mba.simplify(mba.Unary("neg", mba.Unary("neg", x))) == x
    # (x | y) - (x & y) == x ^ y
    y = mba.Var("y")
    lhs = mba.Binary("sub", mba.Binary("or", x, y), mba.Binary("and", x, y))
    assert mba.simplify(lhs) == mba.Binary("xor", x, y)
    # (x | y) + (x & y) == x + y
    rhs = mba.Binary("add", mba.Binary("or", x, y), mba.Binary("and", x, y))
    assert mba.simplify(rhs) == mba.Binary("add", x, y)


def test_mba_constant_folding_across_arithmetic():
    c = lambda v: mba.Const(v, 64)
    assert mba.as_constant(mba.Binary("add", c(0x1000), c(0x20))) == 0x1020
    assert mba.as_constant(mba.Binary("xor", c(0xDEAD), c(0xBEEF))) == (0xDEAD ^ 0xBEEF)
    assert mba.as_constant(mba.Binary("rol", c(1), c(4))) == 0x10
    assert mba.as_constant(mba.Unary("trunc", c(0x1122334455667788), 64, 4)) == 0x55667788


def test_mba_affine_parts_recovers_the_base_through_mba_layers():
    """The shape that actually occurs: Const(base) + <non-constant MBA term>."""
    base = mba.Const(0x1802A2186, 64)
    dyn = mba.Unary("sext", mba.Binary(
        "xor", mba.Unknown("load-dyn"), mba.Const(0xC6274D22, 64)), 64, 4)
    got = mba.affine_parts(mba.Binary("add", base, dyn))
    assert got is not None
    assert got[0] == 0x1802A2186
    assert got[1] != 0


def test_mba_affine_parts_is_none_for_pure_constants():
    assert mba.affine_parts(mba.Const(0x1234, 64)) == (0x1234, 0)
    assert mba.affine_parts(mba.Binary("add", mba.Const(1, 64), mba.Const(2, 64))) \
        == (3, 0)


def test_contains_unknown_distinguishes_self_reference():
    """A slice that never established the base must be detectable."""
    self_ref = mba.Binary("add", mba.Unknown("reg:rsi"), mba.Const(8, 64))
    assert mba.contains_unknown(self_ref, "reg:rsi")
    assert not mba.contains_unknown(self_ref, "load-dyn")
    assert not mba.contains_unknown(mba.Const(5, 64))


# ----------------------------------------------------- fallthrough detection

def test_selfpc_idiom_resolves_and_uses_add_not_lea():
    """The real idiom applies its shift with `add reg, imm32`, not `lea`.

    Modelled on bytes observed at RVA 0x2A248F:
        call $+5 ; xor ebx,[..] ; pop r8 ; add r8, 0x13 ; jmp r8
    target = call+5+0x13, which is the instruction after the `jmp` -> a null branch.
    """
    buf = (b"\xe8\x00\x00\x00\x00"                      # call $+5
           b"\x33\x9c\x0f\xb8\xd4\xe6\xb5"              # xor ebx,[rdi+rcx-..]
           b"\x41\x58"                                  # pop r8
           b"\x49\x81\xc0\x13\x00\x00\x00"              # add r8, 0x13
           b"\x41\xff\xe0")                             # jmp r8
    va = 0x1802A248F
    r = deobf.resolve_hidden_branches(buf, va, va, va + len(buf))
    assert r.resolved == 1
    b = r.branches[0]
    assert b.delta == 0x13
    assert b.register == "r8"
    assert b.target == va + 5 + 0x13
    # and crucially: the target is the fall-through, i.e. the branch is a no-op
    assert b.fallthrough
    assert r.summary()["real_edges"] == 0
    assert r.summary()["fallthrough_only"] == 1


def test_pop_jmp_without_a_shift_is_not_claimed_as_resolved():
    """`pop reg ; jmp reg` walks the stack; its target is not a fixed address."""
    buf = b"\xe8\x00\x00\x00\x00" b"\x58" b"\xff\xe0"
    va = 0x180000000
    r = deobf.resolve_hidden_branches(buf, va, va, va + len(buf))
    assert r.resolved == 0
    assert r.unresolved == 1


def test_patch_span_covers_only_the_shift_and_jump_tail():
    """De-obfuscation must not delete the live bytes between call and pop."""
    buf = (b"\xe8\x00\x00\x00\x00"
           b"\x33\x9c\x0f\xb8\xd4\xe6\xb5"      # this executes: must be preserved
           b"\x41\x58"
           b"\x49\x81\xc0\x13\x00\x00\x00"
           b"\x41\xff\xe0")
    va = 0x1802A248F
    r = deobf.resolve_hidden_branches(buf, va, va, va + len(buf))
    b = r.branches[0]
    start, end = b.patch_span
    assert start == va + 5 + 7 + 2          # the `add`, not the `call`
    assert end == va + len(buf)
    assert start > b.anchor                 # the call and junk stay untouched
