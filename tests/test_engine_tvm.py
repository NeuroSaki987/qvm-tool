"""Tests for the engine-profile layer and the TVM-specific analysis.

These follow the house rule of `test_units.py`: no sample binary is required, so the
answers are provable from constructed bytes. The byte sequences are the real ones
observed in the corpus (quoted from `out/g6/g6_dump_site.txt` and the reference
report), not invented shapes -- a test built on an invented shape is what let the
`add`-vs-`lea` confusion in `deobf` survive a round.
"""

from __future__ import annotations

from qvmtool import engine, regions, tvm


# ----------------------------------------------------------- engine profiles

def test_profiles_are_distinct_on_the_axes_that_matter():
    qvm = engine.profile_for_section(".qvm0")
    tv = engine.profile_for_section(".tvm0")
    assert qvm.name == "qvm" and tv.name == "tvm"
    # the four axes that the QVM-era code wrongly treated as engine-independent
    assert qvm.pdata_expectation == "encrypted"
    assert tv.pdata_expectation == "stale-plaintext"
    assert qvm.native_entry_form == "call-rel32"
    assert tv.native_entry_form == "e9-trampoline"
    assert tv.bytecode_unit == 2 and qvm.bytecode_unit is None
    assert tv.sample_anchors != qvm.sample_anchors
    # neither has a handler table -- that is shared, not a discriminator
    assert not qvm.has_handler_table and not tv.has_handler_table


def test_unknown_section_gets_a_fallback_profile():
    p = engine.profile_for_section(None)
    assert p.name == "unknown" and p.sample_anchors == ("call-self-pc",)


# ------------------------------------------------------------- fetch census

def test_fetch_census_counts_the_canonical_vip_pair():
    """`mov r9,[rbp+60h]` + `mov r8w,[r9]`, twice, plus one publish."""
    buf = (b"\x4c\x8b\x4d\x60"          # mov r9,[rbp+60h]
           b"\x66\x45\x8b\x01"          # mov r8w,[r9]
           b"\x4c\x8b\x4d\x60"
           b"\x66\x45\x8b\x01"
           b"\x4c\x89\x45\x60")         # mov [rbp+60h],r8
    c = tvm.fetch_idiom_census(buf)
    assert c["vip_loads"] == 2
    assert c["vip_loads_by_register"] == {"vip_load_r9": 2}
    assert c["vip_publishes"] == 1
    assert c["mem16_dereferences"] == 2
    assert c["canonical_pair_present"] is True


def test_fetch_census_is_zero_on_qvm_style_padding():
    """A run of `call $+5` must not look like a bytecode fetch."""
    buf = b"\xe8\x00\x00\x00\x00" * 64
    c = tvm.fetch_idiom_census(buf)
    assert c["vip_loads"] == 0 and c["mem16_dereferences"] == 0
    assert c["canonical_pair_present"] is False


# --------------------------------------------------------- VIP slot discovery

def test_vip_slot_is_discovered_from_the_load_then_deref_shape():
    """The slot is found by behaviour, not by hard-coding 0x60.

    A decoy slot 0x28 is loaded three times and never dereferenced; the real slot
    0x60 is loaded once and immediately used to read a 16-bit word.
    """
    buf = (b"\x48\x8b\x4d\x28" * 3                  # mov rcx,[rbp+28h] x3, no deref
           + b"\x4c\x8b\x4d\x60" + b"\x66\x45\x8b\x01")  # mov r9,[rbp+60h]; [r9]
    ranked = tvm.discover_vip_slot(buf)
    assert ranked[0]["slot"] == 0x60
    assert ranked[0]["loads"] == 1
    assert ranked[0]["load_then_16bit_deref"] == 1


def test_discover_vip_slot_ranks_a_different_offset_when_that_is_where_it_is():
    buf = b"\x48\x8b\x4d\x20" + b"\x66\x44\x8b\x01"    # mov rcx,[rbp+20h]; [rcx]
    ranked = tvm.discover_vip_slot(buf)
    assert ranked and ranked[0]["slot"] == 0x20


# ------------------------------------------------------- dispatch epilogues

def _epilogue(site_va_delta: int = 0) -> bytes:
    """The real ACE-GAME epilogue at 0x140185518:

        push r8
        mov  r8, 1401C1F82h
        pushfq
        add  r8, -2E0E1h
        popfq
        jmp  r8              -> 0x140193EA1
    """
    return (b"\x41\x50"                                  # push r8
            b"\x49\xb8\x82\x1f\x1c\x40\x01\x00\x00\x00"  # mov r8,1401C1F82h
            b"\x9c"                                      # pushfq
            b"\x49\x81\xc0\x1f\x1f\xfd\xff"              # add r8,-2E0E1h
            b"\x9d"                                      # popfq
            b"\x41\xff\xe0")                             # jmp r8


def test_threaded_epilogue_resolves_through_the_flags_neutral_add():
    """The whole point: `mov imm64` is the BASE, not the target."""
    buf = _epilogue()
    site = len(buf) - 3
    got = tvm.resolve_epilogue(buf, site, "r8", 0x140000000)
    assert got is not None
    target, form, anchor, delta, shift = got
    assert form == "threaded-imm64"
    assert delta == -0x2E0E1
    assert target == 0x1401C1F82 - 0x2E0E1 == 0x140193EA1


def test_epilogue_requires_the_popfq_sandwich():
    """Without `popfq ;` directly before the jmp the shape is not claimed."""
    buf = bytearray(_epilogue())
    buf[-3 - 1] = 0x90                      # replace the popfq
    site = len(buf) - 3
    assert tvm.resolve_epilogue(bytes(buf), site, "r8", 0x140000000) is None


def test_epilogue_requires_the_shift_to_land_on_the_popfq():
    buf = _epilogue()
    site = len(buf) - 3
    # an `add r8, imm32` that does not close exactly on the popfq must not be used
    buf2 = bytearray(buf)
    del buf2[site - 9]                      # drop one imm byte -> shift no longer fits
    assert tvm.resolve_epilogue(bytes(buf2), site - 1, "r8", 0x140000000) is None


def test_alignment_skip_branch_is_not_a_real_edge():
    """`target == falling-through + 1` is CFG-neutral: it skips one junk byte.

    Real bytes, ACE-GAME 0x1400D0144:

        41 FF E0 | E8 41 58 48 91
        jmp r8     | a decoder reads `call rel32` here, but the branch lands on
                     0x1400D0148 where `41 58` is `pop r8`.
    """
    base = 0x1400D0000
    epilogue = (b"\x41\x50"                                   # push r8
                + b"\x49\xb8" + (0x1400D013E).to_bytes(8, "little")
                + b"\x9c"
                + b"\x49\x81\xc0" + (0x0A).to_bytes(4, "little")
                + b"\x9d"
                + b"\x41\xff\xe0")                            # jmp r8  (3 bytes)
    site_off = 0x144
    buf = b"\x90" * (site_off - (len(epilogue) - 3)) + epilogue \
        + b"\xe8\x41\x58\x48\x91"                             # decoy operand bytes
    site = base + site_off
    got = tvm.resolve_epilogue(buf, site_off, "r8", base)
    assert got is not None
    target, _form, _anchor, delta, _shift = got
    assert delta == 0x0A
    assert target == site + 4 == site + 3 + 1     # fall-through + 1
    assert target != site + 3
    # and the two-byte-jmp variant lands on fall-through + 1 as well
    succ = site + 3
    assert not (target == succ) and target == succ + 1


def test_epilogue_site_anchor_requires_popfq_immediately_before():
    assert tvm.epilogue_sites(_epilogue()) == [(len(_epilogue()) - 3, "r8")]
    assert tvm.epilogue_sites(b"\xff\xe0") == []          # bare jmp rax, no popfq


def test_dispatch_summary_separates_real_edges_from_null_branches():
    from qvmtool.tvm import DispatchResult, DispatchSite
    res = DispatchResult(epilogue_candidates=3, jmp_reg_sites=5)
    res.branches = [
        DispatchSite(site=0x1000, register="r8", target=0x2000, form="threaded-imm64",
                     delta=1, anchor=0x1010, jmp_len=3),
        DispatchSite(site=0x3000, register="r8", target=0x3003, form="threaded-imm64",
                     delta=1, anchor=0x3010, jmp_len=3, fallthrough=True),
        DispatchSite(site=0x4000, register="r8", target=0x4004, form="threaded-imm64",
                     delta=1, anchor=0x4010, jmp_len=3, alignment_skip=True),
    ]
    s = res.summary()
    assert s["real_edges"] == 1
    assert s["null_branches"] == 2
    assert s["fallthrough_only"] == 1 and s["alignment_skip_only"] == 1


# --------------------------------------------------------------- code marker

def test_tvm_code_markers_recognise_handler_code_not_ciphertext():
    code = (b"\x4c\x8b\x4d\x60\x66\x45\x8b\x01" * 8)
    assert regions.tvm_code_markers(code) == 16


def test_classify_block_default_behaviour_is_unchanged():
    """The engine-aware parameter must be opt-in: `.qvm0` results cannot move."""
    assert regions.classify_block(b"\xe8\x00\x00\x00\x00" * 40)[0] == "code"
    assert regions.classify_block(b"\x00" * 4096)[0] == "zero"
    enc = bytes((i * 37 + 11) % 256 for i in range(4096))
    assert regions.classify_block(enc)[0] == "encrypted"


def test_classify_block_with_tvm_markers_rescues_dense_handler_code():
    """Dense plaintext TVM code trips the entropy threshold; the marker saves it."""
    import random
    rng = random.Random(11)
    block = bytearray(rng.randrange(256) for _ in range(4096))
    for i in range(20):
        off = i * 200
        block[off:off + 4] = b"\x4c\x8b\x4d\x60"
    blob = bytes(block)
    assert regions.classify_block(blob)[0] == "encrypted"
    assert regions.classify_block(
        blob, code_markers=regions.tvm_code_markers)[0] == "code"


# ------------------------------------------------------------------ carrier

def test_tvm_carrier_verdict_is_stack_when_a_vip_slot_exists():
    ranked = [{"slot": 0x60, "loads": 1264, "stores": 1955,
               "load_then_16bit_deref": 1034},
              {"slot": 0x4E, "loads": 2, "stores": 0, "load_then_16bit_deref": 2}]
    sc = tvm.assess_carrier(".tvm0", ranked, {"distinct_slots_loaded": 4,
                                             "distinct_slots_stored": 18,
                                             "largest_stride8_write_ladder": 14})
    assert sc.carrier == "stack"
    assert any("RBP" in n for n in sc.notes)


def test_tvm_carrier_is_unknown_without_a_vip_slot():
    sc = tvm.assess_carrier(".tvm0", [], {})
    assert sc.carrier == "unknown"
    assert any("not present" in n for n in sc.notes)


def test_frame_census_counts_the_guest_register_ladder():
    buf = b"".join(bytes([0x48, 0x89, 0x45 | ((i & 7) << 3), off])
                   for i, off in enumerate(range(8, 0x88, 8)))
    c = tvm.context_frame_census(buf)
    assert c["largest_stride8_write_ladder"] >= 8
