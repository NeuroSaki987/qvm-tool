"""Engine detection: decide whether an image is QVM/TVM/VMP protected, and why.

Detection is deliberately multi-signal and each signal is recorded as Evidence so
a verdict can always be audited. No single signal is trusted:

  S1  a vendor VM section exists (.qvm0/.tvm0/.vmp0/.rubyN)          weak
  S2  the Exception directory is relocated INTO the VM section       strong
      while a .pdata section still exists and its content is encrypted
  S3  the VM section is CODE|EXEC|READ|NOT_PAGED but NOT writable    medium
  S4  a VM section whose tail is consumed by the relocated           strong
      RUNTIME_FUNCTION array
  S5  get-PC idiom density (E8/E9 00000000, 0F 8x 00000000) far      strong
      above what random or ordinary code produces
  S6  native code branches directly into the VM section              strong

S2 and S4 above are the QVM *shape* of a relocated table: the array at the very tail
of the section, with the `.pdata` section left holding ciphertext. They are recorded
as evidence but they are **not** engine tests, because on TVM the same relocation
looks different in both respects -- `.pdata` keeps a stale *plaintext* copy and the
array ends well short of the section tail. Two discriminating signals were therefore
added, and they are the ones that decide `engine_core`:

  S7a QVM core: get-PC idiom density far above the mutation-noise floor  (QVM only)
  S7b TVM core: a VIP slot reloaded from the RBP-relative context and    (TVM only)
      dereferenced as a 16-bit bytecode word
  S8  `.pdata` still holds a readable RUNTIME_FUNCTION[] that largely    (TVM only)
      overlaps the live table: a stale plaintext copy, not ciphertext

Without S7b/S8 a `.tvm0` image scores 0.40 against a `.qvm0` image's 0.95 purely
because two QVM-specific claims cannot fire, which is a calibration bug rather than
a property of the sample.
"""

from __future__ import annotations

import collections
import math

from . import engine as engine_mod
from . import tvm as tvm_mod
from .model import Verdict
from .pe import ENGINE_SECTIONS, IMAGE_DIR_EXCEPTION, PEMap

#: Bytes E8 00 00 00 00 / E9 00 00 00 00 -- `call $+5` / `jmp $+5` get-PC idioms.
CALL_5 = b"\xe8\x00\x00\x00\x00"
JMP_5 = b"\xe9\x00\x00\x00\x00"
JCC_6 = [bytes([0x0F, 0x80 + k]) + b"\x00\x00\x00\x00" for k in range(16)]

#: Self-PC sites per MB above which the idiom is the engine's own core mechanism
#: rather than mutation noise. Measured: `.qvm0` reference sample ~4,900/MB; the
#: highest `.tvm0` module in the corpus 10.6/MB. Three orders of magnitude apart, so
#: the exact cut matters far less than having one at all.
SELF_PC_CORE_DENSITY = 100.0

#: Minimum evidence for "this module really contains the TVM interpreter core": VIP
#: reloads that are immediately dereferenced as a 16-bit word. Lives in `tvm` so the
#: carrier verdict cannot use a different bar from the detection verdict.
VIP_CORE_MIN = tvm_mod.CORE_MIN_DEREFS


def entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    counts = collections.Counter(buf)
    n = len(buf)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def sample_entropy(buf: bytes, cap: int = 1 << 20) -> float:
    return entropy(buf[:cap])


def idiom_census(buf: bytes) -> dict[str, int]:
    """Byte-level (desync-immune) census of the injector's signature idioms."""
    return {
        "call_self_pc": buf.count(CALL_5),
        "jmp_self_pc": buf.count(JMP_5),
        "jcc_self_pc": sum(buf.count(p) for p in JCC_6),
        "pushfq": buf.count(0x9C),
        "popfq": buf.count(0x9D),
        "xchg": buf.count(0x87),
    }


def detect(pm: PEMap) -> Verdict:
    v = Verdict()

    vm_secs = pm.vm_sections()
    if not vm_secs:
        v.add("S1 no vendor VM section", "no .qvm0/.tvm0/.vmp0/.rubyN present")
        return v

    vm = max(vm_secs, key=lambda s: s.vsize)
    v.vm_section = vm.name
    v.vm_range = (vm.va, vm.va + vm.vsize)
    v.engine = ENGINE_SECTIONS[vm.name]
    v.add("S1 vendor VM section", f"{vm.name} VA=0x{vm.va:X} VSize=0x{vm.vsize:X}", 0.10)

    # S3 -- section flags
    if vm.is_exec and not vm.is_write:
        v.add("S3 VM section is exec, read, not-paged, NOT writable",
              f"Characteristics=0x{vm.chars:08X}", 0.05)

    # S2/S4 -- relocated exception directory
    exc = pm.data_directory(IMAGE_DIR_EXCEPTION)
    if exc is not None and exc.VirtualAddress:
        holder = pm.section_name_for_rva(exc.VirtualAddress)
        v.add("S5 exception directory location",
              f"RVA=0x{exc.VirtualAddress:X} size=0x{exc.Size:X} -> section {holder}")
        if holder == vm.name:
            v.add("S2 Exception directory relocated INTO the VM section",
                  f"exception RVA 0x{exc.VirtualAddress:X} lies in {vm.name}; "
                  f"the loader therefore reads unwind info from the VM payload",
                  0.25)
            pd = pm.section_named(".pdata")
            if pd is not None:
                h = sample_entropy(pm.section_bytes(pd))
                if h > 7.0:
                    v.add("S2 .pdata section still exists but is encrypted",
                          f".pdata H={h:.3f} over {pd.rsize} bytes", 0.20)
                else:
                    v.add("S2 .pdata section present and readable",
                          f".pdata H={h:.3f}", 0.0)
            tail = vm.va + vm.vsize
            if abs((exc.VirtualAddress + exc.Size) - tail) <= 0x1000:
                v.add("S4 relocated function table ends at the VM section tail",
                      f"0x{exc.VirtualAddress + exc.Size:X} vs section end 0x{tail:X}",
                      0.10)

    # S5 -- idiom density
    buf = pm.section_bytes(vm)
    cen = idiom_census(buf)
    total_self = cen["call_self_pc"] + cen["jmp_self_pc"] + cen["jcc_self_pc"]
    per_mb = total_self / max(1, len(buf)) * (1 << 20)
    v.add("S6 get-PC idiom density in VM section",
          f"{total_self} sites over {len(buf)} bytes = {per_mb:,.0f}/MB "
          f"(random data would give ~0; P(5 specific bytes)=2^-40)",
          min(0.25, per_mb / 3000 * 0.25))

    # S7/S8 -- which CORE is actually present. This is the part that discriminates
    # the two engines instead of lumping them together by section name.
    _core_evidence(pm, vm, buf, per_mb, v)

    profile = engine_mod.profile_for_section(vm.name)
    if profile.name != "unknown" and v.engine_core == "section-only":
        v.add("S9 VM section present but no engine core located",
              f"`{vm.name}` exists and relocates the function table, but the "
              f"{profile.name} core mechanism "
              f"({', '.join(profile.core_signals)}) was not found in it -- either a "
              f"decoy section, a different VM variant, or a module that only carries "
              f"the bytecode half")
    return v


def _core_evidence(pm: PEMap, vm, buf: bytes, per_mb: float, v: Verdict) -> None:
    """Fill in `Verdict.engine_core` and its supporting evidence."""
    v.engine_core = "section-only"

    # ---- QVM core: the get-PC idiom IS the engine.
    if per_mb >= SELF_PC_CORE_DENSITY:
        v.engine_core = "qvm-mutation"
        v.add("S7a QVM mutation core — self-PC idiom is the engine's own mechanism",
              f"{per_mb:,.0f} self-PC sites/MB, far above the mutation floor "
              f"({SELF_PC_CORE_DENSITY:,.0f}/MB); the engine resolves constants "
              f"through `call $+5; ...; add reg,imm; jmp reg`",
              0.20)
        return

    # ---- TVM core: a VIP slot in the RBP-relative context + a 16-bit fetch.
    ranked = tvm_mod.discover_vip_slot(buf)
    top = ranked[0] if ranked else None
    if top is not None and top["load_then_16bit_deref"] >= VIP_CORE_MIN:
        rival = (f"; next candidate slot 0x{ranked[1]['slot']:X} has "
                 f"{ranked[1]['load_then_16bit_deref']}" if len(ranked) > 1 else "")
        v.engine_core = "tvm-threaded"
        v.add("S7b TVM threaded-interpreter core — VIP slot in the RBP context",
              f"slot [rbp+0x{top['slot']:X}] is loaded {top['loads']:,}x and "
              f"dereferenced as a 16-bit word {top['load_then_16bit_deref']:,}x"
              f"{rival}; state is a per-function stack frame, not a heap struct",
              0.25)
        cen_tvm = tvm_mod.fetch_idiom_census(buf, top["slot"])
        if cen_tvm["mem16_dereferences"] and cen_tvm["vip_publishes"]:
            v.add("S7b bytecode fetch idiom present (16-bit words)",
                  f"VIP slot published {cen_tvm['vip_publishes']:,}x, "
                  f"{cen_tvm['mem16_dereferences']:,} 16-bit dereferences, "
                  f"advance-by-2 sites {cen_tvm['advance_by_2']:,}")
        epi = tvm_mod.epilogue_sites(buf)
        if epi:
            v.add("S7b threaded dispatch epilogues present",
                  f"{len(epi):,} `popfq ; jmp reg` handler exits: dispatch is "
                  f"threaded, so there is no handler table even though there IS a "
                  f"bytecode fetch")
    _stale_pdata(pm, v)


def _stale_pdata(pm: PEMap, v: Verdict) -> None:
    """S8: `.pdata` holds a readable, largely-live RUNTIME_FUNCTION[].

    On QVM the `.pdata` section is ciphertext (H 7.9) and unreadable; on every `.tvm0`
    module measured it is plaintext (H 4.9-6.8) and its entries mostly appear in the
    live table too. That makes "the decoy section is merely stale" a TVM-shaped
    signal, and its absence one of the two reasons a QVM image can never be mistaken
    for TVM by score alone.
    """
    pd = pm.section_named(".pdata")
    if pd is None:
        v.add("S8 no .pdata section", "the image carries no unwind-table section "
                                      "besides the relocated one")
        return
    h = sample_entropy(pm.section_bytes(pd))
    if h >= 7.0:
        return                              # ciphertext: nothing to compare
    entries = _pdata_entries(pm)
    if not entries:
        return
    exc = pm.data_directory(IMAGE_DIR_EXCEPTION)
    live = set()
    if exc is not None and exc.VirtualAddress:
        blob = pm.read_rva(exc.VirtualAddress, exc.Size)
        if blob:
            for i in range(0, len(blob) - 11, 12):
                live.add((int.from_bytes(blob[i:i + 4], "little"),
                          int.from_bytes(blob[i + 4:i + 8], "little"),
                          int.from_bytes(blob[i + 8:i + 12], "little")))
    shared = len(entries & live)
    share = shared / len(entries)
    if share >= 0.5:
        v.add("S8 .pdata is a stale PLAINTEXT copy of the live table",
              f".pdata H={h:.3f}, {len(entries):,} RUNTIME_FUNCTIONs of which "
              f"{shared:,} ({share:.1%}) also appear in the live array; "
              f"{len(entries - live):,} are stale-only. Not ciphertext, and not the "
              f"authoritative map either", 0.15)


def _pdata_entries(pm: PEMap) -> set[tuple[int, int, int]]:
    sec = pm.section_named(".pdata")
    if sec is None:
        return set()
    raw = pm.section_bytes(sec)
    out = set()
    for i in range(0, len(raw) - 11, 12):
        b = int.from_bytes(raw[i:i + 4], "little")
        e = int.from_bytes(raw[i + 4:i + 8], "little")
        u = int.from_bytes(raw[i + 8:i + 12], "little")
        if b == 0 and e == 0:
            continue
        out.add((b, e, u))
    return out
