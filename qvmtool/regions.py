"""Region classification inside the VM section, and encrypted-band location.

Two jobs:

  1. Classify the VM section into contiguous runs so the real structure is visible
     (megabyte-scale mutated code, a relocated unwind table, encrypted blobs).
  2. Locate high-entropy bands image-wide -- these are what actually needs
     unpacking. Code in this family is plaintext (the get-PC idiom cannot occur by
     chance), so high entropy means encrypted *data*, not packed code.
"""

from __future__ import annotations

import collections
import math

from .model import Region, Section
from .pe import PEMap

BLOCK = 0x1000
ENCRYPTED_ENTROPY = 7.5


def entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    counts = collections.Counter(buf)
    n = len(buf)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def _self_pc_sites(block: bytes) -> int:
    n = block.count(b"\xe8\x00\x00\x00\x00") + block.count(b"\xe9\x00\x00\x00\x00")
    n += sum(block.count(bytes([0x0F, 0x80 + k]) + b"\x00\x00\x00\x00") for k in range(16))
    return n


#: Byte patterns that can only come from plaintext machine code, whatever the
#: engine. The QVM test ("eight 5-byte get-PC idioms in a 4 KB block") has no power
#: on `.tvm0`, whose engine never emits that idiom at all, so a TVM-aware code marker
#: is needed or megabytes of handler code get filed as ciphertext. The pattern set is
#: generated over all 16 GPRs in `tvm.code_marker_patterns` — a marker that only
#: responded to `r9` would report "no code" for SGuardAgent64.dll, which has 481 VIP
#: reloads through a different scratch register.
def tvm_code_markers(block: bytes, slot: int | None = None) -> int:
    """Count TVM-specific plaintext-code patterns in a block."""
    from . import tvm as tvm_mod
    if slot is None:
        slot = tvm_mod.VIP_SLOT_KNOWN
    n = sum(block.count(p) for p in tvm_mod.code_marker_patterns(slot))
    return n + tvm_mod.count_16bit_dereferences(block)


def classify_block(block: bytes, code_markers=None,
                   marker_minimum: int = 8) -> tuple[str, float]:
    """Classify one block.

    Idiom density is tested BEFORE entropy, and before the padding test, because a
    run of `call $+5` cells is simultaneously low-entropy and unambiguously code.
    Only a block with no idioms at all can be padding (zeros / 0xCC) or encrypted
    data. Note the idiom patterns are 5-6 specific bytes, so random or encrypted
    data cannot produce eight of them by chance.

    `code_markers` selects the engine's own "this is code" pattern set. The default
    keeps the QVM-era behaviour byte-for-byte (the module's own test asserts it); a
    TVM caller passes `tvm_code_markers`. The reasoning is unchanged -- the argument
    is only *which* patterns count as proof of plaintext code -- and the counts are
    just as impossible to produce by chance: `mov r9,[rbp+60h]` is 4 specific bytes
    that occur 1,264 times in ACE-GAME's 1.7 MB `.tvm0`.
    """
    h = entropy(block)
    idioms = code_markers(block) if code_markers else _self_pc_sites(block)
    zeros = block.count(0) / len(block)
    if idioms >= marker_minimum:
        return "code", h
    if h < 3.0:
        return "zero", h
    if h > 6.8 and zeros < 0.12:
        return "encrypted", h
    return "sparse", h


def vm_regions(pm: PEMap, vm_section: str, code_markers=None) -> list[Region]:
    """Contiguous same-class runs across the VM section.

    `code_markers` is the engine's plaintext-code test (see `classify_block`).
    """
    sec = pm.section_named(vm_section)
    if sec is None:
        return []
    raw = pm.section_bytes(sec)
    seq: list[tuple[int, str, float]] = []
    for i in range(0, len(raw) - BLOCK + 1, BLOCK):
        cls, h = classify_block(raw[i:i + BLOCK], code_markers=code_markers)
        seq.append((sec.va + i, cls, h))

    out: list[Region] = []
    for rva, cls, h in seq:
        if out and out[-1].kind == cls and out[-1].end == rva:
            prev = out[-1]
            n = (prev.end - prev.start) // BLOCK
            out[-1] = Region(prev.section, prev.start, rva + BLOCK, cls,
                             (prev.entropy * n + h) / (n + 1))
        else:
            out.append(Region(sec.name, rva, rva + BLOCK, cls, h))
    return out


def encrypted_bands(pm: PEMap, threshold: float = ENCRYPTED_ENTROPY) -> list[Region]:
    """Runs of blocks whose entropy exceeds `threshold`, in every section."""
    out: list[Region] = []
    for sec in pm.sections:
        raw = pm.section_bytes(sec)
        if len(raw) < BLOCK:
            continue
        runs: list[list[int]] = []
        hs: list[float] = []
        for i in range(0, len(raw) - BLOCK + 1, BLOCK):
            h = entropy(raw[i:i + BLOCK])
            hs.append(h)
            if h > threshold:
                if runs and runs[-1][1] == i // BLOCK - 1:
                    runs[-1][1] = i // BLOCK
                else:
                    runs.append([i // BLOCK, i // BLOCK])
        for a, b in runs:
            seg = hs[a:b + 1]
            out.append(Region(sec.name, sec.va + a * BLOCK, sec.va + (b + 1) * BLOCK,
                              "encrypted", sum(seg) / len(seg)))
    out.sort(key=lambda r: -r.size)
    return out


def annotate_unwind_table(pm: PEMap, regions: list[Region], exc_rva: int, exc_size: int
                          ) -> list[Region]:
    """Relabel the region occupied by the relocated RUNTIME_FUNCTION array."""
    lo, hi = exc_rva, exc_rva + exc_size
    out: list[Region] = []
    for r in regions:
        if r.start >= lo and r.end <= hi:
            out.append(Region(r.section, r.start, r.end, "unwind", r.entropy))
        elif r.end <= lo or r.start >= hi:
            out.append(r)
        else:
            if r.start < lo:
                out.append(Region(r.section, r.start, lo, r.kind, r.entropy))
            out.append(Region(r.section, max(r.start, lo), min(r.end, hi), "unwind",
                              r.entropy))
            if r.end > hi:
                out.append(Region(r.section, hi, r.end, r.kind, r.entropy))
    return out


def window_profile(pm: PEMap, section: Section) -> list[tuple[int, float, int]]:
    """(rva, entropy, self-pc-idiom count) per block -- for profiling/reporting."""
    raw = pm.section_bytes(section)
    out = []
    for i in range(0, len(raw) - BLOCK + 1, BLOCK):
        b = raw[i:i + BLOCK]
        out.append((section.va + i, entropy(b), _self_pc_sites(b)))
    return out
