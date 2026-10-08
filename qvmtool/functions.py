"""Recovery of the function map from the image's Exception directory.

This is the single most valuable artefact for these binaries. The vendor moves the
RUNTIME_FUNCTION[] array out of `.pdata` and into the tail of the VM section, so:

  * the `.pdata` SECTION is a decoy filled with encrypted bytes, and
  * the only authoritative function map is reachable through the Exception
    directory data directory, which the Windows loader itself still honours.

Parsing it therefore yields real function boundaries for the whole image without
depending on the disassembler at all -- which matters because these images
desynchronise a linear sweep within a few hundred bytes.
"""

from __future__ import annotations

from typing import Optional

from .model import Function
from .pe import IMAGE_DIR_EXCEPTION, PEMap

RUNTIME_FUNCTION_SIZE = 12


def recover_functions(pm: PEMap) -> list[Function]:
    """Parse the Exception directory into a de-duplicated, sanity-checked list."""
    exc = pm.data_directory(IMAGE_DIR_EXCEPTION)
    if exc is None or not exc.VirtualAddress or not exc.Size:
        return []

    count = exc.Size // RUNTIME_FUNCTION_SIZE
    blob = pm.read_rva(exc.VirtualAddress, count * RUNTIME_FUNCTION_SIZE)
    if blob is None:
        # The array can straddle into bytes past SizeOfRawData; read what we can.
        blob = pm.read_rva(exc.VirtualAddress, 0)
        if not blob:
            return []

    out: list[Function] = []
    for i in range(len(blob) // RUNTIME_FUNCTION_SIZE):
        begin, end, unwind = (
            int.from_bytes(blob[i * 12:i * 12 + 4], "little"),
            int.from_bytes(blob[i * 12 + 4:i * 12 + 8], "little"),
            int.from_bytes(blob[i * 12 + 8:i * 12 + 12], "little"),
        )
        if begin == 0 and end == 0:
            continue
        if not plausible(pm, begin, end):
            continue
        out.append(Function(begin, end, unwind, pm.section_name_for_rva(begin)))

    out.sort(key=lambda f: f.begin)
    return dedupe(out)


def plausible(pm: PEMap, begin: int, end: int) -> bool:
    if begin >= end:
        return False
    sec = pm.section_for_rva(begin)
    if sec is None:
        return False
    if not sec.is_exec:
        return False
    # end must stay inside the image
    return pm.section_for_rva(end - 1) is not None


def dedupe(funcs: list[Function]) -> list[Function]:
    seen: set[tuple[int, int]] = set()
    out: list[Function] = []
    for f in funcs:
        key = (f.begin, f.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def owner_of(funcs: list[Function], rva: int) -> Optional[Function]:
    for f in funcs:
        if f.begin <= rva < f.end:
            return f
    return None


def function_index(funcs: list[Function]) -> dict[int, Function]:
    return {f.begin: f for f in funcs}
