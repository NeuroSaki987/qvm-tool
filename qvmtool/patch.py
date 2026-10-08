"""Length-preserving de-obfuscation: rewrite branch idioms into direct jumps.

Two transformations, both **length-preserving**, so every other relative offset in
the file stays valid and no relocation or unwind entry is invalidated:

* a real hidden branch  -> ``E9 rel32`` to its resolved target, ``nop`` padded
* a null (fall-through) branch -> ``nop``

Critical design constraint
--------------------------
Only the idiom's **shift+jump tail** is rewritten, never the whole
``call $+5 .. jmp reg`` span. The bytes between the ``call`` and the ``pop`` really
execute and have side effects; deleting them would change behaviour. See
``deobf.HiddenBranch.patch_span``.

Because the rewrite still drops the ``add``'s effect on flags, the result is an
**analysis aid, not a functionally equivalent binary**. Every rewritten range is
recorded in a manifest so the change is auditable and reversible.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

from . import deobf
from .pe import PEMap

NOP = 0x90


@dataclass
class PatchResult:
    input: str = ""
    output: str = ""
    vm_section: str = ""
    idioms_seen: int = 0
    rewritten_real: int = 0
    excised_null: int = 0
    bytes_touched: int = 0
    patches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "patches"}
        d["manifest_entries"] = len(self.patches)
        return d


def depatch(pm: PEMap, out_path: str, vm_section: str = ".qvm0") -> PatchResult:
    """Write a patched copy of `pm`'s file with branch idioms rewritten."""
    sec = pm.section_named(vm_section)
    res = PatchResult(input=pm.path, output=out_path, vm_section=vm_section)
    if sec is None:
        return res

    buf = bytearray(pm.section_bytes(sec))
    va = pm.base + sec.va
    d = deobf.resolve_hidden_branches(bytes(buf), va, va, va + sec.vsize)
    res.idioms_seen = len(d.branches)

    done_until = -1
    for b in sorted(d.branches, key=lambda x: x.patch_span[0]):
        start, end = b.patch_span
        start -= va
        end -= va
        if start < 0 or end > len(buf) or start < done_until or end <= start:
            continue
        span = end - start
        if span < 5:
            continue
        if b.fallthrough:
            buf[start:end] = bytes([NOP]) * span
            res.excised_null += 1
            res.patches.append({"start": hex(va + start), "end": hex(va + end),
                                "kind": "null-branch tail -> nop", "target": None})
        else:
            rel = b.target - (va + start + 5)
            if not (-0x80000000 <= rel < 0x80000000):
                continue
            buf[start:start + 5] = b"\xe9" + struct.pack("<i", rel)
            if span > 5:
                buf[start + 5:end] = bytes([NOP]) * (span - 5)
            res.rewritten_real += 1
            res.patches.append({"start": hex(va + start), "end": hex(va + end),
                                "kind": "hidden branch tail -> jmp rel32",
                                "target": hex(b.target)})
        done_until = end
        res.bytes_touched += span

    out = bytearray(pm.data)
    out[sec.raw:sec.raw + len(buf)] = buf
    import os
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(out)
    return res


def write_manifest(res: PatchResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"info": res.to_dict(), "patches": res.patches}, f, indent=2)


def verify_contained(pm: PEMap, out_path: str, vm_section: str = ".qvm0") -> dict:
    """Confirm the patch changed bytes only inside the VM section and kept the size.

    A de-obfuscation that quietly rewrote a byte in `.text` would corrupt unrelated
    semantics, so this check is part of the deliverable rather than an afterthought.
    """
    sec = pm.section_named(vm_section)
    other = open(out_path, "rb").read()
    if len(other) != len(pm.data):
        return {"size_preserved": False, "differing_bytes": None,
                "outside_vm_section": None}
    lo, hi = sec.raw, sec.raw + sec.rsize
    diff = [i for i in range(len(pm.data)) if pm.data[i] != other[i]]
    outside = [i for i in diff if not (lo <= i < hi)]
    return {
        "size_preserved": True,
        "differing_bytes": len(diff),
        "outside_vm_section": len(outside),
    }
