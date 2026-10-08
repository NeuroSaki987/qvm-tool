#!/usr/bin/env python3
"""Task 4: emit a de-obfuscated copy of the image for a second Ghidra pass.

Two transformations, both **length-preserving** so that every other relative
offset in the file stays valid and no relocation or RUNTIME_FUNCTION entry is
invalidated:

  1. Each obfuscated branch idiom is replaced by an equivalent direct `jmp rel32`
     padded with `nop`:
       * real branches  -> `E9 <rel32>` to the resolved target
       * null padding   -> `EB 00`-style true nop sled (the idiom computed the
                           fall-through, so plain nops are exactly equivalent)
     This removes the anti-disassembly structure that stops Ghidra (and every
     linear sweeper) from following the code.

  2. The `call $+5` bytes that belong to a replaced idiom are consumed by the
     rewrite, so they no longer push a return address that nothing pops.

The output is a full PE with only `.qvm0` modified, plus a manifest listing every
byte range rewritten, so the transformation is auditable and reversible.
"""
from __future__ import annotations

import json
import struct
import os
import sys

# ensure `import qvmtool` resolves: this file lives in the repo, so run
# it as `python scripts/<name>` from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qvmtool import deobf
from qvmtool.pe import PEMap

NOP = 0x90


def build(path: str, out_path: str, manifest_path: str) -> dict:
    pm = PEMap(path)
    sec = pm.section_named(".qvm0")
    buf = bytearray(pm.section_bytes(sec))
    va = pm.base + sec.va
    lo, hi = va, va + sec.vsize

    d = deobf.resolve_hidden_branches(bytes(buf), lo, lo, hi)
    manifest = []
    patched_real = 0
    patched_null = 0

    # Process in ascending order; idioms never overlap (each starts at a distinct
    # `call $+5`), but guard anyway by tracking the highest written end.
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
            patched_null += 1
            manifest.append({"start": hex(va + start), "end": hex(va + end),
                             "kind": "null-branch tail -> nop", "target": None})
        else:
            rel = b.target - (va + start + 5)
            if not (-0x80000000 <= rel < 0x80000000):
                continue
            buf[start:start + 5] = b"\xe9" + struct.pack("<i", rel)
            if span > 5:
                buf[start + 5:end] = bytes([NOP]) * (span - 5)
            patched_real += 1
            manifest.append({"start": hex(va + start), "end": hex(va + end),
                             "kind": "hidden branch tail -> jmp rel32",
                             "target": hex(b.target)})
        done_until = end

    out = bytearray(pm.data)
    out[sec.raw:sec.raw + len(buf)] = buf
    with open(out_path, "wb") as f:
        f.write(out)
    info = {
        "input": path,
        "output": out_path,
        "vm_section": sec.name,
        "idioms_seen": len(d.branches),
        "rewritten_real_branches": patched_real,
        "excised_null_padding": patched_null,
        "bytes_touched": sum(int(m["end"], 16) - int(m["start"], 16)
                             for m in manifest),
        "manifest_entries": len(manifest),
    }
    with open(manifest_path, "w") as f:
        json.dump({"info": info, "patches": manifest}, f, indent=2)
    return info


if __name__ == "__main__":
    info = build(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(info, indent=2))
