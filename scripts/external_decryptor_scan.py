#!/usr/bin/env python3
"""Test the external-decryptor hypothesis for ACE-PBC-Game64.dll's `.qvm0`.

Subagent A established, with five independent negatives, that **this module cannot
decrypt its own VM section**:

  * `.qvm0` lacks MEM_WRITE (Characteristics 0x68000020)
  * `VirtualProtect` / `VirtualProtectEx` / `NtProtectVirtualMemory` /
    `ZwProtectVirtualMemory` / `FlushInstructionCache` are NOT imported at all
  * zero static references into the `.qvm0` VA range anywhere in the image
  * no `rep movs/stos` bulk-copy loop
  * DllMain is the stock CRT `_DllMainCRTStartup`; the TLS callback is the stock
    CRT thread-attach walk with an empty initialiser table

Leading hypothesis therefore: **a sibling ACE module** decrypts/maps the VM section
(it loads `ACE-PBC-Game64.dll` into its own process and can flip page protections).

This script tests that hypothesis directly: for every sibling module, does it import
a page-protection API, and does it contain a write loop whose stores could reach
another image? A sibling that both imports `VirtualProtect` and touches foreign
images is the decryptor candidate.
"""
from __future__ import annotations

import glob
import os
import sys

import pefile

PROTECT_APIS = (
    "virtualprotect", "virtualprotectex", "ntprotectvirtualmemory",
    "zwprotectvirtualmemory", "flushinstructioncache",
)
ALLOC_APIS = ("virtualalloc", "virtualallocex", "ntallocatevirtualmemory",
              "zwallocatevirtualmemory", "mapviewoffile", "createsection",
              "ntmapviewofsection", "zwmapviewofsection")
WRITE_APIS = ("writeprocessmemory", "ntwritevirtualmemory",
              "zwritevirtualmemory", "ntmapviewofsection")
LOAD_APIS = ("loadlibrary", "loadlibraryw", "loadlibraryexw", "loadlibraryexa",
             "getprocaddress")


def api_hits(pe) -> dict[str, list[str]]:
    """Map api-name -> importing dll for every import in the image."""
    out: dict[str, list[str]] = {}
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
        dll = entry.dll.decode(errors="replace").lower()
        for imp in entry.imports:
            if not imp.name:
                continue
            out.setdefault(imp.name.decode(errors="replace").lower(), []).append(dll)
    return out


def main(root: str) -> None:
    files = []
    for pat in ("**/*.dll", "**/*.sys", "**/*.exe"):
        files += glob.glob(os.path.join(root, pat), recursive=True)
    files = sorted(set(files))
    print(f"== external-decryptor hypothesis scan over {len(files)} modules")
    print(f"   root: {root}\n")

    rows = []
    for path in files:
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        except Exception:
            continue
        hits = api_hits(pe)
        prot = sorted({n for n in hits if n in PROTECT_APIS})
        alloc = sorted({n for n in hits if n in ALLOC_APIS})
        write = sorted({n for n in hits if n in WRITE_APIS})
        load = sorted({n for n in hits if n in LOAD_APIS})
        secs = [s.Name.rstrip(b"\0").decode("latin1") for s in pe.sections]
        has_qvm = ".qvm0" in secs
        rows.append((os.path.basename(path), len(files), prot, alloc, write, load,
                     has_qvm, os.path.getsize(path)))

    print("== modules importing a page-protection API ==")
    any_prot = False
    for name, _n, prot, alloc, write, load, has_qvm, size in rows:
        if not prot:
            continue
        any_prot = True
        tag = " <== .qvm0 OWNER" if has_qvm else ""
        print(f"   {name:<26} {size:>10}  PROTECT={prot} ALLOC={alloc} "
              f"WRITE={write}{tag}")

    if not any_prot:
        print("   (none)")

    print("\n== candidates that could be the external decryptor ==")
    cands = []
    for name, _n, prot, alloc, write, load, has_qvm, size in rows:
        if has_qvm:
            continue
        # needs a protection or mapping primitive AND a way to reach the target
        if (prot or write) and (alloc or load):
            score = len(prot) * 3 + len(write) * 3 + len(alloc) * 2 + len(load)
            cands.append((score, name, size, prot, alloc, write, load))
    cands.sort(reverse=True)
    for score, name, size, prot, alloc, write, load in cands[:12]:
        print(f"   score={score:<3} {name:<26} {size:>10}")
        print(f"        protect={prot or '-'}")
        print(f"        map/alloc={alloc or '-'}   write={write or '-'}")
        print(f"        load={load or '-'}")
    if not cands:
        print("   (none)")

    print("\n-- which module LOADS the target? --")
    for name, _n, prot, alloc, write, load, has_qvm, size in rows:
        if not load:
            continue
        if "loadlibrary" in " ".join(load):
            print(f"   {name:<26} uses {load}")
    print("\n   NOTE: a static import of LoadLibrary* does not prove it loads")
    print("   ACE-PBC-Game64.dll; confirming that needs the string/argument, which")
    print("   this scan does not attempt.")


if __name__ == "__main__":
    main(sys.argv[1])
