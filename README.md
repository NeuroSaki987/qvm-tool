# qvmtool — ACE `.qvm0` analysis and structure recovery

`qvmtool` is a dependency-light static-analysis toolkit for PE images protected
with ACE's `.qvm0` engine. It recovers information conventional analysis often
misses:

- the real `RUNTIME_FUNCTION` map, even when the Exception directory points
  outside the section named `.pdata`;
- native-to-QVM entry edges and shared entry clusters;
- virtualized-function stubs and synthetic symbols;
- control-flow edges hidden behind constant and MBA expressions;
- Ghidra and IDA scripts that apply the recovered structure;
- an analysis-oriented repaired image with an auditable patch manifest.

The project is a reverse-engineering aid. It recovers structure and normalizes
known obfuscation patterns; it is not a general-purpose decryptor and does not
claim to reconstruct original source code.

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url>
cd qvm-tool
python -m pip install -e .
```

Optional dynamic experiments use Unicorn:

```bash
python -m pip install -e ".[emu]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

```bash
# Identify the protection engine and summarize the image.
qvmtool detect sample.dll

# Run the automatic QVM probe.
qvmtool probe sample.dll

# Recover the function map, entries, regions, symbols, and reports.
qvmtool survey sample.dll -o out/

# List recoverable QVM entry points.
qvmtool entries sample.dll -o out/entries.tsv

# Produce an analysis-oriented repaired image and companion artifacts.
qvmtool write-devirt sample.dll out/sample_repaired.dll
```

All commands also work through the module entry point:

```bash
python -m qvmtool --help
```

## Commands

| Command | Purpose |
|---|---|
| `detect` | Classify the protection engine and show the evidence behind the verdict |
| `probe` | Summarize QVM structure and report what can be recovered or repaired |
| `functions` | Recover the authoritative `RUNTIME_FUNCTION` map |
| `trampolines` | Find anchored native-to-QVM edges, entry clusters, and whole-function stubs |
| `regions` | Profile the VM section and identify code-like, structured, and high-entropy regions |
| `symbols` | Emit recovered symbols as JSON, TSV, Ghidra Python, and IDA IDC |
| `edges` | Recover and validate CFG edges, then emit Ghidra/IDA injection scripts |
| `dynamic` | Recover edges by emulating from real native call contexts |
| `patch` | Normalize known branch idioms and verify patch containment |
| `entries` | List recoverable QVM entry points |
| `write-devirt` | Probe, repair, validate, and emit the complete analysis bundle |
| `survey` | Run the main static pipeline and write human- and machine-readable reports |

Long-running edge recovery can be reused without repeating the static pass:

```bash
qvmtool edges sample.dll -o out/edges
qvmtool edges sample.dll --from-edges out/edges.tsv -o out/edges-checked
```

## `write-devirt`

`write-devirt` is the shortest path from an input image to a Ghidra/IDA-ready
analysis bundle:

```bash
qvmtool write-devirt sample.dll out/sample_repaired.dll \
  --edges out/edges.tsv \
  --dynamic out/dynamic_edges.tsv
```

The edge files are optional. Supplying previously recovered files avoids repeating
expensive analysis.

Typical outputs are written next to the repaired image:

| Artifact | Contents |
|---|---|
| `*_repaired.dll` | Same-size image with changes restricted to the selected VM section |
| `*.manifest.json` | Every rewritten byte range, including original and replacement bytes |
| `*.symbols.json` / `*.symbols.tsv` | Recovered exports, functions, stubs, and QVM entries |
| `*.apply_ghidra.py` / `*.apply_ida.idc` | Scripts that apply recovered symbols |
| `*.inject_edges.java` / `*.inject_edges.idc` | Scripts that add recovered CFG references |
| `*.probe.txt` | Probe, repair, and verification summary |

The command exits non-zero if containment verification fails, including a size
change or a modification outside the selected VM section.

> The repaired image is an analysis aid. Some normalizations intentionally remove
> flag side effects from obfuscation idioms, so the result must not be treated as a
> functionally equivalent production binary.

## Why QVM analysis needs dedicated recovery

Two details cause ordinary PE analysis to miss much of the image.

### Relocated exception metadata

The authoritative `RUNTIME_FUNCTION[]` array may live in the tail of `.qvm0`
instead of the section named `.pdata`. Windows locates it through the PE Exception
directory, so `qvmtool` follows that directory rather than trusting section names.

### Linear-sweep desynchronization

x86-64 uses variable-length instructions. A linear sweep from the beginning of a
large obfuscated section quickly loses instruction boundaries and silently misses
real calls and branches. `qvmtool` starts decoding from known function entries and
uses byte-level measurements only where decoding is not required.

The tool also rejects a common false positive: UTF-16 strings can resemble arrays
of RVAs into a large VM section. Candidate tables therefore need structural and
instruction-boundary evidence; an address-range match alone is insufficient.

## Recovery model

The main static pipeline is deliberately evidence driven:

1. Parse the PE and locate the effective exception directory.
2. Recover function boundaries and identify code that enters `.qvm0`.
3. Cluster shared QVM entries and recognize whole-function stubs.
4. Resolve known constant branch idioms.
5. Apply bounded backward slicing and MBA simplification to opaque targets.
6. Validate every proposed target against image bounds and instruction boundaries.
7. Emit symbols, CFG references, reports, and optional contained patches.

Unresolved sites remain unresolved. The tool does not invent an edge when the
available evidence cannot establish one.

## Reference-sample results

The current implementation was developed against two builds of
`ACE-PBC-Game64.dll`. The smaller reference image produced:

| Measurement | Result |
|---|---:|
| Functions recovered | 4,669 |
| Native-to-`.qvm0` edges | 408 |
| Native callers / QVM entries | 128 / 263 |
| Whole-function QVM stubs | 92 |
| Static QVM CFG edges recovered | 554 |
| Boundary-verified recovered edges | 527 |
| Symbols emitted | 5,028 |
| Functions successfully decompiled after symbol application | 4,384 / 4,386 |

These numbers describe those samples, not a compatibility guarantee for every ACE
version. The second build preserved the same broad architecture while changing the
scale and counts.

### What the measurements support

- Executed QVM code in the tested images is plaintext.
- Known self-PC branch idioms are mostly CFG-neutral padding; normalizing them
  alone produced negligible decompiler improvement.
- Symbol recovery and value-based edge recovery provide the useful gains.
- A large high-entropy region remains unclassified. Its bytes are consistent with
  more than one explanation, so the tool does not label it as ciphertext or filler.
- No writes to `.qvm0` were observed in the exercised dynamic paths. That is a
  path-bounded observation, not proof about every possible execution.

The CLI and report model retain entropy-based region classification for triage.
High entropy means "not statically characterized"; it does not by itself prove
encryption.

## Ghidra integration

The `ghidra/` directory contains reusable headless-analysis scripts:

| Script | Purpose |
|---|---|
| `ExportQvmReport.java` | Export per-function addresses, sizes, instruction/block counts, references, and VM membership |
| `DecompileExport.java` | Apply a symbol TSV, decompile functions, and record success or failure |

Generated edge scripts add the recovered computed-flow references.

```powershell
# Emit symbols.
qvmtool symbols sample.dll -o out\qvm

# Apply symbols and export decompilation results.
& "<ghidra>\support\analyzeHeadless.bat" <project-dir> qvm-analysis `
  -import sample.dll `
  -scriptPath ghidra `
  -postScript DecompileExport.java out\qvm.symbols.tsv out\decomp 0x180000000 0x180A00000 80
```

Ghidra's `-scriptPath` accepts one directory. Copy generated scripts into the same
script directory when several post-scripts must run together.

## `survey` output

```text
survey.md               human-readable report
survey.json             summary
survey.full.json        complete measurements
functions.tsv           recovered function map
qvm.symbols.json        recovered symbols
qvm.symbols.tsv         tabular symbols
qvm.apply_ghidra.py     Ghidra symbol script
qvm.apply_ida.idc       IDA symbol script
```

## Limitations

- Engine and state-carrier detection use measured heuristics. Inspect the emitted
  evidence and counters rather than relying on the label alone.
- Backward slicing is bounded by window and instruction limits. `opaque` means the
  configured analysis did not resolve the site, not that it is provably dynamic.
- Dynamic results cover only the exercised entry contexts and execution paths.
- Entropy separates structured/readable regions from high-entropy regions; it does
  not distinguish encryption, compression, and random padding.
- Synthetic names such as `qvm_stub_XXXXXXXX` and `qvm_entry_XXXXXXXX` recover
  structure, not original identifiers.
- The optional patcher preserves file size and checks containment, but its output
  remains intended for analysis.
- Validation so far is concentrated on the documented QVM sample family. New
  versions may introduce entry or branch patterns the tool does not recognize.

## Repository layout

```text
qvmtool/
  model.py       shared records and evidence types
  pe.py          PE access and RVA mapping
  detect.py      multi-signal engine detection
  functions.py   exception-directory function recovery
  xrefs.py       anchored references, clusters, and stub detection
  deobf.py       known QVM branch-idiom resolution
  mba.py         bit-vector expressions and MBA simplification
  slice.py       bounded backward slicing
  cfg.py         edge assembly, validation, and TSV I/O
  emulate.py     optional Unicorn-based dynamic recovery
  regions.py     VM-section profiling
  state.py       state-carrier assessment
  symbols.py     JSON/TSV/Ghidra/IDA symbol emitters
  ghidra.py      CFG-reference script emitters
  patch.py       contained analysis patching
  report.py      Markdown reporting
  survey.py      pipeline orchestration
  cli.py         command-line interface
ghidra/          reusable Ghidra headless scripts
scripts/         measurement and verification drivers
tests/           unit and regression tests
```

The drivers under `scripts/` reproduce individual measurements and negative
controls used during development. See [`scripts/README.md`](scripts/README.md).

## Scope

The main pipeline performs static analysis of locally supplied binaries. Dynamic
experiments use emulation and do not load the target as a native module. No sample
binaries or sample-derived output are included in this repository.

## License

MIT. See [`LICENSE`](LICENSE).
