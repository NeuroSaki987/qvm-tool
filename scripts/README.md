# Driver scripts

These are the analysis drivers used to produce the measurements quoted in the
README and in the case notes. They are **argument-driven and reusable**: each
takes the PE path (and output paths) on the command line.

Run from the repository root, e.g.:

```bash
python scripts/extended_census.py <pe> <dynamic_edges.tsv>
```

| script | purpose |
|---|---|
| `verify_target_decodability.py` | decode-density verification of native->VM targets against a control population |
| `verify_dynamic_coverage.py` | page-level set intersection proving whether executed code overlaps branch sites |
| `verify_dynamic_edges.py` | cross-reference and boundary-validate dynamic vs static edge sets |
| `opaque_coverage.py` | how much of the statically-opaque site set the dynamic pass reached |
| `extended_census.py` | reconcile the full indirect-branch census against observed sites |
| `external_decryptor_scan.py` | scan sibling modules for page-protection/mapping/write capability |
| `task1_slice_all.py` | full static slice: idiom resolution then backward slicing, with edge output |
| `task1b_dynamic_codelike.py` | dynamic edge recovery restricted to code-like VM entries |
| `task2_realctx.py` | emulate from real call contexts; recover the stack frame slot layout |
| `task4_patch.py` | length-preserving de-obfuscation patch plus containment verification |

## Why these specific scripts exist

Several of them exist to **falsify a claim**, which is why they are kept:

* `verify_target_decodability.py` — compares a decode-density metric against a
  control population. It is the script that showed a spot-check of three targets
  was not representative.
* `verify_dynamic_coverage.py` — the page-level set intersection that proved a
  tracer bug rather than an architectural fact when the tracer reported zero
  indirect branches across 3M instructions.
* `extended_census.py` — showed the original indirect-branch census enumerated
  only 35% of the real population because it matched register forms only.
* `external_decryptor_scan.py` — capability scan across sibling modules; the
  hypothesis it was built for was subsequently disproved by a dynamic
  no-write observation, and the negative is recorded with it.

Keep them honest: a driver that only ever confirms is not worth keeping.
