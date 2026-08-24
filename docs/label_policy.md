# Label policy: confirmed_vs_false_positive_v1

The source field is `koi_disposition` from the frozen DR25 KOI snapshot:

| Source disposition | Derived label | Default cohort |
|---|---:|---|
| `CONFIRMED` | 1 | Included unless conflicting |
| `FALSE POSITIVE` | 0 | Included unless conflicting |
| `CANDIDATE` | null | Excluded |
| `NOT DISPOSITIONED` or null | null | Excluded |

`koi_pdisposition` is retained separately as the Kepler-data/Robovetter
disposition. It is never substituted for `koi_disposition` by this policy.

Conflicts include a confirmed archive disposition paired with a Robovetter false
positive, a confirmed row lacking a Kepler name, or a non-confirmed row carrying
a Kepler confirmation name. Conflicts receive no binary label in the default
cohort and are copied to the conflict report. Source fields are never rewritten.

This policy creates a confirmation-selected cohort. It is not an unbiased census
of planets, a complete TCE sample, or a uniform PC-versus-FP Robovetter benchmark.
