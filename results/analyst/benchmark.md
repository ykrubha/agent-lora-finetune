Model: `analyst`, run 2026-11-05T16:34:52+00:00.

| Category | Correct | Accuracy | Mean steps | Mean seconds | Mean tokens |
|---|---:|---:|---:|---:|---:|
| aggregation | 6/8 | 75% | 4.12 | 6.12 | 5973 |
| join | 3/10 | 30% | 4.6 | 7.32 | 6803 |
| multi_step | 2/8 | 25% | 6.12 | 12.56 | 10831 |
| python | 3/6 | 50% | 5.17 | 8.74 | 9026 |
| refusal | 3/6 | 50% | 3.83 | 5.89 | 5736 |
| **Overall** | 17/38 | 45% | 4.79 | 8.17 | 7659 |

Tool errors: 14 across 5 questions. 0 of those questions still ended correct.

How runs ended: {'answered': 36, 'too_many_errors': 2}. 0 answers were written as JSON text and recovered, not sent as native tool calls.
