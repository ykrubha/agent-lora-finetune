Model: `base`, run 2026-11-05T16:29:40+00:00.

| Category | Correct | Accuracy | Mean steps | Mean seconds | Mean tokens |
|---|---:|---:|---:|---:|---:|
| aggregation | 0/8 | 0% | 5.12 | 5.43 | 6718 |
| join | 0/10 | 0% | 5.3 | 6.57 | 6777 |
| multi_step | 0/8 | 0% | 2.88 | 5.49 | 3858 |
| python | 0/6 | 0% | 3.17 | 3.84 | 3385 |
| refusal | 0/6 | 0% | 3.0 | 3.89 | 3132 |
| **Overall** | 0/38 | 0% | 4.05 | 5.25 | 5039 |

Tool errors: 77 across 37 questions. 0 of those questions still ended correct.

How runs ended: {'no_final_answer': 5, 'answered': 28, 'too_many_errors': 5}. 0 answers were written as JSON text and recovered, not sent as native tool calls.
