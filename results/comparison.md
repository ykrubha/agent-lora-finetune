| Metric | Qwen 1.5B base | Qwen 1.5B + LoRA | Gemini 3.5 Flash Lite |
|---|---:|---:|---:|
| Model | `base` | `analyst` | `gemini-3.5-flash-lite` |
| **Correct** | **0/38** | **17/38** | **37/38** |
| aggregation | 0/8 | 6/8 | 8/8 |
| join | 0/10 | 3/10 | 10/10 |
| multi_step | 0/8 | 2/8 | 8/8 |
| python | 0/6 | 3/6 | 5/6 |
| refusal | 0/6 | 3/6 | 6/6 |
| Finished with an answer | 28/38 | 36/38 | 37/38 |
| Answers sent as native tool calls | 28/38 | 36/38 | 37/38 |
| Answers recovered from plain text | 0 | 0 | 0 |
| Tool errors | 77 | 14 | 20 |
| Mean steps | 4.05 | 4.79 | 6.5 |
| Mean seconds per question | 5.2 | 8.2 | 92.9 |
| How runs ended | answered: 28, no_final_answer: 5, too_many_errors: 5 | answered: 36, too_many_errors: 2 | answered: 37, max_steps: 1 |
