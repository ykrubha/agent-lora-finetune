# Fine-Tuning a Small LLM for Agent Tool Calling

This project fine-tunes Qwen2.5-1.5B with LoRA to drive the [Agentic AI Data Analyst](https://github.com/ykrubha/agentic-data-analyst), then serves it with vLLM. The base and fine-tuned models are scored on the agent's 38-question benchmark.

Small models often know the right answer but don't follow the agent's tool protocol. In the agent's testing, for example, a 4B model found the correct answer and then wrote it as plain text instead of calling `final_answer`. The goal here is to measure whether fine-tuning fixes failures like that.

## Status

**Results are pending. The pipeline has only been tested on a tiny random model on CPU, and it hasn't yet trained the real model or been run through vLLM.**

Open [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) in Google Colab with a T4 GPU to run everything.

## How it works

- **Training data:** the agent's real tools run on 8 synthetic databases (hospital, logistics and others), so every tool result is real. Training includes error recovery and refusals.
- **No benchmark leakage:** the training databases don't use the benchmark's schema or data, and a test rejects any training question that exactly matches a benchmark question. The question types are similar in kind, because that's the skill being trained.
- **Held-out schemas:** two database domains are only used for validation.
- **Labels:** only the assistant's turns count toward the loss. A check stops training if the chat template would misplace those labels.
- **Training:** a small PyTorch loop with LoRA (rank 16) on the attention layers, saving the adapter with the lowest validation loss. It can resume after a disconnect.
- **Serving:** vLLM serves the base model and adapter from one server, so both run with the same settings.

## Tests

```bash
python -m venv .venv && source .venv/bin/activate
git clone https://github.com/ykrubha/agentic-data-analyst.git agent
pip install -e agent -e ".[dev,train]" jinja2
pytest -q
```

## License

MIT for this code. Qwen2.5-1.5B-Instruct has its own license on its Hugging Face page, so check it before reusing the model.
