"""Tests for the data and training code, using a tiny random model that needs no download."""

import json
import re
import sqlite3

import pytest

from eval.questions import QUESTIONS
from finetune.databases import DOMAINS, VALIDATION_DOMAINS, build_database
from finetune.traces import build_examples, write_jsonl
from scripts.compare import summarize, to_markdown

CHINOOK_TABLES = {"Album", "Artist", "Customer", "Employee", "Genre", "Invoice", "InvoiceLine",
                  "MediaType", "Playlist", "PlaylistTrack", "Track"}

# Follows the structure of Qwen2.5's chat template: tools in the system turn, tool calls as
# <tool_call> JSON, and tool results grouped in one user turn.
TEMPLATE = """
{%- if tools %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '\n\n# Tools\n<tools>' }}
    {%- for tool in tools %}{{- '\n' + (tool | tojson) }}{%- endfor %}
    {{- '\n</tools><|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant' }}
        {%- if message.content %}{{- '\n' + message.content }}{%- endif %}
        {%- for call in message.tool_calls %}
            {{- '\n<tool_call>\n{"name": "' + call.function.name + '", "arguments": ' }}
            {{- call.function.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == 'tool' %}
        {%- if loop.index0 == 0 or messages[loop.index0 - 1].role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}
        {{- '\n<tool_response>\n' + message.content + '\n</tool_response>' }}
        {%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}{{- '<|im_end|>\n' }}{%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}
"""


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    return build_examples(tmp_path_factory.mktemp("db"), train_seeds=(7,))


def test_databases_build_with_valid_foreign_keys(tmp_path):
    for domain in DOMAINS:
        conn = sqlite3.connect(build_database(domain, tmp_path))
        conn.execute("PRAGMA foreign_keys = ON")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        for table in domain.tables:
            assert conn.execute(f'SELECT COUNT(*) FROM "{table.name}"').fetchone()[0] == table.rows


def test_no_benchmark_tables_or_questions_in_training_data(examples):
    train, validation, _ = examples
    for domain in DOMAINS:
        assert not {t.name for t in domain.tables} & CHINOOK_TABLES
    benchmark = {q.text.casefold() for q in QUESTIONS}
    for example in train + validation:
        assert example["question"].casefold() not in benchmark
        text = json.dumps(example["messages"][2:])
        for term in ("Iron Maiden", "Chinook", "InvoiceLine", "PlaylistTrack"):
            assert term not in text


def test_validation_domains_are_never_trained_on(examples):
    train, validation, _ = examples
    assert not {e["domain"] for e in train} & set(VALIDATION_DOMAINS)
    assert {e["domain"] for e in validation} == set(VALIDATION_DOMAINS)


def test_every_conversation_is_well_formed(examples):
    train, validation, _ = examples
    for example in train + validation:
        messages = example["messages"]
        assert [m["role"] for m in messages[:2]] == ["system", "user"]
        final = messages[-1]
        assert final["role"] == "assistant"
        assert final["tool_calls"][0]["function"]["name"] == "final_answer"
        for index, message in enumerate(messages):
            if message["role"] == "tool":
                assert messages[index - 1]["role"] == "assistant"


def test_refusals_answer_nothing_and_answers_answer_something(examples):
    train, _, _ = examples
    for example in train:
        arguments = example["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        if example["kind"] == "refusal":
            assert arguments["answerable"] is False and arguments["value"] is None
            assert "run_sql" not in json.dumps(example["messages"][2:])
        else:
            assert arguments["answerable"] is True and arguments["value"] is not None


def test_final_values_come_from_real_tool_results(examples):
    train, _, _ = examples
    for example in [e for e in train if e["kind"] == "join"][:40]:
        arguments = example["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        sql_result = next(m for m in reversed(example["messages"])
                          if m["role"] == "tool" and m["name"] == "run_sql")["content"]
        value = arguments["value"]
        for item in value if isinstance(value, list) else [value]:
            assert str(item) in sql_result


def test_error_recovery_examples_contain_a_real_error_then_a_fix(examples):
    train, _, _ = examples
    recovered = [e for e in train if any(m["role"] == "tool" and "Sql" in m["content"][:12] for m in e["messages"])]
    assert recovered
    for example in recovered:
        calls = [m for m in example["messages"] if m["role"] == "assistant"]
        names = [c["tool_calls"][0]["function"]["name"] for c in calls]
        assert names.count("run_sql") == 2


def test_all_question_kinds_are_present(examples):
    train, validation, _ = examples
    for split in (train, validation):
        assert {e["kind"] for e in split} == {"aggregation", "join", "multi_step", "python", "refusal"}


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, examples):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    train, validation, _ = examples
    directory = tmp_path_factory.mktemp("tiny")
    corpus = [json.dumps(e["messages"]) for e in train[:20] + validation[:5]]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    from tokenizers import decoders
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    wrapped = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>",
        chat_template=TEMPLATE)
    wrapped.save_pretrained(directory)
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(wrapped), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
        use_sliding_window=False, pad_token_id=wrapped.pad_token_id, eos_token_id=wrapped.eos_token_id)
    transformers.Qwen2ForCausalLM(config).save_pretrained(directory)
    return directory


def test_labels_cover_exactly_the_assistant_turns(tiny_model_dir, examples):
    from transformers import AutoTokenizer

    from finetune.train import IGNORE, encode_example

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    example = next(e for e in examples[0] if e["kind"] == "python")
    encoded = encode_example(example, tokenizer, 20000)
    assert len(encoded["input_ids"]) == len(encoded["labels"])

    targets = [i for i, label in zip(encoded["input_ids"], encoded["labels"]) if label != IGNORE]
    text = tokenizer.decode(targets)
    context = tokenizer.decode([i for i, label in zip(encoded["input_ids"], encoded["labels"]) if label == IGNORE])
    assert text.count("<tool_call>") == sum(m["role"] == "assistant" for m in example["messages"])
    assert "final_answer" in text and "run_sql" in text
    assert "<tool_response>" not in text and "careful data analyst" not in text
    assert "<tool_response>" in context and "careful data analyst" in context
    assert example["question"] not in text


def test_too_long_examples_are_dropped(tiny_model_dir, examples):
    from transformers import AutoTokenizer

    from finetune.train import encode_example

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    assert encode_example(examples[0][0], tokenizer, 5) is None


def test_inconsistent_chat_template_is_detected(tiny_model_dir, examples):
    from transformers import AutoTokenizer

    from finetune.train import encode_example

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    tokenizer.chat_template = "{{ messages | length }}"
    with pytest.raises(ValueError, match="prefix-consistent"):
        encode_example(examples[0][0], tokenizer, 20000)


def test_training_learns_saves_and_resumes(tiny_model_dir, examples, tmp_path):
    from finetune.train import train

    train_rows, validation_rows, _ = examples
    write_jsonl(train_rows[:6], tmp_path / "t.jsonl")
    write_jsonl(validation_rows[:2], tmp_path / "v.jsonl")
    out = tmp_path / "adapter"

    first = train(str(tiny_model_dir), tmp_path / "t.jsonl", tmp_path / "v.jsonl", out, epochs=1,
                  batch_size=2, accumulation=1, max_length=20000, lora_rank=4, peak_lr=5e-3)
    assert (out / "best" / "adapter_config.json").exists()
    assert 0 < first["trainable_parameters"] < first["total_parameters"]

    resumed = train(str(tiny_model_dir), tmp_path / "t.jsonl", tmp_path / "v.jsonl", out, epochs=3,
                    batch_size=2, accumulation=1, max_length=20000, lora_rank=4, peak_lr=5e-3)
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]
    assert resumed["history"][-1]["train_loss"] < resumed["history"][0]["train_loss"]


def test_comparison_table():
    row = {"correct": True, "status": "answered", "tool_errors": 1, "steps": 4, "seconds": 2.0}
    summary = {"model": "m", "rows": [row, {**row, "correct": False, "answer_recovered_from_text": True}],
               "by_category": {"join": {"correct": 1, "n": 2}}, "statuses": {"answered": 2}}
    table = to_markdown({"A": summarize(summary)})
    assert "**1/2**" in table
    assert "| Answers recovered from plain text | 1 |" in table
    assert "| join | 1/2 |" in table
