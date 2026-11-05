"""Generate training conversations by running the real agent tools.

Every tool result in the training data comes from actually running the query or code, so the
model learns from the same outputs it will see in use. No result is written by hand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from analyst_agent.agent import SYSTEM_PROMPT
from analyst_agent.sql_guard import ReadOnlyDatabase
from analyst_agent.tools import TOOL_SCHEMAS, ToolExecutor

from .databases import DOMAINS, VALIDATION_DOMAINS, Domain, Table, build_database

CHAT_TOOLS = [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]


@dataclass
class Task:
    kind: str
    question: str
    tables: list[str]
    sql: str = ""
    python: str = ""
    value_from: str = "scalar"  # scalar, list, or stdout
    round_to: int | None = None
    answer_template: str = "{value}"
    refuse_reason: str = ""


def _words(name: str) -> str:
    """Turn a column name into plain words: WeightKg becomes weight kg."""
    out = ""
    for char in name:
        if char.isupper() and out:
            out += " "
        out += char.lower()
    return out.replace(" id", "").strip()


def _plural(name: str) -> str:
    word = _words(name)
    if word.endswith(("s", "x", "ch")):
        return word + "es"
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


def _columns(table: Table, kind: str) -> list:
    return [c for c in table.columns if c.kind == kind]


def _label_column(table: Table):
    return next((c for c in table.columns if c.kind == "name"), None)


def _table(domain: Domain, name: str) -> Table:
    return next(t for t in domain.tables if t.name == name)


def tasks_for(domain: Domain, rng: random.Random) -> list[Task]:
    """Build every question type this schema supports."""
    tasks: list[Task] = []
    for table in domain.tables:
        tasks.append(Task(
            "aggregation",
            rng.choice([f"How many {_plural(table.name)} are there?",
                        f"What is the total number of {_plural(table.name)}?"]),
            [table.name], f'SELECT COUNT(*) AS n FROM "{table.name}"',
            answer_template=f"There are {{value}} {_plural(table.name)}.",
        ))
        for column in _columns(table, "money"):
            tasks.append(Task(
                "aggregation", f"What is the total {_words(column.name)} across all {_plural(table.name)}?",
                [table.name], f'SELECT SUM("{column.name}") AS total FROM "{table.name}"',
                answer_template=f"The total {_words(column.name)} is {{value}}.",
            ))
            tasks.append(Task(
                "aggregation", f"What is the average {_words(column.name)} per {_words(table.name)}?",
                [table.name], f'SELECT AVG("{column.name}") AS average FROM "{table.name}"',
                answer_template=f"The average {_words(column.name)} is {{value}}.",
            ))
        for column in _columns(table, "category"):
            choice = rng.choice(column.choices)
            tasks.append(Task(
                "aggregation", f"How many {_plural(table.name)} have a {_words(column.name)} of {choice}?",
                [table.name], f"""SELECT COUNT(*) AS n FROM "{table.name}" WHERE "{column.name}" = '{choice}'""",
                answer_template=f"{{value}} {_plural(table.name)} have a {_words(column.name)} of {choice}.",
            ))
            tasks.append(Task(
                "aggregation", f"Which {_words(column.name)} has the most {_plural(table.name)}?",
                [table.name],
                f'SELECT "{column.name}" AS label, COUNT(*) AS metric FROM "{table.name}" '
                f'GROUP BY "{column.name}" ORDER BY metric DESC LIMIT 2',
                value_from="top", answer_template=f"{{value}} has the most {_plural(table.name)}.",
            ))

    for fact in domain.tables:
        for foreign in _columns(fact, "fk"):
            parent = _table(domain, foreign.choices[0])
            label = _label_column(parent)
            if label is None:
                continue
            join = (f'FROM "{fact.name}" f JOIN "{parent.name}" p '
                    f'ON f."{foreign.name}" = p."{parent.id_column}"')
            tasks.append(Task(
                "join", f"Which {_words(parent.name)} has the most {_plural(fact.name)}? Give its {_words(label.name)}.",
                [fact.name, parent.name],
                f'SELECT p."{label.name}" AS label, COUNT(*) AS metric {join} '
                f'GROUP BY p."{parent.id_column}" ORDER BY metric DESC LIMIT 2',
                value_from="top", answer_template=f"{{value}} has the most {_plural(fact.name)}.",
            ))
            for money in _columns(fact, "money"):
                tasks.append(Task(
                    "join", f"Which {_words(parent.name)} has the highest total {_words(money.name)}?",
                    [fact.name, parent.name],
                    f'SELECT p."{label.name}" AS label, SUM(f."{money.name}") AS metric {join} '
                    f'GROUP BY p."{parent.id_column}" ORDER BY metric DESC LIMIT 2',
                    value_from="top", answer_template=f"{{value}} has the highest total {_words(money.name)}.",
                ))
                tasks.append(Task(
                    "join", f"List the top 3 {_plural(parent.name)} by total {_words(money.name)}, highest first.",
                    [fact.name, parent.name],
                    f'SELECT p."{label.name}" AS label, SUM(f."{money.name}") AS metric {join} '
                    f'GROUP BY p."{parent.id_column}" ORDER BY metric DESC, p."{parent.id_column}" LIMIT 3',
                    value_from="list", answer_template="The top 3 are {value}.",
                ))
                for category in _columns(parent, "category"):
                    tasks.append(Task(
                        "multi_step",
                        f"Which {_words(category.name)} of {_words(parent.name)} has the highest total {_words(money.name)}?",
                        [fact.name, parent.name],
                        f'SELECT p."{category.name}" AS label, SUM(f."{money.name}") AS metric {join} '
                        f'GROUP BY p."{category.name}" ORDER BY metric DESC LIMIT 2',
                        value_from="top", answer_template=f"{{value}} has the highest total {_words(money.name)}.",
                    ))
            tasks.append(Task(
                "multi_step", f"How many {_plural(parent.name)} have no {_plural(fact.name)}?",
                [fact.name, parent.name],
                f'SELECT COUNT(*) AS n FROM "{parent.name}" p LEFT JOIN "{fact.name}" f '
                f'ON f."{foreign.name}" = p."{parent.id_column}" WHERE f."{fact.id_column}" IS NULL',
                answer_template=f"{{value}} {_plural(parent.name)} have no {_plural(fact.name)}.",
            ))

        for date in _columns(fact, "date"):
            for money in _columns(fact, "money"):
                year = rng.randint(2019, 2024)
                tasks.append(Task(
                    "aggregation", f"How much {_words(money.name)} was recorded during {year}?", [fact.name],
                    f"""SELECT SUM("{money.name}") AS total FROM "{fact.name}" """
                    f"""WHERE strftime('%Y', "{date.name}") = '{year}'""",
                    answer_template=f"The {_words(money.name)} recorded during {year} was {{value}}.",
                ))
                tasks.append(Task(
                    "multi_step", f"In which month was total {_words(money.name)} highest? Answer as YYYY-MM.",
                    [fact.name],
                    f"""SELECT strftime('%Y-%m', "{date.name}") AS label, SUM("{money.name}") AS metric """
                    f"""FROM "{fact.name}" GROUP BY label ORDER BY metric DESC LIMIT 2""",
                    value_from="top", answer_template=f"Total {_words(money.name)} was highest in {{value}}.",
                ))
                tasks.append(Task(
                    "python", f"Fit a straight line to monthly total {_words(money.name)}. What is the slope per month?",
                    [fact.name],
                    f"""SELECT strftime('%Y-%m', "{date.name}") AS month, SUM("{money.name}") AS total """
                    f"""FROM "{fact.name}" GROUP BY month ORDER BY month""",
                    python="print(np.polyfit(np.arange(len(df)), df['total'].astype(float), 1)[0])",
                    value_from="stdout", answer_template="The slope is {value} per month.",
                ))
                tasks.append(Task(
                    "python", f"Which year had the highest year-over-year growth in total {_words(money.name)}?",
                    [fact.name],
                    f"""SELECT strftime('%Y', "{date.name}") AS year, SUM("{money.name}") AS total """
                    f"""FROM "{fact.name}" GROUP BY year ORDER BY year""",
                    python="print(df.set_index('year')['total'].astype(float).pct_change().idxmax())",
                    value_from="stdout", answer_template="Growth was highest in {value}.",
                ))
        numbers = _columns(fact, "number") + _columns(fact, "money")
        if len(numbers) >= 2:
            first, second = numbers[0], numbers[1]
            tasks.append(Task(
                "python", f"What is the Pearson correlation between {_words(first.name)} and {_words(second.name)}?",
                [fact.name], f'SELECT "{first.name}" AS a, "{second.name}" AS b FROM "{fact.name}"',
                python="print(df['a'].astype(float).corr(df['b'].astype(float)))",
                value_from="stdout", answer_template="The correlation is {value}.",
            ))
        for number in numbers:
            tasks.append(Task(
                "python", f"What is the median {_words(number.name)} per {_words(fact.name)}?",
                [fact.name], f'SELECT "{number.name}" AS x FROM "{fact.name}"',
                python="print(df['x'].astype(float).median())",
                value_from="stdout", answer_template=f"The median {_words(number.name)} is {{value}}.",
            ))

    any_table = domain.tables[-1]
    for missing in domain.missing:
        tasks.append(Task(
            "refusal", rng.choice([f"What are the {missing}?", f"Give me a summary of the {missing}."]),
            [any_table.name],
            refuse_reason=f"The database has no data on {missing}, so this can't be answered.",
        ))
    tasks.append(Task(
        "refusal", f"Delete all {_plural(any_table.name)} and then tell me how many are left.", [any_table.name],
        refuse_reason="I have read-only access, so I can't delete data.",
    ))
    tasks.append(Task(
        "refusal", f"Add a new {_words(any_table.name)} to the database and confirm when finished.", [any_table.name],
        refuse_reason="I have read-only access, so I can't add data.",
    ))
    return tasks


class TaskSkipped(Exception):
    """A question has no clear answer, such as a tie or an empty result, so it isn't used."""


def _assistant(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def _call(messages: list[dict], tools: ToolExecutor, name: str, arguments: dict):
    messages.append(_assistant(name, arguments))
    result = tools.execute(name, arguments)
    messages.append({"role": "tool", "name": name, "content": result.content})
    return result


def _clean_number(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _broken_sql(sql: str, rng: random.Random) -> str | None:
    """A realistic mistake: a misspelled keyword or a wrong column name."""
    if rng.random() < 0.5:
        return sql.replace("SELECT", "SELEC", 1)
    if 'FROM "' not in sql:
        return None
    return sql.replace('AS ', 'AS "bad name" ', 1) if rng.random() < 0.3 else sql.replace("SELECT ", "SELECT missing_column, ", 1)


def run_task(task: Task, database: ReadOnlyDatabase, rng: random.Random, error_rate: float = 0.12) -> dict:
    """Run one task through the real tools and return a training conversation."""
    tools = ToolExecutor(database)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.question},
    ]
    _call(messages, tools, "list_tables", {})
    for table in task.tables:
        _call(messages, tools, "describe_table", {"table": table})

    assumptions: list[str] = []
    if task.refuse_reason:
        messages.append(_assistant("final_answer", {
            "answer": task.refuse_reason, "value": None, "answerable": False, "assumptions": [],
        }))
        return {"kind": task.kind, "question": task.question, "messages": messages}

    if rng.random() < error_rate:
        broken = _broken_sql(task.sql, rng)
        if broken:
            attempt = _call(messages, tools, "run_sql", {"query": broken, "save_as": "df"})
            if not attempt.is_error:
                raise TaskSkipped("The broken query unexpectedly worked.")

    result = _call(messages, tools, "run_sql", {"query": task.sql, "save_as": "df"})
    if result.is_error:
        raise RuntimeError(f"Valid SQL failed: {task.sql}: {result.content}")
    frame = tools.frames["df"]
    if frame.empty:
        raise TaskSkipped("Empty result.")

    if task.value_from == "scalar":
        value = _clean_number(frame.iloc[0, 0])
        if value is None:
            raise TaskSkipped("Null result.")
    elif task.value_from == "top":
        if len(frame) > 1 and frame["metric"].iloc[0] == frame["metric"].iloc[1]:
            raise TaskSkipped("Tied result.")
        value = str(frame["label"].iloc[0])
    elif task.value_from == "list":
        if len(frame) < 3:
            raise TaskSkipped("Too few rows.")
        value = [str(v) for v in frame["label"]]
    elif task.value_from == "stdout":
        outcome = _call(messages, tools, "run_python", {"code": task.python})
        if outcome.is_error:
            raise RuntimeError(f"Valid Python failed: {task.python}: {outcome.content}")
        printed = json.loads(outcome.content)["stdout"].strip()
        if not printed or printed == "nan":
            raise TaskSkipped("No usable output.")
        try:
            value = _clean_number(float(printed))
        except ValueError:
            value = printed
    else:
        raise ValueError(task.value_from)

    shown = ", ".join(value) if isinstance(value, list) else value
    messages.append(_assistant("final_answer", {
        "answer": task.answer_template.format(value=shown), "value": value,
        "answerable": True, "assumptions": assumptions,
    }))
    return {"kind": task.kind, "question": task.question, "messages": messages}


def _examples_for(domain: Domain, work_dir, seed: int, max_refusals: int, stats: dict) -> list[dict]:
    rng = random.Random(f"{domain.name}:{seed}")
    database = ReadOnlyDatabase(build_database(domain, work_dir, seed))
    tasks = tasks_for(domain, rng)
    rng.shuffle(tasks)
    examples, seen, refusals = [], set(), 0
    for task in tasks:
        if task.question in seen:
            continue
        seen.add(task.question)
        if task.kind == "refusal":
            refusals += 1
            if refusals > max_refusals:
                continue
        try:
            example = run_task(task, database, rng)
        except TaskSkipped:
            stats["skipped"] += 1
            continue
        example.update(domain=domain.name, seed=seed, tools=CHAT_TOOLS)
        examples.append(example)
    return examples


def build_examples(work_dir: str | Path, train_seeds: tuple[int, ...] = (7, 8, 9),
                   max_refusals: int = 3) -> tuple[list[dict], list[dict], dict]:
    """Return training and validation examples. Validation uses domains training never sees.

    Each training seed rebuilds the databases with new contents and picks new question details,
    so the same schema appears with different data and different answers.
    """
    train, validation = [], []
    stats: dict = {"skipped": 0}
    for domain in DOMAINS:
        if domain.name in VALIDATION_DOMAINS:
            validation += _examples_for(domain, work_dir, train_seeds[0], max_refusals, stats)
        else:
            for seed in train_seeds:
                train += _examples_for(domain, work_dir, seed, max_refusals, stats)
    random.Random(train_seeds[0]).shuffle(train)
    return train, validation, stats


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
