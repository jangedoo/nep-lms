#!/usr/bin/env python
"""Translate sentence-transformers/NanoBEIR-en to Nepali.

This script keeps the original NanoBEIR configs and split names intact. Only
the textual columns in the `queries` and `corpus` configs are translated; qrels
and bm25 relation IDs are copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from queue import Empty, Queue
from collections import Counter
from threading import Event, Semaphore, Thread
from datasets import Dataset, DatasetDict, disable_progress_bar, get_dataset_config_names
from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm


SOURCE_DATASET = "sentence-transformers/NanoBEIR-en"
TEXT_CONFIGS = {"queries", "corpus"}
DEFAULT_ENDPOINT = "http://localhost:8888/v1/chat/completions"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SYSTEM_PROMPT = """You are a professional English-to-Nepali dataset translator.

Translate retrieval benchmark text into natural, fluent Nepali written in Devanagari.
Preserve the original meaning, factual claims, named entities, numbers, units,
URLs, citations, formulas, code-like strings, and punctuation as much as Nepali
style allows. Do not add explanations, corrections, warnings, or answers. Do not
summarize or expand. Keep line breaks and list structure when present.

Return only valid JSON matching the requested schema."""


USER_PROMPT_TEMPLATE = """Translate each item from English to Nepali.

Rules:
- Return one translation object for each input object, in the same order.
- Each output object must have the same "id" and a "translation" string.
- Translate only the "text" value.
- If a term is a proper noun, acronym, identifier, URL, citation marker, or
  cannot be translated naturally, keep it unchanged.
- Do not include markdown fences, notes, or any text outside JSON.

Input JSON:
{items_json}
"""


@dataclass(frozen=True)
class ClientConfig:
    name: str
    endpoint: str
    api_key: str
    model: str | None
    temperature: float
    timeout: float
    max_retries: int
    retry_sleep: float
    use_response_format: bool
    chat_template_enable_thinking: bool | None
    reasoning_enabled: bool | None
    max_parallel_requests: int


class TranslationCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            create table if not exists translations (
                cache_key text primary key,
                source_text text not null,
                translated_text text not null,
                created_at real not null
            )
            """
        )
        self.conn.commit()

    def get(self, source_dataset: str, source_text: str) -> str | None:
        cache_key = self._key(source_dataset, source_text)
        row = self.conn.execute(
            "select translated_text from translations where cache_key = ?",
            (cache_key,),
        ).fetchone()
        return row[0] if row else None

    def put(self, source_dataset: str, source_text: str, translated_text: str) -> None:
        cache_key = self._key(source_dataset, source_text)
        self.conn.execute(
            """
            insert or replace into translations
            (cache_key, source_text, translated_text, created_at)
            values (?, ?, ?, ?)
            """,
            (cache_key, source_text, translated_text, time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _key(source_dataset: str, source_text: str) -> str:
        payload = f"{source_dataset}\0{source_text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate NanoBEIR-en to Nepali and optionally upload it to Hugging Face."
    )
    parser.add_argument("--source-dataset", default=SOURCE_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", help=f"Local/OpenAI-compatible chat endpoint. Defaults to OPENAI_BASE_URL or {DEFAULT_ENDPOINT}.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file to load before reading API keys.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", help="Local/OpenAI-compatible model. Defaults to OPENAI_MODEL when set.")
    parser.add_argument(
        "--disable-openrouter",
        action="store_true",
        help="Disable the OpenRouter client even when OPENROUTER_API_KEY is available.",
    )
    parser.add_argument(
        "--openrouter-endpoint",
        help=f"OpenRouter chat endpoint. Defaults to OPENROUTER_BASE_URL or {DEFAULT_OPENROUTER_ENDPOINT}.",
    )
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--openrouter-model",
        help=f"OpenRouter model. Defaults to OPENROUTER_MODEL or {DEFAULT_OPENROUTER_MODEL}.",
    )
    parser.add_argument(
        "--openrouter-timeout",
        type=float,
        help="OpenRouter request timeout in seconds. Defaults to OPENROUTER_TIMEOUT or 60.",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        help="Maximum LLM batches to process concurrently. Defaults to the number of enabled clients.",
    )
    parser.add_argument(
        "--local-parallel-requests",
        type=int,
        default=1,
        help="Maximum concurrent requests sent to the local LLM endpoint. Defaults to 1.",
    )
    parser.add_argument(
        "--openrouter-parallel-requests",
        type=int,
        help=(
            "Maximum concurrent requests sent to OpenRouter. Defaults to the remaining "
            "parallel request budget after other enabled clients are assigned."
        ),
    )
    parser.add_argument(
        "--log-llm-requests",
        action="store_true",
        help="Print which client each uncached batch is dispatched to.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-chars", type=int, default=6000)
    parser.add_argument("--configs", nargs="*", help="Optional subset/config names to process.")
    parser.add_argument("--splits", nargs="*", help="Optional split names to process.")
    parser.add_argument("--limit", type=int, help="Optional row limit per config/split for testing.")
    parser.add_argument("--show-progress", action="store_true", help="Show Hugging Face dataset progress bars.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs already saved under output-dir/dataset/<config>.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", help="Target Hugging Face dataset repo, e.g. username/NanoBEIR-ne.")
    parser.add_argument("--private", action="store_true", help="Create/upload the target dataset as private.")
    parser.add_argument(
        "--commit-message",
        default="Add Nepali NanoBEIR translation",
        help="Commit message used when pushing configs to Hugging Face.",
    )
    return parser.parse_args()


def require_api_key(env_name: str, client_name: str) -> str:
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is not set; it is required for the {client_name} LLM endpoint.")
    return api_key


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(f"python-dotenv is unavailable; skipping env file: {path}", file=sys.stderr)
        return
    load_dotenv(path)


def build_clients(args: argparse.Namespace) -> list[ClientConfig]:
    clients = [
        ClientConfig(
            name="local",
            endpoint=args.endpoint or os.environ.get("OPENAI_BASE_URL", DEFAULT_ENDPOINT),
            api_key=require_api_key(args.api_key_env, "local"),
            model=args.model or os.environ.get("OPENAI_MODEL"),
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            use_response_format=True,
            chat_template_enable_thinking=False,
            reasoning_enabled=None,
            max_parallel_requests=args.local_parallel_requests,
        )
    ]

    if args.disable_openrouter:
        return clients

    openrouter_api_key = os.environ.get(args.openrouter_api_key_env)
    if not openrouter_api_key:
        print(
            f"{args.openrouter_api_key_env} is not set; using local client only.",
            file=sys.stderr,
            flush=True,
        )
        return clients

    clients.insert(
        0,
        ClientConfig(
            name="openrouter",
            endpoint=args.openrouter_endpoint
            or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_ENDPOINT),
            api_key=openrouter_api_key,
            model=args.openrouter_model
            or os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            temperature=args.temperature,
            timeout=args.openrouter_timeout
            or float(os.environ.get("OPENROUTER_TIMEOUT", 60)),
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            use_response_format=True,
            chat_template_enable_thinking=None,
            reasoning_enabled=False,
            max_parallel_requests=args.openrouter_parallel_requests or 1_000_000,
        ),
    )
    return clients


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    client_index: int


def build_worker_specs(clients: list[ClientConfig], parallel_requests: int) -> list[WorkerSpec]:
    if parallel_requests < 1:
        raise ValueError("--parallel-requests must be at least 1.")

    remaining_capacity = {
        index: max(0, client.max_parallel_requests)
        for index, client in enumerate(clients)
    }
    specs: list[WorkerSpec] = []

    def add_worker(client_index: int) -> bool:
        if len(specs) >= parallel_requests:
            return False
        if remaining_capacity.get(client_index, 0) <= 0:
            return False
        specs.append(WorkerSpec(worker_id=len(specs), client_index=client_index))
        remaining_capacity[client_index] -= 1
        return True

    # First give every enabled client one worker, in client order. Since OpenRouter
    # is inserted before local in build_clients(), this means OpenRouter gets the
    # first worker and local gets at most one worker by default.
    for client_index in range(len(clients)):
        add_worker(client_index)

    # Then spend the remaining budget on clients that still have capacity. With
    # defaults and --parallel-requests 16, this becomes 15 OpenRouter + 1 local.
    while len(specs) < parallel_requests:
        added = False
        for client_index in range(len(clients)):
            added = add_worker(client_index) or added
            if len(specs) >= parallel_requests:
                break
        if not added:
            break

    if not specs:
        raise ValueError("No workers could be created. Check per-client parallel request limits.")
    return specs


def discover_layout(source_dataset: str, configs: Iterable[str] | None, splits: Iterable[str] | None) -> dict[str, list[str]]:
    requested_configs = set(configs or [])
    requested_splits = set(splits or [])
    available_configs = get_dataset_config_names(source_dataset)
    missing_configs = requested_configs.difference(available_configs)
    if missing_configs:
        raise ValueError(f"Unknown configs for {source_dataset}: {sorted(missing_configs)}")

    selected_configs = [cfg for cfg in available_configs if not requested_configs or cfg in requested_configs]
    layout: dict[str, list[str]] = {}
    for config_name in selected_configs:
        available_splits = get_dataset_split_names(source_dataset, config_name)
        missing_splits = requested_splits.difference(available_splits)
        if missing_splits:
            raise ValueError(f"Unknown splits for {config_name}: {sorted(missing_splits)}")
        layout[config_name] = [
            split for split in available_splits if not requested_splits or split in requested_splits
        ]
    return layout


def call_chat_completion(client: ClientConfig, items: list[dict[str, str]]) -> list[dict[str, str]]:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    items_json=json.dumps(items, ensure_ascii=False)
                ),
            },
        ],
        "temperature": client.temperature,
    }
    if client.use_response_format:
        body["response_format"] = translation_response_format(len(items))
    if client.chat_template_enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": client.chat_template_enable_thinking}
    if client.reasoning_enabled is not None:
        body["reasoning"] = {"enabled": client.reasoning_enabled}
    if client.model:
        body["model"] = client.model

    for attempt in range(client.max_retries + 1):
        try:
            response = requests.post(
                client.endpoint,
                headers={
                    "Authorization": f"Bearer {client.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=client.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            expected_ids = [item["id"] for item in items]
            return parse_translation_json(content, expected_ids=expected_ids)
        except Exception as exc:
            if attempt >= client.max_retries:
                raise RuntimeError(f"{client.name} LLM request failed after retries: {exc}") from exc
            sleep_for = client.retry_sleep * (2**attempt)
            print(
                f"{client.name} LLM request failed ({exc}); retrying in {sleep_for:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)

    raise AssertionError("unreachable")


def translation_response_format(batch_len: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "nanobeir_ne_translation_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "translations": {
                        "type": "array",
                        "minItems": batch_len,
                        "maxItems": batch_len,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "translation": {"type": "string"},
                            },
                            "required": ["id", "translation"],
                        },
                    }
                },
                "required": ["translations"],
            },
        },
    }


def parse_translation_json(content: str, expected_ids: list[str]) -> list[dict[str, str]]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    parsed = json.loads(stripped)
    translations = parsed.get("translations") if isinstance(parsed, dict) else None
    if not isinstance(translations, list):
        raise ValueError("LLM response must be an object with a translations array.")
    if len(translations) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} translations, got {len(translations)}.")

    normalized: list[dict[str, str]] = []
    for expected_id, item in zip(expected_ids, translations, strict=True):
        if not isinstance(item, dict):
            raise ValueError("Each translation item must be an object.")
        item_id = str(item.get("id", ""))
        translation = item.get("translation")
        if item_id != expected_id:
            raise ValueError(f"Expected id {expected_id!r}, got {item_id!r}.")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"Missing translation for id {expected_id!r}.")
        normalized.append({"id": expected_id, "translation": translation.strip()})
    return normalized


def translate_texts(
    clients: list[ClientConfig],
    cache: TranslationCache,
    source_dataset: str,
    texts: list[str],
    batch_size: int,
    max_batch_chars: int,
    parallel_requests: int,
    log_llm_requests: bool,
) -> list[str]:
    """Translate texts using long-lived workers assigned to concrete clients.

    The old implementation submitted one future per batch and only used the
    round-robin client index inside each future. This made it hard to reason
    about whether every configured endpoint was continuously working. Here the
    scheduling is explicit: each worker owns a primary client, pulls the next
    available batch, and immediately pulls another batch when it finishes.
    """
    if not clients:
        raise ValueError("At least one LLM client is required.")

    output: list[str | None] = [None] * len(texts)
    pending: list[tuple[int, str]] = []

    for idx, text in enumerate(texts):
        # there are some texts with empty text in original dataset
        # for consistency we also keep those here.
        if not text.strip():
            output[idx] = ""
            continue
        cached = cache.get(source_dataset, text)
        if cached is None:
            pending.append((idx, text))
        else:
            output[idx] = cached

    batches = list(iter_batches(pending, batch_size=batch_size, max_batch_chars=max_batch_chars))
    cached_count = len(texts) - len(pending)

    with tqdm(
        total=len(texts),
        initial=cached_count,
        unit="row",
        desc=source_dataset,
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(cached=cached_count, pending=len(pending), refresh=False)
        if batches:
            worker_specs = build_worker_specs(clients, min(parallel_requests, len(batches)))
            worker_client_counts = Counter(spec.client_index for spec in worker_specs)
            client_semaphores = {
                client_index: Semaphore(count)
                for client_index, count in worker_client_counts.items()
            }
            batch_queue: Queue[tuple[int, list[tuple[int, str]]]] = Queue()
            result_queue: Queue[
                tuple[int, list[tuple[int, str]], list[str] | None, BaseException | None]
            ] = Queue()
            stop_event = Event()

            for batch_index, batch in enumerate(batches):
                batch_queue.put((batch_index, batch))

            if log_llm_requests:
                assigned_clients = [clients[spec.client_index].name for spec in worker_specs]
                print(
                    f"{source_dataset}: translating {len(batches)} uncached batches "
                    f"with {len(worker_specs)} workers. Worker clients: {assigned_clients}",
                    file=sys.stderr,
                    flush=True,
                )

            workers: list[Thread] = []
            for spec in worker_specs:
                worker = Thread(
                    target=translation_worker,
                    args=(
                        spec.worker_id,
                        clients,
                        spec.client_index,
                        client_semaphores,
                        batch_queue,
                        result_queue,
                        stop_event,
                        source_dataset,
                        log_llm_requests,
                    ),
                    daemon=True,
                )
                worker.start()
                workers.append(worker)

            completed_batches = 0
            try:
                while completed_batches < len(batches):
                    batch_index, batch, translations, error = result_queue.get()
                    completed_batches += 1

                    if error is not None:
                        stop_event.set()
                        raise RuntimeError(
                            f"Translation batch {batch_index + 1}/{len(batches)} failed: {error}"
                        ) from error

                    if translations is None:
                        stop_event.set()
                        raise RuntimeError(
                            f"Translation batch {batch_index + 1}/{len(batches)} returned no result."
                        )

                    for (idx, source_text), translated_text in zip(batch, translations, strict=True):
                        cache.put(source_dataset, source_text, translated_text)
                        output[idx] = translated_text
                    progress.update(len(batch))
                    progress.set_postfix(
                        cached=cached_count,
                        pending=max(len(texts) - progress.n, 0),
                    )
            finally:
                stop_event.set()
                for worker in workers:
                    worker.join(timeout=0.2)

    missing = [idx for idx, value in enumerate(output) if value is None]
    if missing:
        raise RuntimeError(f"Missing translations at row indexes: {missing[:10]}")
    return [value for value in output if value is not None]


def translation_worker(
    worker_id: int,
    clients: list[ClientConfig],
    start_client_index: int,
    client_semaphores: dict[int, Semaphore],
    batch_queue: Queue[tuple[int, list[tuple[int, str]]]],
    result_queue: Queue[tuple[int, list[tuple[int, str]], list[str] | None, BaseException | None]],
    stop_event: Event,
    source_dataset: str,
    log_llm_requests: bool,
) -> None:
    primary_client = clients[start_client_index]
    while not stop_event.is_set():
        try:
            batch_index, batch = batch_queue.get_nowait()
        except Empty:
            return

        try:
            if log_llm_requests:
                row_indexes = [idx for idx, _ in batch]
                print(
                    f"{source_dataset}: worker={worker_id} primary_client={primary_client.name} "
                    f"batch={batch_index + 1} rows={row_indexes[0]}-{row_indexes[-1]}",
                    file=sys.stderr,
                    flush=True,
                )
            translations = translate_batch_with_client_fallback(
                clients=clients,
                start_index=start_client_index,
                client_semaphores=client_semaphores,
                batch=batch,
            )
            result_queue.put((batch_index, batch, translations, None))
        except BaseException as exc:
            result_queue.put((batch_index, batch, None, exc))
            stop_event.set()
        finally:
            batch_queue.task_done()


def iter_batches(
    pending: list[tuple[int, str]],
    batch_size: int,
    max_batch_chars: int,
) -> Iterable[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    char_count = 0
    for item in pending:
        item_chars = len(item[1])
        would_exceed_size = len(batch) >= batch_size
        would_exceed_chars = batch and char_count + item_chars > max_batch_chars
        if would_exceed_size or would_exceed_chars:
            yield batch
            batch = []
            char_count = 0
        batch.append(item)
        char_count += item_chars
    if batch:
        yield batch


def translate_batch_with_fallback(
    client: ClientConfig,
    client_semaphore: Semaphore,
    batch: list[tuple[int, str]],
) -> list[str]:
    items = [{"id": str(idx), "text": text} for idx, text in batch]
    try:
        # This guard is important for fallback paths too: even if 15 OpenRouter
        # workers all fall back to local, only one local HTTP request can run at a
        # time when --local-parallel-requests=1.
        with client_semaphore:
            result = call_chat_completion(client, items)
        return [item["translation"] for item in result]
    except Exception:
        if len(batch) == 1:
            raise
        midpoint = len(batch) // 2
        return translate_batch_with_fallback(
            client, client_semaphore, batch[:midpoint]
        ) + translate_batch_with_fallback(
            client, client_semaphore, batch[midpoint:]
        )


def translate_batch_with_client_fallback(
    clients: list[ClientConfig],
    start_index: int,
    client_semaphores: dict[int, Semaphore],
    batch: list[tuple[int, str]],
) -> list[str]:
    ordered_client_indexes = list(range(start_index, len(clients))) + list(range(0, start_index))
    ordered_client_indexes = [
        client_index
        for client_index in ordered_client_indexes
        if client_index in client_semaphores
    ]
    last_error: Exception | None = None
    for client_index in ordered_client_indexes:
        client = clients[client_index]
        try:
            return translate_batch_with_fallback(
                client,
                client_semaphores[client_index],
                batch,
            )
        except Exception as exc:
            last_error = exc
            if len(ordered_client_indexes) > 1:
                print(
                    f"{client.name} failed for a batch; trying the next configured client.",
                    file=sys.stderr,
                    flush=True,
                )
    raise RuntimeError(f"All LLM clients failed for batch: {last_error}") from last_error


def load_split(source_dataset: str, config_name: str, split_name: str, limit: int | None) -> Dataset:
    dataset = load_dataset(source_dataset, config_name, split=split_name)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def translate_dataset(
    dataset: Dataset,
    config_name: str,
    split_name: str,
    source_dataset: str,
    clients: list[ClientConfig],
    cache: TranslationCache,
    batch_size: int,
    max_batch_chars: int,
    parallel_requests: int,
    log_llm_requests: bool,
) -> Dataset:
    if config_name not in TEXT_CONFIGS:
        return dataset

    rows = list(dataset)
    translated = translate_texts(
        clients=clients,
        cache=cache,
        source_dataset=f"{source_dataset}:{config_name}:{split_name}",
        texts=[str(row["text"]) for row in rows],
        batch_size=batch_size,
        max_batch_chars=max_batch_chars,
        parallel_requests=parallel_requests,
        log_llm_requests=log_llm_requests,
    )
    for row, translated_text in zip(rows, translated, strict=True):
        row["text"] = translated_text
    return Dataset.from_list(rows, features=dataset.features)


def save_config(output_dir: Path, config_name: str, dataset_dict: DatasetDict) -> None:
    target = output_dir / "dataset" / config_name
    if target.exists():
        # Datasets refuses to overwrite. Remove only this script's generated config dir.
        import shutil

        shutil.rmtree(target)
    dataset_dict.save_to_disk(target)


def write_dataset_card(output_dir: Path, source_dataset: str, layout: dict[str, list[str]]) -> Path:
    card_path = output_dir / "README.md"
    splits = sorted({split for split_names in layout.values() for split in split_names})
    card = f"""---
language:
- ne
license: apache-2.0
task_categories:
- sentence-similarity
- feature-extraction
pretty_name: NanoBEIR Nepali
source_datasets:
- {source_dataset}
---

# NanoBEIR Nepali

This dataset is a Nepali translation of
[`{source_dataset}`](https://huggingface.co/datasets/{source_dataset}).

It preserves the original NanoBEIR config names, split names, IDs, qrels, and
BM25 candidate ID lists. The `text` fields in the `queries` and `corpus`
configs are translated to Nepali; ID-only configs are copied unchanged.

Configs: {", ".join(layout)}

Splits: {", ".join(splits)}
"""
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    return card_path


def push_configs(
    output_dir: Path,
    repo_id: str,
    layout: dict[str, list[str]],
    private: bool,
    commit_message: str,
) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for config_name in layout:
        dataset_path = output_dir / "dataset" / config_name
        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        dataset_dict.push_to_hub(
            repo_id,
            config_name=config_name,
            private=private,
            commit_message=f"{commit_message}: {config_name}",
        )

    card_path = output_dir / "README.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update dataset card",
        )


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if not args.show_progress:
        disable_progress_bar()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.local_parallel_requests < 0:
        raise ValueError("--local-parallel-requests must be >= 0.")
    if args.openrouter_parallel_requests is not None and args.openrouter_parallel_requests < 0:
        raise ValueError("--openrouter-parallel-requests must be >= 0.")

    layout = discover_layout(args.source_dataset, args.configs, args.splits)
    card_path = write_dataset_card(args.output_dir, args.source_dataset, layout)
    print(f"Wrote dataset card: {card_path}", flush=True)

    clients = build_clients(args)
    parallel_requests = args.parallel_requests or len(clients)
    if parallel_requests < 1:
        raise ValueError("--parallel-requests must be at least 1.")
    print(
        f"Enabled LLM clients: {', '.join(client.name for client in clients)} "
        f"(parallel requests: {parallel_requests})",
        flush=True,
    )
    cache = TranslationCache(args.output_dir / "translation_cache.sqlite3")

    try:
        for config_name, split_names in layout.items():
            target = args.output_dir / "dataset" / config_name
            if args.skip_existing and target.exists():
                print(f"Skipping existing config: {config_name}", flush=True)
                continue

            translated_splits: dict[str, Dataset] = {}
            for split_name in split_names:
                print(f"Loading {args.source_dataset} / {config_name} / {split_name}", flush=True)
                dataset = load_split(args.source_dataset, config_name, split_name, args.limit)
                print(f"Processing {config_name}/{split_name}: {len(dataset)} rows", flush=True)
                translated_splits[split_name] = translate_dataset(
                    dataset=dataset,
                    config_name=config_name,
                    split_name=split_name,
                    source_dataset=args.source_dataset,
                    clients=clients,
                    cache=cache,
                    batch_size=args.batch_size,
                    max_batch_chars=args.max_batch_chars,
                    parallel_requests=parallel_requests,
                    log_llm_requests=args.log_llm_requests,
                )

            dataset_dict = DatasetDict(translated_splits)
            save_config(args.output_dir, config_name, dataset_dict)
            print(f"Saved config: {target}", flush=True)
    finally:
        cache.close()

    if args.push_to_hub:
        if not args.repo_id:
            raise RuntimeError("--repo-id is required with --push-to-hub.")
        push_configs(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            layout=layout,
            private=args.private,
            commit_message=args.commit_message,
        )
        print(f"Pushed dataset configs to: {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()