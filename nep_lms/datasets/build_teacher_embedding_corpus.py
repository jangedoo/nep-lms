"""Build a reusable text corpus and attach native teacher embeddings.

The text corpus is materialised once below ``OUTPUT_DIR/corpus``.  Every teacher
configuration is then derived from those exact Parquet shards, which guarantees
that row IDs and train/validation assignments agree across teachers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

DEFAULT_REPO_ID = "jangedoo/teacher-embedding-corpus"
DEFAULT_OUTPUT_DIR = "data/teacher-embedding-corpus"
NEPALI_DATASET_ID = "jangedoo/nepali-corpus"
FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
REDDIT_DATASET_ID = "jangedoo/nepali-reddit"
REDDIT_CONFIGS = ("comments", "posts")
REDDIT_SPLITS = ("nepalsocial", "nepalstock", "technepal")
DEFAULT_CORPUS_SIZE = 1_000_000
DEFAULT_CORPUS_SHARD_SIZE = 10_000
CORPUS_FORMAT_VERSION = 2

CorpusLoader = Callable[..., Iterable[Mapping[str, Any]]]


def sanitize_config_name(model_id: str) -> str:
    """Convert a model ID into a stable Hugging Face configuration name."""

    components = []
    for component in model_id.strip().split("/"):
        component = re.sub(r"[^a-z0-9._-]+", "-", component.lower()).strip("-.")
        if component:
            components.append(component)
    if not components:
        raise ValueError(f"Cannot derive a configuration name from {model_id!r}")
    return "--".join(components)


def clean_text(value: Any) -> str:
    """Normalise supported scalar/list text fields and reject empty sentinels."""

    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (list, tuple)):
        text = "\n".join(part for item in value if (part := str(item).strip())).strip()
    else:
        return ""
    if text.casefold() in {"[deleted]", "[removed]", "deleted", "removed"}:
        return ""
    return text


def reddit_text(row: Mapping[str, Any]) -> str:
    """Extract comments or posts while tolerating historical Reddit schemas."""

    parts: list[str] = []
    seen: set[str] = set()
    for key in ("title", "selftext", "body", "text", "content"):
        part = clean_text(row.get(key))
        if part and part not in seen:
            parts.append(part)
            seen.add(part)
    return "\n".join(parts)


def stable_row_id(source: str, text: str) -> str:
    payload = f"{source}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_for_id(row_id: str) -> str:
    """Assign roughly 98/2 percent using only the persistent row ID."""

    return "validation" if int(row_id[:16], 16) % 100 < 2 else "train"


def _proportional_counts(total: int, weights: list[int]) -> list[int]:
    if total < 0:
        raise ValueError("Corpus size/limit cannot be negative")
    if total == 0:
        return [0] * len(weights)
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def corpus_source_quotas(
    total: int = DEFAULT_CORPUS_SIZE,
) -> list[tuple[str, str, int]]:
    """Return (source, language, quota) in deterministic traversal order."""

    nepali, english, reddit = _proportional_counts(total, [60, 20, 20])
    partitions = [
        (config, split) for config in REDDIT_CONFIGS for split in REDDIT_SPLITS
    ]
    reddit_counts = _proportional_counts(reddit, [1] * len(partitions))
    quotas = [
        (NEPALI_DATASET_ID, "ne", nepali),
        (FINEWEB_DATASET_ID, "en", english),
    ]
    quotas.extend(
        (
            f"{REDDIT_DATASET_ID}/{config}/{split}",
            "ne",
            count,
        )
        for (config, split), count in zip(partitions, reddit_counts, strict=True)
    )
    return quotas


def _default_loader(
    dataset_id: str,
    *,
    name: str | None = None,
    split: str = "train",
) -> Iterable[Mapping[str, Any]]:
    return datasets.load_dataset(dataset_id, name=name, split=split, streaming=True)


def _source_streams(
    loader: CorpusLoader,
) -> Iterator[tuple[str, str, Iterable, Callable]]:
    yield (
        NEPALI_DATASET_ID,
        "ne",
        loader(NEPALI_DATASET_ID, split="train"),
        lambda row: clean_text(row.get("text")),
    )
    yield (
        FINEWEB_DATASET_ID,
        "en",
        loader(FINEWEB_DATASET_ID, name=FINEWEB_CONFIG, split="train"),
        lambda row: clean_text(row.get("text")),
    )
    for config in REDDIT_CONFIGS:
        for split in REDDIT_SPLITS:
            source = f"{REDDIT_DATASET_ID}/{config}/{split}"
            yield (
                source,
                "ne",
                loader(REDDIT_DATASET_ID, name=config, split=split),
                reddit_text,
            )


CORPUS_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "language": datasets.Value("string"),
        "source": datasets.Value("string"),
    }
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_corpus_shard(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    datasets.Dataset.from_list(rows, features=CORPUS_FEATURES).to_parquet(temporary)
    os.replace(temporary, path)


def _validate_cached_corpus(corpus_dir: Path, total: int) -> dict[str, Any] | None:
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_quotas = corpus_source_quotas(total)
    expected_sources = {source for source, _, _ in expected_quotas}
    nepali_target, english_target, reddit_target = _proportional_counts(
        total, [60, 20, 20]
    )
    source_counts = metadata.get("source_counts", {})
    reddit_count = sum(
        source_counts.get(source, 0)
        for source in expected_sources
        if source.startswith(f"{REDDIT_DATASET_ID}/")
    )
    if (
        metadata.get("format_version") != CORPUS_FORMAT_VERSION
        or metadata.get("row_count") != total
        or set(source_counts) != expected_sources
        or source_counts.get(NEPALI_DATASET_ID) != nepali_target
        or source_counts.get(FINEWEB_DATASET_ID) != english_target
        or reddit_count != reddit_target
    ):
        raise ValueError(
            f"Cached corpus at {corpus_dir} was built with different settings; "
            "choose another --output-dir or remove that corpus cache"
        )
    paths = [corpus_dir / relative for relative in metadata.get("shards", [])]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(
            f"Cached corpus metadata at {corpus_dir} references missing shards"
        )
    return metadata


def build_or_load_corpus(
    output_dir: str | Path,
    *,
    total: int = DEFAULT_CORPUS_SIZE,
    loader: CorpusLoader = _default_loader,
    shard_size: int = DEFAULT_CORPUS_SHARD_SIZE,
) -> tuple[datasets.DatasetDict, dict[str, Any]]:
    """Build the deterministic corpus once, or validate and load its cache."""

    if total <= 0:
        raise ValueError("Corpus size/limit must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    corpus_dir = Path(output_dir) / "corpus"
    cached = _validate_cached_corpus(corpus_dir, total)
    if cached is not None:
        return _load_corpus(corpus_dir, cached), cached

    # A missing completion marker means any existing shards are incomplete.  Use
    # a new build directory rather than silently treating them as resumable data.
    if corpus_dir.exists() and any(corpus_dir.rglob("*.parquet")):
        raise ValueError(
            f"Incomplete corpus cache found at {corpus_dir}; remove it before rebuilding"
        )
    corpus_dir.mkdir(parents=True, exist_ok=True)

    source_specs = list(_source_streams(loader))
    quotas = {source: quota for source, _, quota in corpus_source_quotas(total)}
    seen_ids: set[str] = set()
    buffers: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    shard_numbers = {"train": 0, "validation": 0}
    relative_shards: list[str] = []
    split_counts = {"train": 0, "validation": 0}
    source_counts: dict[str, int] = {source: 0 for source, _, _, _ in source_specs}
    fingerprint = hashlib.sha256()

    def flush(split: str) -> None:
        rows = buffers[split]
        if not rows:
            return
        relative = Path(split) / f"part-{shard_numbers[split]:05d}.parquet"
        _write_corpus_shard(rows, corpus_dir / relative)
        relative_shards.append(relative.as_posix())
        shard_numbers[split] += 1
        buffers[split] = []

    def accept(source: str, language: str, text: str) -> bool:
        if not text:
            return False
        row_id = stable_row_id(source, text)
        if row_id in seen_ids:
            return False
        seen_ids.add(row_id)
        split = split_for_id(row_id)
        buffers[split].append(
            {"id": row_id, "text": text, "language": language, "source": source}
        )
        fingerprint.update(row_id.encode("ascii"))
        fingerprint.update(b"\n")
        split_counts[split] += 1
        source_counts[source] += 1
        if len(buffers[split]) >= shard_size:
            flush(split)
        return True

    # Nepali and English have fixed 60/20 percent quotas.
    for source, language, stream, extract_text in source_specs[:2]:
        quota = quotas[source]
        if quota == 0:
            continue
        for row in stream:
            if (
                accept(source, language, extract_text(row))
                and source_counts[source] == quota
            ):
                break
        if source_counts[source] != quota:
            raise ValueError(
                f"Source {source!r} yielded only {source_counts[source]:,} unique non-empty rows; "
                f"{quota:,} are required"
            )

    # Reddit's 20 percent quota is shared. Round-robin traversal keeps the
    # partitions balanced, while exhausted small partitions relinquish their
    # remaining share to partitions that still have usable rows.
    reddit_target = sum(quotas[source] for source, _, _, _ in source_specs[2:])
    reddit_accepted = 0
    active = [
        (source, language, iter(stream), extract_text)
        for source, language, stream, extract_text in source_specs[2:]
    ]
    while reddit_accepted < reddit_target and active:
        remaining = []
        for source, language, stream, extract_text in active:
            for row in stream:
                if accept(source, language, extract_text(row)):
                    reddit_accepted += 1
                    remaining.append((source, language, stream, extract_text))
                    break
            if reddit_accepted == reddit_target:
                break
        active = remaining
    if reddit_accepted != reddit_target:
        raise ValueError(
            f"Reddit partitions yielded only {reddit_accepted:,} unique non-empty rows; "
            f"{reddit_target:,} are required in total"
        )

    flush("train")
    flush("validation")
    metadata = {
        "format_version": CORPUS_FORMAT_VERSION,
        "row_count": total,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "fingerprint": fingerprint.hexdigest(),
        "id_algorithm": "sha256(source\\0text)",
        "split_algorithm": "first-64-bits(id) modulo 100; validation < 2",
        "shards": relative_shards,
    }
    _atomic_json(corpus_dir / "metadata.json", metadata)
    return _load_corpus(corpus_dir, metadata), metadata


def _load_corpus(corpus_dir: Path, metadata: Mapping[str, Any]) -> datasets.DatasetDict:
    files: dict[str, list[str]] = {"train": [], "validation": []}
    for relative in metadata["shards"]:
        split = Path(relative).parts[0]
        files[split].append(str(corpus_dir / relative))
    populated = {split: paths for split, paths in files.items() if paths}
    loaded = datasets.load_dataset(
        "parquet", data_files=populated, cache_dir=str(corpus_dir / ".hf-cache")
    )
    for split in files:
        if split not in loaded:
            loaded[split] = datasets.Dataset.from_dict(
                {name: [] for name in CORPUS_FEATURES}, features=CORPUS_FEATURES
            )
    return loaded


def _resolved_model_revision(model: SentenceTransformer, requested: str | None) -> str:
    try:
        transformer = model[0]
        revision = transformer.auto_model.config._commit_hash
        if revision:
            return revision
    except (AttributeError, IndexError, TypeError):
        pass
    return requested or "unknown"


def _embedding_features(dimension: int) -> datasets.Features:
    return datasets.Features(
        {
            **dict(CORPUS_FEATURES),
            "embedding": datasets.Sequence(datasets.Value("float32"), length=dimension),
        }
    )


def _shard_metadata_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    return all(actual.get(key) == value for key, value in expected.items())


def extract_teacher_embeddings(
    corpus: datasets.DatasetDict,
    corpus_metadata: Mapping[str, Any],
    *,
    model_id: str,
    output_dir: str | Path,
    model_revision: str | None = None,
    config_name: str | None = None,
    prompt_name: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    max_seq_length: int | None = None,
    model_dtype: str | None = None,
    trust_remote_code: bool = False,
    model: SentenceTransformer | None = None,
) -> tuple[datasets.DatasetDict, dict[str, Any]]:
    """Embed all cached corpus shards, validating completed shards on resume."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dtype_values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if model_dtype is not None and model_dtype not in dtype_values:
        raise ValueError(
            f"model_dtype must be one of {sorted(dtype_values)}, got {model_dtype!r}"
        )
    config_name = config_name or sanitize_config_name(model_id)
    if model is None:
        model_kwargs = (
            {"torch_dtype": dtype_values[model_dtype]}
            if model_dtype is not None
            else None
        )
        teacher = SentenceTransformer(
            model_id,
            revision=model_revision,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
    else:
        teacher = model
    if max_seq_length is not None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        teacher.max_seq_length = max_seq_length

    config_dir = Path(output_dir) / "teachers" / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, list[str]] = {"train": [], "validation": []}
    native_dimension: int | None = None
    resolved_revision = _resolved_model_revision(teacher, model_revision)
    prompts = getattr(teacher, "prompts", None) or {}
    prompt_text = prompts.get(prompt_name) if prompt_name is not None else None

    for split, split_dataset in corpus.items():
        for shard_index, start in enumerate(
            range(0, len(split_dataset), DEFAULT_CORPUS_SHARD_SIZE)
        ):
            stop = min(start + DEFAULT_CORPUS_SHARD_SIZE, len(split_dataset))
            output_path = config_dir / split / f"part-{shard_index:05d}.parquet"
            expected = {
                "model_id": model_id,
                "resolved_revision": resolved_revision,
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "max_seq_length": teacher.max_seq_length,
                "normalization": False,
                "corpus_fingerprint": corpus_metadata["fingerprint"],
                "start": start,
                "stop": stop,
            }
            # Older default-precision shards did not record compute dtype. Keep
            # them resumable, but never mix them with an explicitly selected dtype.
            if model_dtype is not None:
                expected["model_dtype"] = model_dtype
            if _shard_metadata_matches(output_path, expected):
                shard_meta = json.loads(output_path.with_suffix(".json").read_text())
                dimension = int(shard_meta["native_dimension"])
                if native_dimension is not None and native_dimension != dimension:
                    raise ValueError(
                        "Completed shards disagree on native embedding dimension"
                    )
                native_dimension = dimension
                completed[split].append(str(output_path))
                continue

            rows = split_dataset.select(range(start, stop))
            prompt_kwargs = (
                {"prompt_name": prompt_name}
                if prompt_name is not None
                else {"prompt": ""}
            )
            embeddings = teacher.encode(
                rows["text"],
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
                device=device,
                **prompt_kwargs,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
                raise ValueError(
                    f"Teacher returned shape {embeddings.shape}; expected ({len(rows)}, dimension)"
                )
            dimension = int(embeddings.shape[1])
            if dimension <= 0:
                raise ValueError("Teacher returned zero-dimensional embeddings")
            if native_dimension is not None and dimension != native_dimension:
                raise ValueError("Teacher embedding dimension changed between shards")
            native_dimension = dimension
            embedded = rows.add_column("embedding", embeddings.tolist()).cast(
                _embedding_features(dimension)
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp.parquet")
            embedded.to_parquet(temporary)
            os.replace(temporary, output_path)
            _atomic_json(
                output_path.with_suffix(".json"),
                {**expected, "native_dimension": dimension, "row_count": len(rows)},
            )
            completed[split].append(str(output_path))

    if native_dimension is None:
        raise ValueError("The corpus contains no rows to embed")
    metadata = {
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "config_name": config_name,
        "native_dimension": native_dimension,
        "prompt_settings": {"name": prompt_name, "text": prompt_text},
        "max_seq_length": teacher.max_seq_length,
        "normalization": False,
        "corpus_fingerprint": corpus_metadata["fingerprint"],
        "corpus_rows": corpus_metadata["row_count"],
        "extraction": {
            "batch_size": batch_size,
            "device": device or str(teacher.device),
            "model_dtype": model_dtype or "checkpoint-default",
            "trust_remote_code": trust_remote_code,
            "storage_dtype": "float32",
        },
        "shards": {
            split: [Path(path).relative_to(config_dir).as_posix() for path in paths]
            for split, paths in completed.items()
        },
    }
    metadata_path = config_dir / "metadata.json"
    _atomic_json(metadata_path, metadata)
    data_files = {split: paths for split, paths in completed.items() if paths}
    embedded_corpus = datasets.load_dataset(
        "parquet", data_files=data_files, cache_dir=str(config_dir / ".hf-cache")
    )
    for split in completed:
        if split not in embedded_corpus:
            embedded_corpus[split] = datasets.Dataset.from_dict(
                {name: [] for name in _embedding_features(native_dimension)},
                features=_embedding_features(native_dimension),
            )
    return embedded_corpus, metadata


def push_configuration(
    dataset: datasets.DatasetDict,
    metadata: Mapping[str, Any],
    *,
    repo_id: str = DEFAULT_REPO_ID,
) -> None:
    """Upload one self-contained dataset configuration and its metadata file."""

    config_name = str(metadata["config_name"])
    dataset.push_to_hub(repo_id, config_name=config_name)
    HfApi().upload_file(
        path_or_fileobj=(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        path_in_repo=f"metadata/{config_name}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add metadata for {config_name}",
    )


def build_teacher_configuration(
    *,
    model_id: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_revision: str | None = None,
    config_name: str | None = None,
    prompt_name: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    max_seq_length: int | None = None,
    model_dtype: str | None = None,
    limit: int | None = None,
    trust_remote_code: bool = False,
    push_to_hub: bool = False,
    repo_id: str = DEFAULT_REPO_ID,
) -> tuple[datasets.DatasetDict, dict[str, Any]]:
    total = limit if limit is not None else DEFAULT_CORPUS_SIZE
    corpus, corpus_metadata = build_or_load_corpus(output_dir, total=total)
    result, metadata = extract_teacher_embeddings(
        corpus,
        corpus_metadata,
        model_id=model_id,
        model_revision=model_revision,
        config_name=config_name,
        prompt_name=prompt_name,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
        model_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        output_dir=output_dir,
    )
    if push_to_hub:
        push_configuration(result, metadata, repo_id=repo_id)
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reusable corpus configuration with native teacher embeddings"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--config-name")
    parser.add_argument("--prompt-name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        help="Teacher compute dtype; stored embeddings remain float32",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()
    _, metadata = build_teacher_configuration(**vars(args))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
