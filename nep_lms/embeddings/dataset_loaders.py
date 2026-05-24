import datasets


RAG_TRAIN_MAX_ROWS = 50_000
TEXTBOOK_TRAIN_MAX_ROWS = 10_000
GUARDRAIL_TRAIN_MAX_ROWS = 10_000
LEGAL_TRAIN_MAX_ROWS = 1_500


def select_max_rows(
    ds: datasets.Dataset, max_rows: int | None = None
) -> datasets.Dataset:
    if max_rows is None:
        return ds
    return ds.select(range(min(max_rows, len(ds))))


def split_ds_to_train_test_valid(ds: datasets.Dataset, seed: int = 10):
    if len(ds) < 5:
        return datasets.DatasetDict({"train": ds, "test": ds, "valid": ds})

    train_rest_ds = ds.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_rest_ds["test"].train_test_split(test_size=0.2, seed=seed)
    train_test_valid = datasets.DatasetDict(
        {
            "train": train_rest_ds["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    return train_test_valid


def ensure_train_test_valid_splits(
    ds: datasets.Dataset | datasets.DatasetDict, seed: int = 10
) -> datasets.DatasetDict:
    if isinstance(ds, datasets.Dataset):
        return split_ds_to_train_test_valid(ds=ds, seed=seed)

    ds_by_split = dict(ds)
    if "valid" not in ds_by_split and "validation" in ds_by_split:
        ds_by_split["valid"] = ds_by_split["validation"]

    if "train" not in ds_by_split:
        first_split_name = next(iter(ds_by_split))
        return split_ds_to_train_test_valid(ds=ds_by_split[first_split_name], seed=seed)

    if "test" not in ds_by_split and "valid" not in ds_by_split:
        return split_ds_to_train_test_valid(ds=ds_by_split["train"], seed=seed)

    if "valid" not in ds_by_split:
        if len(ds_by_split["train"]) < 2:
            ds_by_split["valid"] = ds_by_split["train"]
        else:
            train_valid = ds_by_split["train"].train_test_split(
                test_size=0.1, seed=seed
            )
            ds_by_split["train"] = train_valid["train"]
            ds_by_split["valid"] = train_valid["test"]

    if "test" not in ds_by_split:
        ds_by_split["test"] = ds_by_split["valid"]

    return datasets.DatasetDict(
        {
            "train": ds_by_split["train"],
            "test": ds_by_split["test"],
            "valid": ds_by_split["valid"],
        }
    )


def cap_train_split(
    ds: datasets.DatasetDict, max_rows: int, seed: int = 10
) -> datasets.DatasetDict:
    capped = datasets.DatasetDict(ds)
    capped["train"] = select_max_rows(
        capped["train"].shuffle(seed=seed), max_rows=max_rows
    )
    return capped


def filter_non_empty_pair(
    ds: datasets.Dataset, query_col: str = "query", doc_col: str = "document"
) -> datasets.Dataset:
    return ds.filter(
        lambda row: bool(str(row[query_col]).strip())
        and bool(str(row[doc_col]).strip())
    )


def map_pair_dataset(
    ds: datasets.Dataset,
    query_col: str,
    doc_col: str,
    query_name: str = "query",
    doc_name: str = "document",
) -> datasets.Dataset:
    mapped = ds.map(
        lambda row: {
            query_name: str(row[query_col]).strip(),
            doc_name: str(row[doc_col]).strip(),
        },
        remove_columns=ds.column_names,
    )
    return filter_non_empty_pair(mapped, query_col=query_name, doc_col=doc_name)


def map_split_pair_dataset(
    ds: datasets.DatasetDict,
    query_col: str,
    doc_col: str,
) -> datasets.DatasetDict:
    return datasets.DatasetDict(
        {
            split: map_pair_dataset(split_ds, query_col=query_col, doc_col=doc_col)
            for split, split_ds in ds.items()
        }
    )


def first_human_turn(conversations) -> str:
    if not conversations:
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role")
        if role in {"human", "user"}:
            return str(turn.get("value") or turn.get("content") or "").strip()
    first_turn = conversations[0]
    if isinstance(first_turn, dict):
        return str(first_turn.get("value") or first_turn.get("content") or "").strip()
    return ""


def nepali_news_loader():
    ds = datasets.load_dataset("jangedoo/nepalinews", split="train")
    return split_ds_to_train_test_valid(ds=ds, seed=10)


def en_ne_parallel_corpus_loader():
    return datasets.load_dataset("jangedoo/en_ne_parallel_corpus")


def paraphrase_loader():
    return datasets.load_dataset("jangedoo/paraphrase-nepali")


def nepali_triplets_loader():
    return datasets.load_dataset("jangedoo/nepali-triplets")


def nepali_stsb_loader():
    return datasets.load_dataset("jangedoo/stsb_nepali")


def indic_rag_ne_loader():
    ds = datasets.load_dataset("ai4bharat/Indic-Rag-Suite", "ne")
    ds = ensure_train_test_valid_splits(ds=ds, seed=10)
    ds = map_split_pair_dataset(ds=ds, query_col="question", doc_col="paragraph")
    return cap_train_split(ds=ds, max_rows=RAG_TRAIN_MAX_ROWS, seed=10)


def textbook_qa_nepali_loader(doc_col: str = "context_text"):
    ds = datasets.load_dataset("dineshkarki/textbooks-qa-nepali")
    ds = ensure_train_test_valid_splits(ds=ds, seed=10)

    def add_query(row):
        return {
            "query": first_human_turn(row["conversations"]),
            "document": str(row[doc_col]).strip(),
        }

    split_datasets = {}
    for split, split_ds in ds.items():
        if "average_score" in split_ds.column_names:
            split_ds = split_ds.filter(
                lambda row: row["average_score"] is None or row["average_score"] >= 9.0
            )
        mapped = split_ds.map(add_query, remove_columns=split_ds.column_names)
        split_datasets[split] = filter_non_empty_pair(mapped)

    return cap_train_split(
        ds=datasets.DatasetDict(split_datasets),
        max_rows=TEXTBOOK_TRAIN_MAX_ROWS,
        seed=10,
    )


def textbook_qa_nepali_context_loader():
    return textbook_qa_nepali_loader(doc_col="context_text")


def textbook_qa_nepali_rephrased_loader():
    return textbook_qa_nepali_loader(doc_col="rephrased_text")


def yunika_nepali_qa_loader():
    ds = datasets.load_dataset("Yunika/Nepali-QA")
    ds = ensure_train_test_valid_splits(ds=ds, seed=10)

    def add_pair(row):
        data = row.get("data") or row
        return {
            "query": str(data["question"]).strip(),
            "document": str(data["context"]).strip(),
        }

    return datasets.DatasetDict(
        {
            split: filter_non_empty_pair(
                split_ds.map(add_pair, remove_columns=split_ds.column_names)
            )
            for split, split_ds in ds.items()
        }
    )


def nepal_legal_rag_qa_loader():
    ds = datasets.load_dataset("chhatramani/nepal_5_law_RAG_QA")
    ds = ensure_train_test_valid_splits(ds=ds, seed=10)
    ds = map_split_pair_dataset(ds=ds, query_col="instruction", doc_col="input")
    return cap_train_split(ds=ds, max_rows=LEGAL_TRAIN_MAX_ROWS, seed=10)
