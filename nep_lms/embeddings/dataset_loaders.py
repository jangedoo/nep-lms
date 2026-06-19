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



def nepali_qa_9k_loader():
    ds = datasets.load_dataset("jangedoo/nepali-qa-9k")
    return ensure_train_test_valid_splits(ds=ds, seed=10)

def nepali_nli_20k_loader():
    ds = datasets.load_dataset("jangedoo/nepali-nli-20k")
    return ensure_train_test_valid_splits(ds=ds, seed=10)

def nepali_query_passage_10k_loader():
    ds = datasets.load_dataset("jangedoo/nepali-query-passage-hard-negatives-10k")

    def parse_negs(x):
        negs = x["hard_negative_passages"]

        # make sure there are at least 3 negatives. in most of the case they are
        negs = negs[:3]
        if len(negs) != 3:
            while len(negs) != 3:
                negs.append(negs[-1])

        return {
            "query": x["query"],
            "positive": x["positive"],
            "negative_1": negs[0],
            "negative_2": negs[1],
            "negative_3": negs[2],
        }
    
    ds = ds.map(parse_negs, remove_columns=["hard_negative_passages"])
    
    return ensure_train_test_valid_splits(ds=ds, seed=10)
