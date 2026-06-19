import datasets
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator


def create_ir_evaluator_from_parallel_corpus(
    ds: datasets.Dataset, query_col: str, doc_col: str, evaluator_name: str = ""
):
    queries = {str(i): text for i, text in enumerate(ds[query_col])}
    corpus = {str(i): text for i, text in enumerate(ds[doc_col])}
    relevant_docs = {qid: {qid} for qid in queries.keys()}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=evaluator_name,
        precision_recall_at_k=[10, 50],
        ndcg_at_k=[10],
        accuracy_at_k=[10],
        main_score_function="cosine",
    )


def create_ir_evaluator_from_pair_dataset(
    ds: datasets.Dataset,
    evaluator_name: str = "",
    query_col: str = "query",
    doc_col: str = "document",
):
    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, row in enumerate(ds):
        query = str(row[query_col]).strip()
        document = str(row[doc_col]).strip()
        if not query or not document:
            continue
        query_id = f"q{i}"
        doc_id = f"d{i}"
        queries[query_id] = query
        corpus[doc_id] = document
        relevant_docs[query_id] = {doc_id}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=evaluator_name,
        precision_recall_at_k=[10, 50],
        ndcg_at_k=[10],
        accuracy_at_k=[10],
        main_score_function="cosine",
    )
