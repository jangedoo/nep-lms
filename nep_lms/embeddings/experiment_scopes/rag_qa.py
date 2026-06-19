from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

from nep_lms.embeddings.dataset_loaders import (
    select_max_rows,
    nepali_qa_9k_loader,
    nepali_query_passage_10k_loader,
)
from nep_lms.embeddings.evaluators import create_ir_evaluator_from_pair_dataset
from nep_lms.embeddings.experiment_scopes.base import BaseEmbeddingExperiment


class RagQaEmbeddingExperiment(BaseEmbeddingExperiment):
    name = "rag_qa_embedding"
    ds_name_to_loader = {
        "nepali_qa_9k": nepali_qa_9k_loader,
        "nepali_query_passage_10k": nepali_query_passage_10k_loader,
    }
    main_metrics = [
        "nepali_qa_9k_cosine_recall@10",
        "nepali_query_passage_10k_cosine_recall@10",
    ]

    @property
    def nepal_legal_rag_qa_ds(self):
        return self.get_dataset("nepal_legal_rag_qa")
    
    @property
    def nepali_qa_9k(self):
        return self.get_dataset("nepali_qa_9k")
    
    @property
    def nepali_query_passage_10k(self):
        return self.get_dataset("nepali_query_passage_10k")

    def _get_evaluator(self, max_rows: int | None = None):
        rag_evaluators = []

        rag_evaluators.append(
            create_ir_evaluator_from_pair_dataset(
                select_max_rows(self.nepali_qa_9k['valid'], max_rows=max_rows),
                evaluator_name="nepali_qa_9k", query_col="question", doc_col="answer"
            )
        )

        rag_evaluators.append(
            create_ir_evaluator_from_pair_dataset(
                select_max_rows(self.nepali_query_passage_10k['valid'], max_rows=max_rows),
                evaluator_name="nepali_query_passage_10k", query_col="query", doc_col="positive"
            )
        )

        rag_evaluator_count = max(1, len(rag_evaluators))
        return SequentialEvaluator(
            rag_evaluators,
            main_score_function=lambda scores: sum(scores) / rag_evaluator_count,
        )
