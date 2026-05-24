from sentence_transformers.evaluation import SequentialEvaluator

from nep_lms.embeddings.dataset_loaders import (
    indic_rag_ne_loader,
    nepal_legal_rag_qa_loader,
    select_max_rows,
    textbook_qa_nepali_context_loader,
    textbook_qa_nepali_rephrased_loader,
    yunika_nepali_qa_loader,
)
from nep_lms.embeddings.evaluators import create_ir_evaluator_from_pair_dataset
from nep_lms.embeddings.experiment_scopes.base import BaseEmbeddingExperiment


class RagQaEmbeddingExperiment(BaseEmbeddingExperiment):
    name = "rag_qa_embedding"
    ds_name_to_loader = {
        "indic_rag_ne": indic_rag_ne_loader,
        "textbook_qa_nepali_context": textbook_qa_nepali_context_loader,
        "textbook_qa_nepali_rephrased": textbook_qa_nepali_rephrased_loader,
        "yunika_nepali_qa": yunika_nepali_qa_loader,
        "nepal_legal_rag_qa": nepal_legal_rag_qa_loader,
    }
    main_metrics = [
        "indic_rag_ne_cosine_ndcg@10",
        "textbook_qa_nepali_cosine_ndcg@10",
        "yunika_nepali_qa_cosine_ndcg@10",
        "nepal_legal_rag_qa_cosine_ndcg@10",
    ]

    @property
    def indic_rag_ne_ds(self):
        return self.get_dataset("indic_rag_ne")

    @property
    def textbook_qa_nepali_context_ds(self):
        return self.get_dataset("textbook_qa_nepali_context")

    @property
    def textbook_qa_nepali_rephrased_ds(self):
        return self.get_dataset("textbook_qa_nepali_rephrased")

    @property
    def yunika_nepali_qa_ds(self):
        return self.get_dataset("yunika_nepali_qa")

    @property
    def nepal_legal_rag_qa_ds(self):
        return self.get_dataset("nepal_legal_rag_qa")

    def _get_evaluator(self, max_rows: int | None = None):
        rag_evaluators = []

        indic_rag_valid_ds = select_max_rows(
            self.indic_rag_ne_ds["valid"], max_rows=max_rows
        )
        if len(indic_rag_valid_ds) > 0:
            rag_evaluators.append(
                create_ir_evaluator_from_pair_dataset(
                    ds=indic_rag_valid_ds,
                    evaluator_name="indic_rag_ne",
                )
            )

        textbook_valid_ds = select_max_rows(
            self.textbook_qa_nepali_context_ds["valid"], max_rows=max_rows
        )
        if len(textbook_valid_ds) > 0:
            rag_evaluators.append(
                create_ir_evaluator_from_pair_dataset(
                    ds=textbook_valid_ds,
                    evaluator_name="textbook_qa_nepali",
                )
            )

        yunika_valid_ds = select_max_rows(
            self.yunika_nepali_qa_ds["valid"], max_rows=max_rows
        )
        if len(yunika_valid_ds) > 0:
            rag_evaluators.append(
                create_ir_evaluator_from_pair_dataset(
                    ds=yunika_valid_ds,
                    evaluator_name="yunika_nepali_qa",
                )
            )

        try:
            legal_valid_ds = select_max_rows(
                self.nepal_legal_rag_qa_ds["valid"], max_rows=max_rows
            )
            if len(legal_valid_ds) > 0:
                rag_evaluators.append(
                    create_ir_evaluator_from_pair_dataset(
                        ds=legal_valid_ds,
                        evaluator_name="nepal_legal_rag_qa",
                    )
                )
        except Exception as exc:
            self.logger.warning("Skipping optional legal RAG evaluator: %s", exc)

        rag_evaluator_count = max(1, len(rag_evaluators))
        return SequentialEvaluator(
            rag_evaluators,
            main_score_function=lambda scores: sum(scores) / rag_evaluator_count,
        )
