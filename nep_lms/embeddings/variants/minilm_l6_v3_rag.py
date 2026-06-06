from typing import Callable

import datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses, training_args

from nep_lms.embeddings.dataset_loaders import (
    LEGAL_TRAIN_MAX_ROWS,
    RAG_TRAIN_MAX_ROWS,
    select_max_rows,
)
from nep_lms.embeddings.experiment_scopes import RagQaEmbeddingExperiment
from nep_lms.embeddings.variants.base import BaseSTEmbeddingVariant


class MiniLML6V3RagVariant(BaseSTEmbeddingVariant):
    experiment_cls = RagQaEmbeddingExperiment
    effective_batch_size = 256

    def __init__(self, experiment: RagQaEmbeddingExperiment | None = None):
        super().__init__(
            hub_model_id="jangedoo/all-MiniLM-L6-v3-nepali-rag",
            experiment=experiment,
        )

    def get_model(self) -> SentenceTransformer:
        return SentenceTransformer("jangedoo/all-MiniLM-L6-v3-nepali")

    def _ranking_loss(self, model: SentenceTransformer):
        return losses.MultipleNegativesRankingLoss(
            model,
            directions=("query_to_doc", "doc_to_query"),
            partition_mode="per_direction",
        )

    def get_loss(self, model: SentenceTransformer) -> Callable | dict[str, Callable]:
        train_keys = set(self.train_ds.keys())
        return {key: self._ranking_loss(model) for key in train_keys}

    def get_train__eval_ds(
        self,
    ) -> (
        tuple[datasets.Dataset, datasets.Dataset]
        | tuple[dict[str, datasets.Dataset], dict[str, datasets.Dataset]]
    ):
        try:
            legal_rag_qa_ds = self.experiment.nepal_legal_rag_qa_ds
        except Exception as exc:
            self.experiment.logger.warning(
                "Skipping optional legal RAG training data: %s", exc
            )
            legal_rag_qa_ds = None

        train_ds = {
            "indic_rag_ne": self.experiment.indic_rag_ne_ds["train"],
            "textbook_qa_context": self.experiment.textbook_qa_nepali_context_ds[
                "train"
            ],
            "textbook_qa_rephrased": self.experiment.textbook_qa_nepali_rephrased_ds[
                "train"
            ],
            "yunika_nepali_qa": select_max_rows(
                self.experiment.yunika_nepali_qa_ds["train"].shuffle(seed=10),
                max_rows=RAG_TRAIN_MAX_ROWS,
            ),
        }
        if legal_rag_qa_ds is not None:
            train_ds["nepal_legal_rag_qa"] = select_max_rows(
                legal_rag_qa_ds["train"],
                max_rows=LEGAL_TRAIN_MAX_ROWS,
            )

        eval_ds = {
            "indic_rag_ne": self.experiment.indic_rag_ne_ds["test"],
            "textbook_qa_context": self.experiment.textbook_qa_nepali_context_ds[
                "test"
            ],
            "textbook_qa_rephrased": self.experiment.textbook_qa_nepali_rephrased_ds[
                "test"
            ],
            "yunika_nepali_qa": self.experiment.yunika_nepali_qa_ds["test"],
        }
        if legal_rag_qa_ds is not None:
            eval_ds["nepal_legal_rag_qa"] = legal_rag_qa_ds["test"]
        return train_ds, eval_ds

    def get_training_args(self):
        args = super().get_training_args()
        args.learning_rate = 2e-5
        args.per_device_train_batch_size = 64
        args.gradient_accumulation_steps = 4
        args.gradient_checkpointing = False
        args.num_train_epochs = 1
        args.warmup_steps = 0.05
        args.eval_steps = 250
        args.save_steps = 250
        args.logging_steps = 50
        args.metric_for_best_model = "sequential_score"
        args.greater_is_better = True
        args.batch_sampler = training_args.BatchSamplers.NO_DUPLICATES
        return args

    def get_trainer(
        self, model, training_args, train_ds, eval_ds, loss, evaluator=None
    ):
        return super().get_trainer(
            model=model,
            training_args=training_args,
            train_ds=train_ds,
            eval_ds=None,
            loss=loss,
            evaluator=evaluator,
        )

    def train(
        self,
        epochs: float = 1,
        max_steps: int = -1,
        batch_size: int = 64,
        lr: float = 2e-5,
        early_stop=True,
        early_stop_patience=3,
        push_to_hub: bool = False,
    ):
        return super().train(
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            lr=lr,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            push_to_hub=push_to_hub,
        )
