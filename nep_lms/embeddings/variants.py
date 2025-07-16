import abc
from typing import Callable
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers import util, training_args, losses
import datasets
import transformers
from nep_lms.embeddings.experiment import (
    EmbeddingExperiment,
    split_ds_to_train_test_valid,
)
import pandas as pd


class BaseSTEmbeddingVariant:
    experiment = EmbeddingExperiment()

    def __init__(self, hub_model_id: str | None = None):
        self.hub_model_id = hub_model_id
        self._model: SentenceTransformer = None
        self._train_eval_ds: (
            tuple[datasets.Dataset, datasets.Dataset]
            | tuple[dict[str, datasets.Dataset], dict[str, datasets.Dataset]]
        ) = None
        self._loss: Callable | dict[str, Callable] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self.get_model()
        return self._model

    @property
    def train_ds(self) -> datasets.Dataset | dict[str, datasets.Dataset]:
        if self._train_eval_ds is None:
            self._train_eval_ds = self.get_train__eval_ds()

        train_ds, _ = self._train_eval_ds
        return train_ds

    @property
    def eval_ds(self) -> datasets.Dataset | dict[str, datasets.Dataset]:
        if self._train_eval_ds is None:
            self._train_eval_ds = self.get_train__eval_ds()

        _, eval_ds = self._train_eval_ds
        return eval_ds

    @property
    def loss(self) -> Callable | dict[str, Callable]:
        if self._loss is None:
            self._loss = self.get_loss(self.model)
        return self._loss

    @abc.abstractmethod
    def get_model(self) -> SentenceTransformer:
        pass

    @abc.abstractmethod
    def get_loss(self, model: SentenceTransformer) -> Callable | dict[str, Callable]:
        pass

    @abc.abstractmethod
    def get_train__eval_ds(
        self,
    ) -> (
        tuple[datasets.Dataset, datasets.Dataset]
        | tuple[dict[str, datasets.Dataset], dict[str, datasets.Dataset]]
    ):
        pass

    def get_training_args(self):
        return SentenceTransformerTrainingArguments(
            output_dir=None,
            num_train_epochs=3,
            max_steps=-1,
            per_device_train_batch_size=64,
            learning_rate=2e-6,
            warmup_ratio=0.1,
            fp16=util.get_device_name() != "cpu",
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=100,
        )

    def get_trainer(
        self,
        model: SentenceTransformer,
        training_args: SentenceTransformerTrainingArguments,
        train_ds: datasets.Dataset | dict[str, datasets.Dataset],
        eval_ds: datasets.Dataset | dict[str, datasets.Dataset],
        loss: Callable | dict[str, Callable],
    ) -> SentenceTransformerTrainer:
        return SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            loss=loss,
        )

    def train(
        self,
        epochs: float = 3,
        max_steps: int = -1,
        batch_size: int = 64,
        lr: float = 2e-5,
        early_stop=True,
        early_stop_patience=3,
        push_to_hub: bool = False,
    ):
        training_args = self.get_training_args()
        training_args.num_train_epochs = epochs
        training_args.max_steps = max_steps
        training_args.per_device_train_batch_size = batch_size
        training_args.learning_rate = lr

        if early_stop and training_args.metric_for_best_model is None:
            raise ValueError("metric_for_best_model is required to use early stopping")

        trainer = self.get_trainer(
            model=self.model,
            training_args=training_args,
            train_ds=self.train_ds,
            eval_ds=self.eval_ds,
            loss=self.loss,
        )
        if early_stop:
            training_args.load_best_model_at_end = True
            trainer.add_callback(
                transformers.trainer_callback.EarlyStoppingCallback(
                    early_stopping_patience=early_stop_patience
                )
            )
        trainer.train()
        if push_to_hub:
            if self.hub_model_id is None:
                raise ValueError("hub_model_id is required to push to hub")
            self.push_to_hub(self.hub_model_id)
        return trainer

    def push_to_hub(self, hub_model_id: str | None = None):
        model_id = hub_model_id or self.hub_model_id
        if model_id is None:
            raise ValueError("hub_model_id is required to push to hub")
        self.model.push_to_hub(repo_id=model_id, exist_ok=True)


class MiniLML6V3Variant(BaseSTEmbeddingVariant):
    def __init__(self):
        super().__init__(hub_model_id="jangedoo/all-MiniLM-L6-v3-nepali")

    def get_model(self) -> SentenceTransformer:
        return SentenceTransformer("jangedoo/all-MiniLM-L6-v3-nepali")

    def get_loss(self, model: SentenceTransformer) -> Callable | dict[str, Callable]:
        loss = {
            # "title_excerpt": losses.MultipleNegativesSymmetricRankingLoss(model),
            # "ne_en": losses.MultipleNegativesSymmetricRankingLoss(model),
            # "excerpt_paraphrase": losses.MultipleNegativesSymmetricRankingLoss(model),
            "nepali_triplets": losses.TripletLoss(
                model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.2,
            ),
        }
        return loss

    def get_train__eval_ds(
        self,
    ) -> (
        tuple[datasets.Dataset, datasets.Dataset]
        | tuple[dict[str, datasets.Dataset], dict[str, datasets.Dataset]]
    ):
        title_excerpt_ds = self.experiment.nepali_news_ds.select_columns(
            ["title", "excerpt"]
        )

        ne_en_ds = self.experiment.en_ne_parallel_corpus_ds.select_columns(
            ["title", "translation"]
        )
        paraphrase_ds = self.experiment.paraphrase_ds.filter(
            lambda row: row["label"] == 1
        ).select_columns(["sentence1", "sentence2"])

        triplets_ds = self.experiment.nepali_triplets_ds.select_columns(
            ["sentence", "positive_sentence", "negative_sentence"]
        )

        train_ds = {
            # "title_excerpt": title_excerpt_ds["train"],
            # "ne_en": ne_en_ds["train"],
            # "excerpt_paraphrase": paraphrase_ds["train"],
            "nepali_triplets": triplets_ds["train"],
        }

        eval_ds = {
            # "title_excerpt": title_excerpt_ds["test"],
            # "ne_en": ne_en_ds["test"],
            # "excerpt_paraphrase": paraphrase_ds["test"],
            "nepali_triplets": triplets_ds["test"],
        }
        return train_ds, eval_ds

    def get_training_args(self):
        args = super().get_training_args()
        args.metric_for_best_model = "eval_nepali_triplets_loss"
        args.greater_is_better = False
        return args
