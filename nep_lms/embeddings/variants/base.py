import abc
from typing import Callable

import datasets
import transformers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator

from nep_lms.embeddings.experiment_scopes import BaseEmbeddingExperiment


class BaseSTEmbeddingVariant:
    experiment_cls: type[BaseEmbeddingExperiment] | None = None
    effective_batch_size = 64

    def __init__(
        self,
        hub_model_id: str | None = None,
        experiment: BaseEmbeddingExperiment | None = None,
    ):
        self.hub_model_id = hub_model_id
        self.experiment = experiment or self.get_experiment()
        self._model: SentenceTransformer = None
        self._train_eval_ds: (
            tuple[datasets.Dataset, datasets.Dataset]
            | tuple[dict[str, datasets.Dataset], dict[str, datasets.Dataset]]
        ) = None
        self._loss: Callable | dict[str, Callable] = None

    def get_experiment(self) -> BaseEmbeddingExperiment:
        if self.experiment_cls is None:
            raise ValueError("experiment_cls must be set by concrete variants")
        return self.experiment_cls()

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
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=2e-6,
            warmup_ratio=0.1,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=100,
            save_total_limit=3,
            gradient_checkpointing=True,
        )

    def get_trainer(
        self,
        model: SentenceTransformer,
        training_args: SentenceTransformerTrainingArguments,
        train_ds: datasets.Dataset | dict[str, datasets.Dataset],
        eval_ds: datasets.Dataset | dict[str, datasets.Dataset],
        loss: Callable | dict[str, Callable],
        evaluator: SentenceEvaluator | None = None,
    ) -> SentenceTransformerTrainer:
        return SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            loss=loss,
            evaluator=evaluator,
        )

    def train(
        self,
        epochs: float = 3,
        max_steps: int = -1,
        batch_size: int = 64,
        lr: float = 2e-6,
        early_stop=True,
        early_stop_patience=3,
        push_to_hub: bool = False,
    ):
        training_args = self.get_training_args()
        training_args.num_train_epochs = epochs
        training_args.max_steps = max_steps
        training_args.per_device_train_batch_size = batch_size
        training_args.gradient_accumulation_steps = max(
            1, self.effective_batch_size // batch_size
        )
        training_args.learning_rate = lr

        if early_stop and training_args.metric_for_best_model is None:
            raise ValueError("metric_for_best_model is required to use early stopping")

        trainer = self.get_trainer(
            model=self.model,
            training_args=training_args,
            train_ds=self.train_ds,
            eval_ds=self.eval_ds,
            loss=self.loss,
            evaluator=self.experiment.evaluator,
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
