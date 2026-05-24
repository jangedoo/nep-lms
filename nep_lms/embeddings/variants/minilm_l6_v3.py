from typing import Callable

import datasets
from sentence_transformers import SentenceTransformer, losses

from nep_lms.embeddings.experiment_scopes import GeneralSentenceSimilarityExperiment
from nep_lms.embeddings.variants.base import BaseSTEmbeddingVariant


class MiniLML6V3Variant(BaseSTEmbeddingVariant):
    experiment_cls = GeneralSentenceSimilarityExperiment

    def __init__(self, experiment: GeneralSentenceSimilarityExperiment | None = None):
        super().__init__(
            hub_model_id="jangedoo/all-MiniLM-L6-v3-nepali",
            experiment=experiment,
        )

    def get_model(self) -> SentenceTransformer:
        return SentenceTransformer("jangedoo/all-MiniLM-L6-v2-nepali")

    def get_loss(self, model: SentenceTransformer) -> Callable | dict[str, Callable]:
        cosent_loss = losses.CoSENTLoss(model)
        return {
            "title_excerpt": losses.MultipleNegativesSymmetricRankingLoss(model),
            "ne_en": losses.MultipleNegativesSymmetricRankingLoss(model),
            "excerpt_paraphrase": losses.MultipleNegativesSymmetricRankingLoss(model),
            "nepali_triplets": losses.TripletLoss(
                model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.2,
            ),
            "stsb_en": losses.CosineSimilarityLoss(model),
            "stsb_ne": cosent_loss,
            "stsb_en_ne": cosent_loss,
            "stsb_ne_en": cosent_loss,
        }

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

        stsb_ne_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence1_ne", "sentence2_ne", "score"]
        ).rename_columns({"sentence1_ne": "sentence1", "sentence2_ne": "sentence2"})

        stsb_en_ne_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence1", "sentence2_ne", "score"]
        ).rename_columns({"sentence2_ne": "sentence2"})

        stsb_ne_en_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence1_ne", "sentence2", "score"]
        ).rename_columns({"sentence1_ne": "sentence1"})

        stsb_en_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence1", "sentence2", "score"]
        )

        train_ds = {
            "title_excerpt": title_excerpt_ds["train"],
            "ne_en": ne_en_ds["train"],
            "excerpt_paraphrase": paraphrase_ds["train"],
            "nepali_triplets": triplets_ds["train"],
            "stsb_en": stsb_en_ds["train"],
            "stsb_ne": stsb_ne_ds["train"],
            "stsb_en_ne": stsb_en_ne_ds["train"],
            "stsb_ne_en": stsb_ne_en_ds["train"],
        }

        eval_ds = {
            "title_excerpt": title_excerpt_ds["test"],
            "ne_en": ne_en_ds["test"],
            "excerpt_paraphrase": paraphrase_ds["test"],
            "nepali_triplets": triplets_ds["test"],
            "stsb_en": stsb_en_ds["test"],
            "stsb_ne": stsb_ne_ds["test"],
            "stsb_en_ne": stsb_en_ne_ds["test"],
            "stsb_ne_en": stsb_ne_en_ds["test"],
        }
        return train_ds, eval_ds

    def get_training_args(self):
        args = super().get_training_args()
        args.metric_for_best_model = "stsb_ne_pearson_cosine"
        args.greater_is_better = True
        return args
