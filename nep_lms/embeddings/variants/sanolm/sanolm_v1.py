from typing import Callable

import datasets
from sentence_transformers.base import BatchSamplers
from sentence_transformers.sentence_transformer import SentenceTransformer, losses

from nep_lms.embeddings.experiment_scopes import GeneralSentenceSimilarityExperiment
from nep_lms.embeddings.variants.base import BaseSTEmbeddingVariant


class SanoLMV1Variant(BaseSTEmbeddingVariant):
    experiment_cls = GeneralSentenceSimilarityExperiment

    def __init__(self, experiment: GeneralSentenceSimilarityExperiment | None = None):
        super().__init__(
            hub_model_id="jangedoo/sanolm-v1",
            experiment=experiment,
        )

    def get_model(self) -> SentenceTransformer:
        return SentenceTransformer("jangedoo/sanolm-v1-base")

    def get_loss(self, model: SentenceTransformer) -> Callable | dict[str, Callable]:
        def symmetric_mnrl():
            return losses.MultipleNegativesRankingLoss(
                model,
                directions=("query_to_doc", "doc_to_query"),
                partition_mode="per_direction",
            )

        return {
            "title_excerpt": symmetric_mnrl(),
            "ne_en": symmetric_mnrl(),
            "excerpt_paraphrase": symmetric_mnrl(),
            # Exact English↔Nepali translations.
            "stsb_translation_1": symmetric_mnrl(),
            "stsb_translation_2": symmetric_mnrl(),
            "nepali_triplets": losses.TripletLoss(
                model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.2,
            ),
            "stsb_en": losses.CosineSimilarityLoss(model),
            "stsb_ne": losses.CoSENTLoss(model),
            # Absolute similarity matters across language spaces.
            "stsb_en_ne": losses.CosineSimilarityLoss(model),
            "stsb_ne_en": losses.CosineSimilarityLoss(model),
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

        stsb_translation_1_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence1", "sentence1_ne"]
        ).rename_column("sentence1_ne", "sentence2")

        stsb_translation_2_ds = self.experiment.nepali_stsb_ds.select_columns(
            ["sentence2", "sentence2_ne"]
        ).rename_columns(
            {
                "sentence2": "sentence1",
                "sentence2_ne": "sentence2",
            }
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
            # Repeat the small alignment corpora so they are not overwhelmed
            # by the ~100k title/excerpt examples.
            "ne_en": datasets.concatenate_datasets([ne_en_ds["train"]] * 4),
            "excerpt_paraphrase": paraphrase_ds["train"],
            "nepali_triplets": triplets_ds["train"],
            "stsb_translation_1": datasets.concatenate_datasets(
                [stsb_translation_1_ds["train"]] * 2
            ),
            "stsb_translation_2": datasets.concatenate_datasets(
                [stsb_translation_2_ds["train"]] * 2
            ),
            "stsb_en": stsb_en_ds["train"],
            "stsb_ne": stsb_ne_ds["train"],
            "stsb_en_ne": stsb_en_ne_ds["train"],
            "stsb_ne_en": stsb_ne_en_ds["train"],
        }

        eval_ds = {
            "title_excerpt": title_excerpt_ds["valid"],
            "ne_en": ne_en_ds["valid"],
            "excerpt_paraphrase": paraphrase_ds["valid"],
            "nepali_triplets": triplets_ds["valid"],
            "stsb_translation_1": stsb_translation_1_ds["valid"],
            "stsb_translation_2": stsb_translation_2_ds["valid"],
            "stsb_en": stsb_en_ds["valid"],
            "stsb_ne": stsb_ne_ds["valid"],
            "stsb_en_ne": stsb_en_ne_ds["valid"],
            "stsb_ne_en": stsb_ne_en_ds["valid"],
        }
        return train_ds, eval_ds

    def get_training_args(self):
        args = super().get_training_args()
        args.batch_sampler = BatchSamplers.NO_DUPLICATES
        args.metric_for_best_model = "stsb_ne_pearson_cosine"
        args.greater_is_better = True
        return args
