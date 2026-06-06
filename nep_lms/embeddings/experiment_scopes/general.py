from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
    TripletEvaluator,
)

from nep_lms.embeddings.dataset_loaders import (
    en_ne_parallel_corpus_loader,
    nepali_news_loader,
    nepali_stsb_loader,
    nepali_triplets_loader,
    paraphrase_loader,
    select_max_rows,
)
from nep_lms.evaluators import create_ir_evaluator_from_parallel_corpus
from nep_lms.embeddings.experiment_scopes.base import BaseEmbeddingExperiment


class GeneralSentenceSimilarityExperiment(BaseEmbeddingExperiment):
    name = "general_sentence_similarity"
    ds_name_to_loader = {
        "nepali_news": nepali_news_loader,
        "en_ne_parallel_corpus": en_ne_parallel_corpus_loader,
        "paraphrase": paraphrase_loader,
        "nepali_triplets": nepali_triplets_loader,
        "nepali_stsb": nepali_stsb_loader,
    }
    main_metrics = [
        "translation_mean_accuracy",
        "en_ir_cosine_recall@10",
        "ne_ir_cosine_recall@10",
        "multi_lang_ir_cosine_recall@10",
        "nepali_triplets_cosine_accuracy",
        "stsb_en_pearson_cosine",
        "stsb_ne_pearson_cosine",
    ]

    @property
    def nepali_news_ds(self):
        return self.get_dataset("nepali_news")

    @property
    def en_ne_parallel_corpus_ds(self):
        return self.get_dataset("en_ne_parallel_corpus")

    @property
    def paraphrase_ds(self):
        return self.get_dataset("paraphrase")

    @property
    def nepali_triplets_ds(self):
        return self.get_dataset("nepali_triplets")

    @property
    def nepali_stsb_ds(self):
        return self.get_dataset("nepali_stsb")

    def _get_evaluator(self, max_rows: int | None = None):
        evaluators = []

        valid_ds = select_max_rows(self.nepali_news_ds["valid"], max_rows=max_rows)
        evaluators.append(
            create_ir_evaluator_from_parallel_corpus(
                ds=valid_ds,
                query_col="title",
                doc_col="excerpt",
                evaluator_name="multi_lang_ir",
            )
        )
        evaluators.append(
            create_ir_evaluator_from_parallel_corpus(
                ds=valid_ds.filter(lambda row: row["language"] == "en"),
                query_col="title",
                doc_col="excerpt",
                evaluator_name="en_ir",
            )
        )
        evaluators.append(
            create_ir_evaluator_from_parallel_corpus(
                ds=valid_ds.filter(lambda row: row["language"] == "ne"),
                query_col="title",
                doc_col="excerpt",
                evaluator_name="ne_ir",
            )
        )

        valid_ds = select_max_rows(
            self.en_ne_parallel_corpus_ds["valid"], max_rows=max_rows
        )
        evaluators.append(
            TranslationEvaluator(
                source_sentences=valid_ds["title"],
                target_sentences=valid_ds["translation"],
                name="translation",
            )
        )

        valid_ds = select_max_rows(self.nepali_triplets_ds["valid"], max_rows=max_rows)
        evaluators.append(
            TripletEvaluator(
                anchors=valid_ds["sentence"],
                positives=valid_ds["positive_sentence"],
                negatives=valid_ds["negative_sentence"],
                name="nepali_triplets",
                main_similarity_function="cosine",
                margin=0.1,
            )
        )

        valid_ds = select_max_rows(self.nepali_stsb_ds["valid"], max_rows=max_rows)
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=valid_ds["sentence1"],
                sentences2=valid_ds["sentence2"],
                scores=valid_ds["score"],
                name="stsb_en",
                main_similarity="cosine",
            )
        )
        evaluators.append(
            EmbeddingSimilarityEvaluator(
                sentences1=valid_ds["sentence1_ne"],
                sentences2=valid_ds["sentence2_ne"],
                scores=valid_ds["score"],
                name="stsb_ne",
                main_similarity="cosine",
            )
        )

        return SequentialEvaluator(evaluators)
