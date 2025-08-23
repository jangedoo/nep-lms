import datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
    SentenceEvaluator,
    TranslationEvaluator,
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
)
import pandas as pd
import logging


def split_ds_to_train_test_valid(ds: datasets.Dataset, seed: int = 10):
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


def nepali_news_loader():
    ds = datasets.load_dataset("jangedoo/nepalinews", split="train")
    return split_ds_to_train_test_valid(ds=ds, seed=10)


def en_ne_parallel_corpus_loader():
    # this ds already has train, test, valid splits
    ds = datasets.load_dataset("jangedoo/en_ne_parallel_corpus")
    return ds


def paraphrase_loader():
    # this ds already has train, test, valid splits
    ds = datasets.load_dataset("jangedoo/paraphrase-nepali")
    return ds


def nepali_triplets_loader():
    ds = datasets.load_dataset("jangedoo/nepali-triplets")
    # this ds already has train, test, valid splits
    return ds


def nepali_stsb_loader():
    ds = datasets.load_dataset("jangedoo/stsb_nepali")
    return ds


def create_ir_evaluator_from_parallel_corpus(
    ds: datasets.Dataset, query_col: str, doc_col: str, evaluator_name: str = ""
):
    queries = {str(i): text for i, text in enumerate(ds[query_col])}
    corpus = {str(i): text for i, text in enumerate(ds[doc_col])}
    # since this is a parallel corpus, the relevant doc for a query has the same "id"
    relevant_docs = {qid: set([qid]) for qid in queries.keys()}
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=evaluator_name,
        precision_recall_at_k=[10, 50],
        ndcg_at_k=[10],
        accuracy_at_k=[10],
        main_score_function="cosine",
    )
    return ir_evaluator


class EmbeddingExperiment:
    def __init__(self):
        self.name = "embedding"
        self.ds_name_to_loader = {
            "nepali_news": nepali_news_loader,
            "en_ne_parallel_corpus": en_ne_parallel_corpus_loader,
            "paraphrase": paraphrase_loader,
            "nepali_triplets": nepali_triplets_loader,
            "nepali_stsb": nepali_stsb_loader,
        }
        self._datasets: dict[str, datasets.DatasetDict] = {}
        self._evaluator: SentenceEvaluator = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dataset(self, name: str):
        if name not in self._datasets:
            loader = self.ds_name_to_loader[name]
            self._datasets[name] = loader()
        return self._datasets[name]

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
        valid_ds = self.nepali_news_ds["valid"]
        if max_rows:
            valid_ds = valid_ds.select(range(max_rows))
        multi_lang_ir_evaluator = create_ir_evaluator_from_parallel_corpus(
            ds=valid_ds,
            query_col="title",
            doc_col="excerpt",
            evaluator_name="multi_lang_ir",
        )
        en_only_ir_evalautor = create_ir_evaluator_from_parallel_corpus(
            ds=valid_ds.filter(lambda row: row["language"] == "en"),
            query_col="title",
            doc_col="excerpt",
            evaluator_name="en_ir",
        )

        ne_only_ir_evalautor = create_ir_evaluator_from_parallel_corpus(
            ds=valid_ds.filter(lambda row: row["language"] == "ne"),
            query_col="title",
            doc_col="excerpt",
            evaluator_name="ne_ir",
        )

        # translation evaluator
        valid_ds = self.en_ne_parallel_corpus_ds["valid"]
        if max_rows:
            valid_ds = valid_ds.select(range(max_rows))
        translation_evaluator = TranslationEvaluator(
            source_sentences=valid_ds["title"],
            target_sentences=valid_ds["translation"],
            name="translation",
        )

        # triplets evaluator
        valid_ds = self.nepali_triplets_ds["valid"]
        if max_rows:
            valid_ds = valid_ds.select(range(max_rows))
        triplets_evaluator = TripletEvaluator(
            anchors=valid_ds["sentence"],
            positives=valid_ds["positive_sentence"],
            negatives=valid_ds["negative_sentence"],
            name="nepali_triplets",
            main_similarity_function="cosine",
            margin=0.1,
        )

        # stsb evaluator
        valid_ds = self.nepali_stsb_ds["valid"]
        if max_rows:
            valid_ds = valid_ds.select(range(max_rows))
        stsb_en_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=valid_ds["sentence1"],
            sentences2=valid_ds["sentence2"],
            scores=valid_ds["score"],
            name="stsb_en",
            main_similarity="cosine",
        )
        stsb_ne_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=valid_ds["sentence1_ne"],
            sentences2=valid_ds["sentence2_ne"],
            scores=valid_ds["score"],
            name="stsb_ne",
            main_similarity="cosine",
        )

        evaluator = SequentialEvaluator(
            [
                multi_lang_ir_evaluator,
                en_only_ir_evalautor,
                ne_only_ir_evalautor,
                translation_evaluator,
                triplets_evaluator,
                stsb_en_evaluator,
                stsb_ne_evaluator,
            ]
        )
        return evaluator

    @property
    def evaluator(self) -> SentenceEvaluator:
        if self._evaluator is None:
            self._evaluator = self._get_evaluator()
        return self._evaluator

    def evaluate(self, model: SentenceTransformer, max_rows: int | None = None):
        if self._evaluator is None:
            self._evaluator = self._get_evaluator(max_rows=max_rows)

        results = self._evaluator(model=model)
        return results

    def compare(
        self,
        models: dict[str, SentenceTransformer] | list[tuple[str, SentenceTransformer]],
    ):
        if isinstance(models, list):
            models = dict(models)

        dfs = []
        for model_name, model in models.items():
            model_metrics = self.evaluate(model=model)
            df = (
                pd.DataFrame.from_dict(model_metrics, orient="index")
                .reset_index()
                .rename(columns={"index": "metric", 0: "value"})
            )
            df["model"] = model_name
            dfs.append(df)
        self.all_model_metrics_df = pd.concat(dfs)
        return self.all_model_metrics_df

    def plot_model_comparison(
        self,
        all_model_metrics_df: pd.DataFrame | None = None,
        main_metrics: list[str] | None = None,
    ):
        import lets_plot as lp

        lp.LetsPlot.setup_html()
        all_model_metrics_df = (
            all_model_metrics_df
            if all_model_metrics_df is not None
            else self.all_model_metrics_df
        )
        if all_model_metrics_df is None:
            self.logger.info(f"Either pass all_model_metrics_df or run .compare method")
            return

        main_metrics = main_metrics or [
            "translation_mean_accuracy",
            "en_ir_cosine_recall@10",
            "ne_ir_cosine_recall@10",
            "multi_lang_ir_cosine_recall@10",
            "nepali_triplets_cosine_accuracy",
            "stsb_en_pearson_cosine",
            "stsb_ne_pearson_cosine",
        ]
        fig = (
            lp.ggplot(
                all_model_metrics_df.query("metric in @main_metrics"),
                lp.aes("model", "value", fill="model"),
            )
            + lp.geom_bar(stat="identity")
            + lp.geom_text(lp.aes(label="value"), label_format=".2f")
            + lp.facet_wrap("metric", ncol=2)
            + lp.theme_minimal2()
        )
        return fig
