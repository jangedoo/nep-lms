import abc
import logging

import datasets
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator


class BaseEmbeddingExperiment:
    name = "embedding"
    ds_name_to_loader = {}

    def __init__(self):
        self._datasets: dict[str, datasets.DatasetDict] = {}
        self._evaluator: SentenceEvaluator = None # type: ignore
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dataset(self, name: str):
        if name not in self._datasets:
            loader = self.ds_name_to_loader[name]
            self._datasets[name] = loader()
        return self._datasets[name]

    @abc.abstractmethod
    def _get_evaluator(self, max_rows: int | None = None):
        raise NotImplementedError()

    @property
    def evaluator(self) -> SentenceEvaluator:
        if self._evaluator is None:
            self._evaluator = self._get_evaluator()
        return self._evaluator

    def evaluate(self, model: SentenceTransformer, max_rows: int | None = None):
        if max_rows is not None:
            return self._get_evaluator(max_rows=max_rows)(model=model)

        if self._evaluator is None:
            self._evaluator = self._get_evaluator()

        return self._evaluator(model=model)

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
            self.logger.info("Either pass all_model_metrics_df or run .compare method")
            return

        main_metrics = main_metrics or self.main_metrics
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
