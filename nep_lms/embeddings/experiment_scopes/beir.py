from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator, NanoBEIREvaluator
from nep_lms.embeddings.experiment_scopes.base import BaseEmbeddingExperiment


class BEIRNEmbeddingExperiment(BaseEmbeddingExperiment):
    name = "NanoBEIR_ne"
    main_metrics = ["NanoBEIR_mean_cosine_recall@5", "NanoBEIR_mean_cosine_ndcg@10"]
    def _get_evaluator(self, max_rows: int | None = None):
        return NanoBEIREvaluator(dataset_id="jangedoo/NanoBEIR-ne")