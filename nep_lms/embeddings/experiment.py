from nep_lms.core.experiment import BaseExperiment
from nep_lms.core.variant import Variant, VariantRegistry
from pathlib import Path


class SemanticSimilarityExperiment(BaseExperiment):
    def __init__(self, root_dir: Path | str):
        super().__init__(name="semantic_similarity", root_dir=root_dir)

    def get_variant_names(self):
        return self.registry.get_names()

    def get_variant(self, name: str):
        return self.registry.get(name)

    def run(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass
