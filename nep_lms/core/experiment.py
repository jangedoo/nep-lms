import abc
from nep_lms.core.variant import Variant, VariantConfig, VariantRegistry
from pathlib import Path
from typing import Callable
import copy
import logging


class BaseExperiment(abc.ABC):
    def __init__(self, name: str, root_dir: Path | str):
        self.name = name
        self.root_dir = Path(root_dir)
        self.registry = VariantRegistry()
        self.logger = logging.getLogger(__name__)

    def variant(self, name: str | None = None, parent: str | None = None):
        """Decorator to create a variant from a function that returns a VariantConfig."""

        def decorator(
            fn: Callable[[], VariantConfig] | Callable[[VariantConfig], VariantConfig],
        ):
            variant_name = name or fn.__name__
            if parent:
                # when inherited, we want to deep-copy the parent config
                def config_getter() -> VariantConfig:
                    assert parent is not None, "Parent variant must be specified"
                    base_cfg = copy.deepcopy(self.registry.get(parent).config_getter())
                    return fn(base_cfg)  # type: ignore

            else:
                # new standalone variant — just call fn with no args
                def config_getter() -> VariantConfig:
                    return fn()  # type: ignore

            new_variant = Variant(name=variant_name, config_getter=config_getter)
            self.registry.register(variant_name, new_variant)
            return new_variant

        return decorator

    @abc.abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
