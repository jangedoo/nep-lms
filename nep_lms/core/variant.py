from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING
import datasets
from nep_lms.core.registry import BaseRegistry

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass
class VariantConfig:
    model: str | Callable[[], "SentenceTransformer"]
    loss: (
        Callable[["SentenceTransformer"], Callable]
        | Callable[["SentenceTransformer"], dict[str, Callable]]
    )
    train_dataset: datasets.Dataset | dict[str, datasets.Dataset]
    eval_dataset: datasets.Dataset | dict[str, datasets.Dataset] | None = None


class Variant:
    def __init__(self, name: str, config_getter: Callable[[], VariantConfig]):
        self.name = name
        self.config_getter = config_getter


class VariantRegistry(BaseRegistry[str, Variant]):
    pass
