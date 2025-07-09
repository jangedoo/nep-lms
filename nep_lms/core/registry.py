from typing import Generic, Hashable, TypeVar


V = TypeVar("V")
K = TypeVar("K", bound=Hashable)


class BaseRegistry(Generic[K, V]):
    def __init__(self):
        self._registry: dict[K, V] = {}

    def register(self, name: K, value: V):
        self._registry[name] = value

    def get(self, name: K) -> V:
        return self._registry[name]

    def get_names(self) -> list[K]:
        return list(self._registry.keys())
