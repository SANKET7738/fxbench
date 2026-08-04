from __future__ import annotations

from abc import ABC, abstractmethod

from fxbench.data_models import Problem


class IPromptBuilder(ABC):
    @abstractmethod
    def build(self, problem: Problem) -> list[dict[str, str]]:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
