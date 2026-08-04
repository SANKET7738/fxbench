from __future__ import annotations

from abc import ABC

from pydantic import BaseModel, ConfigDict, Field

from fxbench.data_models.problem import Problem


class BenchmarkSpec(BaseModel):
    name: str = ""
    version: int = 1


class Benchmark(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problems: list[Problem] = Field(default_factory=list)
    spec: BenchmarkSpec = Field(default_factory=BenchmarkSpec)

    def model_post_init(self, __context: object) -> None:
        if type(self) is Benchmark:
            raise TypeError("Cannot instantiate abstract Benchmark directly; subclass it.")
        if not self.spec.name:
            self.spec.name = self.__class__.__name__
