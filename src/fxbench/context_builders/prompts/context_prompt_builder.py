from __future__ import annotations

from fxbench.context_builders.prompts.base import IPromptBuilder
from fxbench.context_builders.prompts.system_prompts import (
    FORMULA_COMPLETION_SYSTEM_PROMPT,
    FORMULA_COMPLETION_USER_TEMPLATE,
)
from fxbench.context_builders.serializers import IContextBuilder
from fxbench.data_models import Problem


class ContextPromptBuilder(IPromptBuilder):
    def __init__(
        self,
        context_builder: IContextBuilder,
        system_prompt: str = FORMULA_COMPLETION_SYSTEM_PROMPT,
        user_template: str = FORMULA_COMPLETION_USER_TEMPLATE,
    ) -> None:
        self.context_builder = context_builder
        self.system_prompt = system_prompt
        self.user_template = user_template

    def build(self, problem: Problem) -> list[dict[str, str]]:
        context = self.context_builder.build(problem)
        user_content = self.user_template.format(
            context=context,
            active_cell=problem.active_cell,
            prefix=problem.prefix,
        )
        system_prompt = getattr(self.context_builder, "system_prompt", None) or self.system_prompt
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @property
    def name(self) -> str:
        return self.context_builder.__class__.__name__
