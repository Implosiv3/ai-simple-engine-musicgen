from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.types.audio import Audio
from abc import ABC, abstractmethod
from typing import TypeVar, Generic


TModel = TypeVar('TModel')
TInfo = TypeVar('TInfo')

class MusicgenModelExecutorAbstract(
    ModelExecutor,
    Generic[TModel],
    ABC
):

    @abstractmethod
    async def generate(
        self,
        model: LoadedModel[TModel, TInfo],
        prompt: str,
        *,
        max_new_tokens: int,
        guidance_scale: float
    ) -> Audio:
        ...