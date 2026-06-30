from ai_simple_engine_musicgen.models.loaded_musicgen_model import LoadedMusicGenModel
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.types.audio import Audio
from abc import ABC, abstractmethod


class MusicGenModelExecutorAbstract(
    ModelExecutor,
    ABC
):

    @abstractmethod
    async def generate(
        self,
        model: LoadedModel[LoadedMusicGenModel],
        prompt: str,
        *,
        max_new_tokens: int,
        guidance_scale: float
    ) -> Audio:
        ...