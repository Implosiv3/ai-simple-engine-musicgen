from ai_simple_engine_musicgen.models.loaded_musicgen_model import LoadedMusicGenModel
from ai_simple_engine_musicgen.models.executor.abstract import MusicGenModelExecutorAbstract
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.registry.base import ModelExecutorRegistry


class MusicGenModelExecutorRegistry(
    ModelExecutorRegistry[
        LoadedModel[LoadedMusicGenModel],
        MusicGenModelExecutorAbstract,
        str
    ]
):

    def key_for(
        self,
        model: LoadedModel[LoadedMusicGenModel]
    ) -> str:
        return model.installed_model.family
    

