from ai_simple_engine_musicgen.models.loaded_musicgen_model import LoadedMusicGenModel
from ai_simple_engine_musicgen.models.executor.abstract import MusicGenModelExecutorAbstract
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.registry.family_model_executor_registry import FamilyModelExecutorRegistry


class MusicGenModelExecutorRegistry(
    FamilyModelExecutorRegistry[
        LoadedModel[LoadedMusicGenModel],
        MusicGenModelExecutorAbstract
    ]
):

    pass
    

