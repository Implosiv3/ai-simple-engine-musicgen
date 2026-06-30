from ai_simple_engine_musicgen.models.family.base import MUSICGEN_MODEL_FAMILY
from ai_simple_engine_musicgen.models.executor.transformers_musicgen_model_executor import TransformersMusicGenModelExecutor
from ai_simple_engine_musicgen.models.executor.registry.musicgen_model_executor_registry import MusicGenModelExecutorRegistry
from ai_simple_engine_musicgen.models.loaders.musicgen_loader import MusicGenLoader
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class MusicgenPlugin(
    Plugin
):
    """
    The plugin to add the `musicgen` model
    functionality.

    This plugin includes:
    - `MusicGenLoader`
    - `TransformersMusicGenModelExecutor` (for
    `musicgen`)

    This plugin gives you the next operations:
    - `GenerateMusic`

    This plugin needs:
    - At least one backend to download the `
    """

    def register(
        self,
        builder: EngineBuilder
    ):
        # Model loaders
        (
            builder
            .add_model_loader(MusicGenLoader())
        )

        # TODO: Is this ok (?)
        # ModelExecutor registry service
        registry = MusicGenModelExecutorRegistry()

        registry.register(
            MUSICGEN_MODEL_FAMILY,
            TransformersMusicGenModelExecutor()
        )

        (
            builder.add_service(
                MusicGenModelExecutorRegistry,
                registry
            )
        )