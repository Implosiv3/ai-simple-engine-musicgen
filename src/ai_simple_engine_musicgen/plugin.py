from ai_simple_engine_musicgen.models.executor.transformers_musicgen_model_executor import TransformersMusicgenModelExecutor
from ai_simple_engine_musicgen.models.executor.registry.musicgen_model_executor_registry import MusicgenModelExecutorRegistry
from ai_simple_engine_musicgen.models.loaders.musicgen_loader import MusicgenLoader
from ai_simple_engine_musicgen.consts import MUSICGEN_MODEL_FAMILY
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class MusicgenPlugin(
    Plugin
):
    """
    The plugin to add the `musicgen` model
    functionality.

    This plugin includes:
    - `MusicgenLoader`
    - `TransformersMusicgenModelExecutor` (for
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
            .add_model_loader(MusicgenLoader())
        )

        """
        Create the specific registry for the `musicgen`
        model executor, register the specific model
        executors, and register it as a service into
        the builder.
        """
        registry = MusicgenModelExecutorRegistry()

        registry.register(
            MUSICGEN_MODEL_FAMILY,
            TransformersMusicgenModelExecutor()
        )

        (
            builder.add_service(
                MusicgenModelExecutorRegistry,
                registry
            )
        )