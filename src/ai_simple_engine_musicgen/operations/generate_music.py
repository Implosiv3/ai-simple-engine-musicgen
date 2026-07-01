from ai_simple_engine_musicgen.models.executor.registry.musicgen_model_executor_registry import MusicgenModelExecutorRegistry
from ai_simple_engine.types.data_type.base import AUDIO, LOADED_MODEL, STRING
from ai_simple_engine.graph.operation.base import Operation
from ai_simple_engine.graph.input import Input
from ai_simple_engine.graph.output import Output


class GenerateMusic(
    Operation
):

    model: 'LoadedModel[LoadedMusicgenModel]' = Input(LOADED_MODEL)
    """
    The `LoadedModel` tha will enter as input.
    """
    prompt = Input(STRING)
    # TODO: Transform this into duration and convert it
    max_new_tokens = Input(
        int,
        default = 256
    )
    guidance_scale = Input(
        float,
        default = 3.0
    )
    audio = Output(AUDIO)

    async def execute(
        self,
        context
    ):
        """
        TODO: We will implement an OperationFactory to
        make this a 'self._runner' that is automatically
        registered.
        """
        registry = context.services.get(MusicgenModelExecutorRegistry)

        runner = registry.resolve(self.model)

        audio = await runner.generate(
            model = self.model.instance,
            prompt = self.prompt,
            max_new_tokens = self.max_new_tokens,
            guidance_scale = self.guidance_scale
        )

        return {
            'audio': audio
        }