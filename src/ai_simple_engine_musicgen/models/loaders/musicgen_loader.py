from ai_simple_engine_musicgen.models.loaded_musicgen_model import LoadedMusicGenModel
from ai_simple_engine_musicgen.consts import MUSICGEN_MODEL_FAMILY_NAME
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.loaders.abstract import ModelLoader
from ai_simple_engine.models.installed_model import InstalledModel
from ai_simple_engine.device.base import Device
from transformers import AutoProcessor, MusicgenForConditionalGeneration


class MusicGenLoader(
    ModelLoader
):
    
    @property
    def family(
        self
    ) -> str:
        return MUSICGEN_MODEL_FAMILY_NAME

    async def load(
        self,
        installed_model: InstalledModel,
        *,
        device: Device
    ) -> LoadedModel[LoadedMusicGenModel]:
        processor = AutoProcessor.from_pretrained(installed_model.path)
        network = MusicgenForConditionalGeneration.from_pretrained(installed_model.path)
        
        network.to(device.identifier)

        return LoadedModel(
            installed_model = installed_model,
            instance = LoadedMusicGenModel(
                processor = processor,
                network = network
            )
        )

    async def unload(
        self,
        model: LoadedModel
    ) -> None:
        del model