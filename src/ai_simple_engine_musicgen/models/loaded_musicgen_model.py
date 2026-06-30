from transformers import AutoProcessor, MusicgenForConditionalGeneration
from dataclasses import dataclass


@dataclass(frozen = True)
class LoadedMusicGenModel:

    processor: AutoProcessor
    network: MusicgenForConditionalGeneration