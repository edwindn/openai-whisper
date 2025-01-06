import torch
import torch.nn as nn
from audio import get_spectrogram
from model import Whisper, WhisperParams

whisper = Whisper(WhisperParams())
