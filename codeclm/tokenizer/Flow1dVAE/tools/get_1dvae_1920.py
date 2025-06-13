import torch
from tqdm import tqdm
import torchaudio
from third_party.stable_audio_tools.stable_audio_tools.models.autoencoders import create_autoencoder_from_config
import numpy as np
import os
import json

def get_model(model_config, path):
    with open(model_config) as f:
        model_config = json.load(f)
    state_dict = torch.load(path)
    model = create_autoencoder_from_config(model_config)
    model.load_state_dict(state_dict['state_dict'])
    return model