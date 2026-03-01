import hydra
import librosa
import torch
import yaml
from prodict import Prodict
import torchaudio

from musiclm_pytorch import AudioSpectrogramTransformerPretrained, TextTransformerPretrained, MuLaN, MuLaNEmbedder
from omegaconf import DictConfig
import os

def get_pretrained_config(root, name):
    if root is None:
        return name
    path = os.path.join(root, name)
    #获取snapshots目录下的目录
    config_dir = os.path.join(path, 'snapshots')
    config_files = os.listdir(config_dir)
    assert len(config_files) == 1
    config_path = os.path.join(config_dir, config_files[0])
    return config_path
        
def create_MuLaN_from_config(config: DictConfig):
    """
    Create a MuLaN model from a configuration file.
    """
    pretraind_root = config.model.pretraind_root

    audio_model_name = get_pretrained_config(pretraind_root, config.model.audio_model.name)
    audio_transformer = AudioSpectrogramTransformerPretrained(
        model_name = audio_model_name, 
        model_dim = config.model.audio_model.model_dim,
        use_layer_idx = config.model.audio_model.use_layer_idx,
        **config.model.audio_transformer
    )
    text_model_name = get_pretrained_config(pretraind_root, config.model.text_model.name)
    text_transformer = TextTransformerPretrained(
        model_name = text_model_name, 
        **config.model.text_transformer
    )

    mulan = MuLaN(
        audio_transformer = audio_transformer,
        text_transformer = text_transformer,
        **config.model.mulan
    )

    return mulan


def create_CLAP_model( model_kwargs = {}, ckpt_path = None ):
    from musiclm_pytorch import SoftmaxContrastiveLearning
    import laion_clap
    
    from torch import nn
    import torch
    from torchaudio.functional import resample

    import numpy as np

    from functools import partial

    # quantization
    def int16_to_float32(x):
        return (x / 32767.0).float()

    def float32_to_int16(x):
        x = torch.clip(x, min=-1., max=1.)
        return (x * 32767.).int()

    model = laion_clap.CLAP_Module(enable_fusion=False, **model_kwargs)
    if ckpt_path is not None:
        model.load_ckpt(ckpt_path)
    else:
        model.load_ckpt()

    class CLAP_Model(nn.Module):
        def __init__(self, model, sr = 24000, decoupled_contrastive_learning = True):
            super().__init__()
            self.model = model
            self.model.eval()
            self.orig_sr = sr

            klass = partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning) 
            self.contrast = klass() 

        
        def forward(self, wavs, raw_texts):
            with torch.no_grad():
                wavs = int16_to_float32(float32_to_int16(resample(wavs, self.orig_sr, 48000)))
                audio_latents = self.model.get_audio_embedding_from_data(x = wavs, use_tensor=True).float()
                text_latents = model.get_text_embedding(raw_texts, use_tensor=True)
            cl_loss = self.contrast(audio_latents, text_latents)
            return cl_loss
    
    clap = CLAP_Model(model)
    return clap

def get_mulan(config):
    with open(config, "r") as stream:
        mulan_config = yaml.safe_load(stream)
        mulan_config = Prodict.from_dict(mulan_config)
    ckpt_path = mulan_config.checkpoint_path
    mulan = create_MuLaN_from_config(mulan_config)
    mulan_embedder = MuLaNEmbedder(mulan, checkpoint_path = ckpt_path)
    mulan_embedder.eval()

    return mulan_embedder

def extract_mert_embeds(mulan_embd_extractor, layer_num, filename):
    input_audios, fs = torchaudio.load(filename)
    mulan_sr = 24000
    if(fs!=mulan_sr):
        input_audios = torchaudio.functional.resample(input_audios, fs, mulan_sr)
        fs = mulan_sr
    # print(input_audios.shape)
    inputs = mulan_embd_extractor.mulan.audio.processor(input_audios, sampling_rate=mulan_embd_extractor.mulan.audio.sr, return_tensors="pt")
    input_values = inputs['input_values'].squeeze(0).to(input_audios.device, dtype = input_audios.dtype)
    prompt_embeds = mulan_embd_extractor.mulan.audio.model(input_values, output_hidden_states=True).hidden_states[layer_num] # batch_size, Time steps, 1024 feature_dim
    return prompt_embeds
