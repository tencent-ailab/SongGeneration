import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def get_whisper_encoder():
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").model.encoder
    return processor, model.eval()

if __name__=="__main__":
    import numpy as np
    processor, model = get_whisper_encoder()
    model = model.cuda()
    
    with torch.no_grad():
        input_features = processor(np.random.rand(16000*30,), sampling_rate=16000, return_tensors="pt").input_features.cuda()
        print(input_features.shape)
        out = model(input_features.repeat(10,1,1))
        import pdb;pdb.set_trace()
        print(list(out.values())[0].shape)
