from thop import profile
from thop import clever_format
import torch
from tqdm import tqdm
import time
import sys
sys.path.append('./')


def analyze_model(model, inputs):
    # model size
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num trainable parameters: {} M".format(num_trainable_parameters/1000./1000.))

    # computation cost
    with torch.no_grad():
        model.eval()
        macs, params = profile(model, inputs=inputs)
        macs, params = clever_format([macs, params], "%.3f")
        print("Macs: {}, Params: {}".format(macs, params))

    run_times = 50
    # eval forward 100 times
    with torch.no_grad():
        model = model.eval().to('cuda')
        inputs = [i.to('cuda') if isinstance(i, torch.Tensor) else i for i in inputs]
        model.init_device_dtype(inputs[0].device, inputs[0].dtype)
        st = time.time()
        for i in tqdm(range(run_times)):
            _ = model(*inputs)
        et = time.time()
        print("Eval forward : {:.03f} secs/per iter".format((et-st)/float(run_times)))

    # train backward 100 times
    model = model.train().to('cuda')
    inputs = [i.to('cuda') if isinstance(i, torch.Tensor) else i for i in inputs]
    model.init_device_dtype(inputs[0].device, inputs[0].dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    st = time.time()
    for i in tqdm(range(run_times)):
        inputs = [torch.rand_like(i) if isinstance(i, torch.cuda.FloatTensor) else i for i in inputs]
        out = model(*inputs)
        optimizer.zero_grad()
        out.mean().backward()
        optimizer.step()
    et = time.time()
    print("Train forward : {:.03f} secs/per iter".format((et-st)/float(run_times)))

def fetch_model_v3_transformer():
    # num params: 326M
    # macs (uncorrect): 261G/iter
    # infer: 0.32s/iter
    # train: 2.54s/iter
    from models_transformercond_winorm_ch16_everything_512 import PromptCondAudioDiffusion
    model = PromptCondAudioDiffusion( \
        "configs/scheduler/stable_diffusion_2.1_largenoise.json", \
        None, \
        "configs/models/transformer2D.json"
    )
    inputs = [
        torch.rand(1,16,1024*3//8,32), 
        torch.rand(1,7,512), 
        torch.tensor([1,]), 
        torch.tensor([0,]), 
        False,
    ]
    return model, inputs

def fetch_model_v3_unet():
    # num params: 310M
    # infer: 0.10s/iter
    # train: 0.70s/iter
    from models_musicldm_winorm_ch16_everything_sepnorm import PromptCondAudioDiffusion
    model = PromptCondAudioDiffusion( \
        "configs/scheduler/stable_diffusion_2.1_largenoise.json", \
        None, \
        "configs/diffusion_clapcond_model_config_ch16_everything.json"
    )
    inputs = [
        torch.rand(1,16,1024*3//8,32), 
        torch.rand(1,7,512), 
        torch.tensor([1,]), 
        torch.tensor([0,]), 
        False,
    ]
    return model, inputs

if __name__=="__main__":
    model, inputs = fetch_model_v3_transformer()
    # model, inputs = fetch_model_v3_unet()
    analyze_model(model, inputs)
