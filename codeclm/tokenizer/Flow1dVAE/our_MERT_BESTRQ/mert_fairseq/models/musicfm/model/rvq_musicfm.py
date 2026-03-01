try:
    from .rvq import *
except:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rvq import *

try:
    from ..modules.random_quantizer import RandomProjectionQuantizer
    from ..modules.features import MelSTFT
    from ..modules.conv import Conv2dSubsampling
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.random_quantizer import RandomProjectionQuantizer
    from modules.features import MelSTFT
    from modules.conv import Conv2dSubsampling


class RVQDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.datas,inds,tot,self.sizes = load_audio_by_json(manifest_path, None, None, self.sample_rate)
        self.dataset_len = len(self.datas)

        self.reader = Read_and_PadCrop_Normalized_T(n_samples=CLIPSECS*sample_rate,sample_rate = self.sample_rate)
        self.normalize = normalize
    

    def __getitem__(self, i):
        # WORLD_SIZE = int(torch.distributed.get_world_size())
        # WORLD_RANK = int(torch.distributed.get_rank())
        # np.random.seed(1337 + self.epoch * WORLD_SIZE + WORLD_RANK + i)
        # index = random.randint(0,len(self.sizes) - 1)
        index = i
        item = None
        while item is None:
            try:
                wav = self.get_audio_by_slice(index)
                # labels = self.get_labels(index) #这个得改
                # labels = None
                # item = {"id": index, "source": wav, "label_list": labels}
                item = {"id": index, "source": wav}
            except Exception as e:
                # print(e)
                traceback.print_exc()
                print(f'skip damaged data {index}')
                index = np.random.randint(0,len(self.sizes)-1)
        return item

    def __len__(self):
        return self.dataset_len
    
    def get_audio_by_slice(self,index):
        
        wav_path = self.datas[index]['path']
        audio_info =  torchaudio.info(wav_path)
        origin_sample_rate = audio_info.sample_rate
        origin_duration = audio_info.num_frames / origin_sample_rate

        wav, *ignored = self.reader(wav_path, origin_duration,origin_sample_rate)
        wav = wav.float()
        
        # _path, slice_ptr = parse_path(wav_path) #这个应该也要改
        # original way
        # if len(slice_ptr) == 0:
        #     wav, cur_sample_rate = sf.read(_path)
        # else:
        #     assert _path.endswith(".zip")
        #     data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #     f = io.BytesIO(data)
        #     wav, cur_sample_rate = sf.read(f)
        # wav = torch.from_numpy(wav).float()
        # print(wav.shape)
        wav = wav.permute(1,0)
        wav = self.postprocess(wav, self.sample_rate) #降至单个声道，确认采样率，归一化
        # print(wav.shape)

        # wav = wav.squeeze(0)
        return wav
    
    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

class Preprocessor(nn.Module):
    def __init__(self, 
            codebook_dim=16,
            codebook_size=4096,
            hop_length=240,
            n_mels=128,
            stat_path='msd_stats.json',
        ) -> None:
        super().__init__()

        self.features=["melspec_2048"]

        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(
            n_fft=2048, hop_length=hop_length, is_db=True
        )
        

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key] # {'melspec_2048_cnt': 14282760192, 'melspec_2048_mean': 6.768444971712967}
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x
    
    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])
        return out

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x, features=self.features) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        x = self.normalize(x)
        x = self.rearrange(x) # -> {'melspec_2048': Tensor{Size([3, 750, 512]) cuda:0 f32}}
        return x['melspec_2048'].permute((0, 2, 1))

if __name__ == "__main__":
    config = dict(
        train_dataset = dict(
            manifest_path = 'music4all_sh/train.json',
            sample_rate = 24000,
            normalize = False,
        ),
        valid_dataset = dict(
            manifest_path = None,
            sample_rate = 24000,
            normalize = False,
        ),
        model = dict(
            input_dim = 128*4, 
            n_codebooks = 8, 
            codebook_size = 1024, 
            codebook_dim = 16, 
            quantizer_dropout = 0.0,
        ),
        train = dict(
            batch_size = 96,
            num_workers = 6,
            valid_interval = 10,
            save_interval = 100,
            max_updates = 5000,
            lr = 1e-4,
            device = 'cuda:1',
            # loss = 'commitment_loss * 0.25 + codebook_loss * 1.0',
            loss = 'commitment_loss * 0.25 + codebook_loss * 1.0 + (x - quantized_prompt_embeds).abs().mean()',
            preprocess = Preprocessor(),
        )
    )
    train_dataset = RVQDataset(**config['train_dataset'])
    if config['valid_dataset']['manifest_path'] is None:
        # split train and valid dataset
        from torch.utils.data import random_split
        train_dataset, valid_dataset = random_split(
            train_dataset, lengths=[len(train_dataset) - 500, 500]
        )
    else:
        valid_dataset = RVQDataset(**config['valid_dataset'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['train']['batch_size'], drop_last=True, num_workers=config['train']['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=config['train']['batch_size'], drop_last=True, num_workers=config['train']['num_workers'])
    model = ResidualVectorQuantize(**config['model'])

    device = config['train']['device']
    preprocess = config['train']['preprocess'].to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    cur_updates = 0
    is_running = True
    result = {}
    from tqdm import tqdm
    from tensorboardX import SummaryWriter 
    writer = SummaryWriter()
    from collections import defaultdict
    import os
    from logging import getLogger
    logger = getLogger()
            
    while is_running:
        results = defaultdict(lambda:0)
        for item in tqdm(train_dataloader, desc='train'): 
            wavs = item['source']
            optimizer.zero_grad()
            wavs = wavs.to(device)
            x = preprocess(wavs)
            model.train()
            quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = model(x)
            loss = eval(config['train']['loss'])
            loss.backward()
            optimizer.step()

            results['loss/train'] += loss.item()
            results['commitment_loss/train'] += commitment_loss.item()
            results['codebook_loss/train'] += codebook_loss.item()
            results['rvq_usage/train'] += rvq_usage.float().mean().item()

            if cur_updates % config['train']['valid_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    for item in tqdm(valid_dataloader, desc='valid'): 
                        wavs = item['source']
                        wavs = wavs.to(device)
                        x = preprocess(wavs)
                        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = model(x)
                        valid_loss = eval(config['train']['loss'])
                        
                        results['loss/valid'] += valid_loss.item()
                        results['commitment_loss/valid'] += commitment_loss.item()
                        results['codebook_loss/valid'] += codebook_loss.item()
                        results['rvq_usage/valid'] += rvq_usage.float().mean().item()

                    results['cur_updates'] = cur_updates
                    results['loss/train'] /= config['train']['valid_interval'] 
                    results['commitment_loss/train'] /= config['train']['valid_interval']
                    results['codebook_loss/train'] /= config['train']['valid_interval']
                    results['rvq_usage/train'] /= config['train']['valid_interval']

                    results['loss/valid'] /= len(valid_dataloader) 
                    results['commitment_loss/valid'] /= len(valid_dataloader)
                    results['codebook_loss/valid'] /= len(valid_dataloader)
                    results['rvq_usage/valid'] /= len(valid_dataloader)

                    print('')
                    logger.info(str(results))
                    for k,v in results.items():
                        writer.add_scalar(k, v, cur_updates)
                    
                    results.clear()

            if cur_updates % config['train']['save_interval'] == 0:
                os.makedirs(f'{writer.logdir}/ckpt/', exist_ok=True)
                logger.info(f'saving checkpoint to {writer.logdir}/ckpt/RVQ_{cur_updates}.pth')
                torch.save(model.state_dict(), f'{writer.logdir}/ckpt/RVQ_{cur_updates}.pth')

            
            if cur_updates < config['train']['max_updates']:
                cur_updates += 1
            else:
                is_running = False
                break

    # x = torch.randn(32, 120, 375)
    # quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = model(x)
    # print(quantized_prompt_embeds.shape)
    # print(codes.shape)
    # # w/o reconstruction
    # loss = commitment_loss * 0.25 + codebook_loss * 1.0
    # # w/ reconstruction
    # loss = commitment_loss * 0.25 + codebook_loss * 1.0 + (x - quantized_prompt_embeds).abs().mean()
