import torch.nn as nn
from transformers import LlamaTokenizer
import os
import typing as tp
import torch
import sys
from pinyin.pinyin import G2P_PinYin


ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask

def process_line(line):
    line = line.strip()[2:]
    if(line[0]=='\'' and line[-1]=='\''):
        line = line[1:-1]
    return line

class LlamaTokenizerConditioner(nn.Module):
    def __init__(self, device: str = 'cpu', max_len = 3000, padding_idx='</s>', tokenizer_type=None,
                 pretrained="hfl/chinese-llama-2-13b"): #"hfl/chinese-llama-2-13b"
        super().__init__()
        print(f"text tokenizer from {pretrained}")
        self.text_tokenizer = LlamaTokenizer.from_pretrained(pretrained,cache_dir="huggingface_cache")
        print(f"tokenizer vocab size: {self.text_tokenizer.vocab_size}")
        self.g2p = G2P_PinYin()
        add_token_list = []
        with open(os.path.dirname(os.path.abspath(__file__))+'/vocab.yaml', 'r') as f:
            for line in f:
                if(line):
                    add_token_list.append(process_line(line))
        type_tokens = []
        with open(os.path.dirname(os.path.abspath(__file__))+'/structure.yaml', 'r') as f:
            for line in f:
                if(line):
                    type_tokens.append(process_line(line))
        if add_token_list != []:
            self.text_tokenizer.add_tokens(add_token_list, special_tokens=True)
        # voc_size = self.text_tokenizer.vocab_size
        voc_size = len(self.text_tokenizer.get_vocab()) # 加了额外token之后vocab_size似乎不会额外增加 ——cyy
        print( voc_size)
        # import pdb; pdb.set_trace()
        padding_idx = str(padding_idx)
        
        self.text_tokenizer.pad_token = padding_idx
        self.max_len = max_len
        self.padding_idx = padding_idx

        vocab = self.text_tokenizer.get_vocab()
        self.type_token_ids = [vocab[i] for i in type_tokens if i in vocab]
        struct_tokens = [padding_idx] + [i for i in add_token_list if i[0]=='[' and i[-1]==']']
        self.struct_token_ids = [vocab[i] for i in struct_tokens]
        print("type tokens: ",{self.text_tokenizer.convert_ids_to_tokens(i):i for i in self.type_token_ids},
                 "\t all structure tokens: ", {self.text_tokenizer.convert_ids_to_tokens(i):i for i in self.struct_token_ids})
        
    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        x = [self.g2p(xi) if xi is not None else "" for xi in x]
        inputs = self.text_tokenizer(x, return_tensors="pt", padding=True)
        # print(x, [self.text_tokenizer.convert_ids_to_tokens(i.tolist()) for i in inputs['input_ids']])
        # import pdb; pdb.set_trace()
        if inputs['input_ids'].shape[-1] > self.max_len:
            warnings.warn(f"Max len limit ({self.max_len}) Exceed! {x}")
            
        # print(x, inputs['input_ids'].shape)
        return inputs
    

if __name__ == "__main__":
    tokenizer = LlamaTokenizerConditioner()
    out = tokenizer.tokenize(["im ok today, and im happy now", "今天我很开心"])
    print(out)
    print(tokenizer.text_tokenizer.decode(out['input_ids'][0][:4]))
    print(tokenizer.text_tokenizer.convert_ids_to_tokens(out['input_ids'][0]))