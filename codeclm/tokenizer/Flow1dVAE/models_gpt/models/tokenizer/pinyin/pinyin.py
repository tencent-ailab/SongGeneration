import re

from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from . import pinyin_dict
import torch


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if (is_chinese(char)):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean


class G2P_PinYin():

    def __init__(self):
        super(G2P_PinYin, self).__init__()
        self.pinyin_parser = Pinyin(MyConverter())

    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    # def chinese_to_phonemes(self, text):
    #     text = clean_chinese(text)
    #     phonemes = ["sil"]
    #     chars = ['[PAD]']
    #     all_pinyins = []
    #     count_phone = []
    #     count_phone.append(1)
    #     for subtext in text.split(","):
    #         if (len(subtext) == 0):
    #             continue
    #         pinyins = self.correct_pinyin_tone3(subtext)
    #         all_pinyins.append(' '.join(pinyins))
    #         sub_p, sub_c = self.get_phoneme4pinyin(pinyins)
    #         phonemes.extend(sub_p)
    #         phonemes.append(",")
    #         count_phone.extend(sub_c)
    #         count_phone.append(1)
    #         chars.append(subtext)
    #         chars.append(',')
    #     phonemes.append("sil")
    #     count_phone.append(1)
    #     chars.append('[PAD]')
    #     # char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
    #     return " ".join(phonemes), " ".join(chars), ' , '.join(all_pinyins)

    def chinese_to_phonemes(self, text):
        all_pinyins = []
        subtext = []
        for chr in text:
            if is_chinese(chr):
                subtext.append(chr)
            else:
                if subtext != []:
                    subtext = ''.join(subtext)
                    pinyins = self.correct_pinyin_tone3(subtext) 
                    pinyins = [f"<{i}>" for i in pinyins]   
                    all_pinyins.append(' '+ ' '.join(pinyins)+ ' ')
                all_pinyins.append(chr)
                subtext = []
        if subtext != []:
            subtext = ''.join(subtext)
            pinyins = self.correct_pinyin_tone3(subtext)
            pinyins = [f"<{i}>" for i in pinyins]     
            all_pinyins.append(' '+ ' '.join(pinyins)+ ' ')
        # char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return  ''.join(all_pinyins)
    
    def correct_pinyin_tone3(self, text):
        pinyin_list = [
            p[0]
            for p in self.pinyin_parser.pinyin(text,
                                               style=Style.TONE3,
                                               strict=False,
                                               neutral_tone_with_five=True)
        ]
        if len(pinyin_list) >= 2:
            for i in range(1, len(pinyin_list)):
                try:
                    if re.findall(r'\d',
                                  pinyin_list[i - 1])[0] == '3' and re.findall(
                                      r'\d', pinyin_list[i])[0] == '3':
                        pinyin_list[i - 1] = pinyin_list[i - 1].replace(
                            '3', '2')
                except IndexError:
                    pass
        return pinyin_list

    # def expand_for_phone(self, char_embeds, length):  # length of phones for char
    #     if(char_embeds.size(0) > len(length)):
    #         print(char_embeds.shape, len(length))
    #         char_embeds = char_embeds[0:len(length),:]
    #     elif(char_embeds.size(0) < len(length)):
    #         print(char_embeds.shape, len(length))
    #         length = length[0:char_embeds.size(0)]
    #     expand_vecs = list()
    #     for vec, leng in zip(char_embeds, length):
    #         vec = vec.expand(leng, -1)
    #         expand_vecs.append(vec)
    #     expand_embeds = torch.cat(expand_vecs, 0)
    #     assert expand_embeds.size(0) == sum(length)
    #     return expand_embeds.numpy()

    def __call__(self, text):
        return self.chinese_to_phonemes(text)