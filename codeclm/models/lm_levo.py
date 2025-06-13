
import torch
import math
import random
import torch.nn as nn
import typing as tp
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from codeclm.models.levo import CausalLM, LlamaConfig
from codeclm.modules.streaming import StreamingModule
from codeclm.modules.conditioners import (
    ConditioningAttributes,
    AudioCondition,
    ConditionType,
    ConditionerProvider,
    ConditionFuser,
    ClassifierFreeGuidanceDropoutInference,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
)
from codeclm.utils.utils import create_norm_fn, init_layer, sample_top_k, sample_top_p, multinomial
from codeclm.modules.pattern import CodebooksPatternProvider
ConditionTensors = tp.Dict[str, ConditionType]

@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class LmModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (ConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        code_depth (int): Number of parallel streams to model.
        code_size (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, 
                 pattern_provider: CodebooksPatternProvider, 
                 condition_provider: ConditionerProvider,
                 fuser: ConditionFuser, 
                 code_depth: int = 8, 
                 code_size: int = 1024, 
                 dim: int = 128,
                 intermediate_size: int = 4096,
                 num_heads: int = 8,
                 norm: str = 'layer_norm', norm_first: bool = False,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, 
                 num_layers=16,
                 max_position_embeddings: int = 8196,
                 max_position_embeddings_sub: int = 10000,
                 rope_theta: float = 100000.0,
                 rope_theta_sub: float = 500000.0,
                 num_layers_sub: int = 12,
                 cfg = None,
                 use_flash_attn_2: bool = True,
                 **kwargs):
        super().__init__()

        self.cfg_coef = cfg_coef
    
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout,seed=random.randint(0, 9999))
        self.att_dropout = AttributeDropout(p=attribute_dropout,seed=random.randint(0, 9999))
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.code_size = code_size + 1   # + EOS
        input_emb_dim = code_size + 2   # EOP
        self.code_depth = code_depth
        self.dim = dim
        self.cfg = cfg
        self.pattern_provider = pattern_provider
        self.emb = nn.ModuleList([nn.Embedding(input_emb_dim, dim)])
                
        model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size = intermediate_size,
            num_attention_heads = num_heads,
            num_hidden_layers = num_layers,
            num_key_value_heads = num_heads,
            vocab_size = self.code_size,
            use_cache=False,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps= 1e-5,
            rope_theta= rope_theta,
            _flash_attn_2_enabled=use_flash_attn_2,
        )

        self.transformer = CausalLM(model_cfg)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.layer2_emb = nn.ModuleList([nn.Embedding(input_emb_dim, dim)
                                  for _ in range(self.code_depth)])
        sub_model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size = intermediate_size,
            num_attention_heads = num_heads,
            num_hidden_layers = num_layers_sub,
            num_key_value_heads = num_heads,
            vocab_size = self.code_size,
            use_cache=False,
            max_position_embeddings=max_position_embeddings_sub,
            rms_norm_eps= 1e-5,
            rope_theta= rope_theta_sub,
            _flash_attn_2_enabled=use_flash_attn_2,
        )

        self.transformer2 = CausalLM(sub_model_cfg)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        # enable EOS prediction
        if code_depth > 1:
            self.linears = nn.ModuleList([nn.Linear(dim, self.code_size, bias=False) 
                                        for _ in range(code_depth - 1)])
        
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

        self.reset_streaming()
        
    def _init_weights(self, weight_init: tp.Optional[str], 
                      depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    
    @property
    def special_token_id(self) -> int:
        return self.code_size   # 10001
    
    @property
    def eos_token_id(self) -> int:
        return self.code_size-1 # 10000
    
    @torch.no_grad()
    def prepare_condition_tensors(self,
                                   batch_size = 1,
                                   text: tp.Optional[tp.List[str]] = None, 
                                   descriptions: tp.Optional[tp.List[str]] = None, 
                                   audio_qt_emb: tp.Optional[tp.List[torch.Tensor]] = None,
                                   prepare_null_condition = False,
                                   ):
        if self.training:
            attributes = []
            for i in range(batch_size):
                attr = ConditioningAttributes()
                if 'description' in self.condition_provider.conditioners:
                    attr["text"]["description"] = ""
                    if text is not None:
                        attr["text"]["description"] = text[i]
                if 'prompt_audio' in self.condition_provider.conditioners:
                    mask = (audio_qt_emb[[i], :, 0] == 16385).bool().unsqueeze(-1)
                    audio_qt_seq = torch.cat([torch.full_like(audio_qt_emb[i][None][:,:,0], self.eos_token_id).unsqueeze(-1), audio_qt_emb[i][None]], dim=-1)    
                    mask = mask.repeat(1, 1, audio_qt_seq.shape[-1])
                    audio_qt_seq[mask] = 16385
                    attr["audio"]['prompt_audio'] = AudioCondition(
                        wav=audio_qt_seq.long(),
                        length=torch.Tensor([audio_qt_seq.shape[-1]]).long(),
                        sample_rate=[self.cfg.sample_rate],)
                if 'type_info' in self.condition_provider.conditioners:
                    attr["text"]["type_info"] = ""
                    if descriptions is not None:
                        attr["text"]["type_info"] = descriptions[i]
                attributes.append(attr)
            attributes = self.cfg_dropout(attributes)   # drop ALL conditions
            attributes = self.att_dropout(attributes)   # selectively drop some attributes (text, wav, or more fine-grained)
            tokenized = self.condition_provider.tokenize(attributes)
            condition_tensors = self.condition_provider(tokenized)
        else:
            conditions = []
            for i in range(batch_size):
                attr = ConditioningAttributes()
                if 'description' in self.condition_provider.conditioners:
                    attr["text"]["description"] = ""
                    if text is not None:
                        attr["text"]["description"] = text[i]
                if 'prompt_audio' in self.condition_provider.conditioners:
                    mask = (audio_qt_emb[[i], :, 0] == 16385).bool().unsqueeze(-1)
                    audio_qt_seq = torch.cat([torch.full_like(audio_qt_emb[i][None][:,:,0], self.eos_token_id).unsqueeze(-1), audio_qt_emb[i][None]], dim=-1)    
                    mask = mask.repeat(1, 1, audio_qt_seq.shape[-1])
                    audio_qt_seq[mask] = 16385
                    attr["audio"]['prompt_audio'] = AudioCondition(
                        wav=audio_qt_seq.long().cuda(), 
                        length=torch.Tensor([audio_qt_seq.shape[-1]]).long(),
                        sample_rate=[self.cfg.sample_rate],)
                if 'type_info' in self.condition_provider.conditioners:
                    attr["text"]["type_info"] = ""
                    if descriptions is not None:
                        attr["text"]["type_info"] = descriptions[i]
                conditions.append(attr)
                print("conditions", conditions)
            if prepare_null_condition:
                cfg_inference = ClassifierFreeGuidanceDropoutInference() 
                null_conditions = cfg_inference(conditions, condition_types=["audio", "text"], 
                                                customized=None)
                conditions = conditions + null_conditions
            tokenized_conditions = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(tokenized_conditions)
        return condition_tensors
        
    def forward(self, 
                sequence: torch.Tensor,
                condition_tensors: ConditionTensors) -> torch.Tensor:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: Logits.
        """

        # import pdb; pdb.set_trace()
        B, K, S = sequence.shape
        assert K == self.code_depth, "Sequence shape must match the specified number of codebooks"
        input_1 = self.emb[0](sequence[:, 0])
        input_2 = sum([self.layer2_emb[k](sequence[:, k]) for k in range(1, K)])
        fused_input1, fused_input2 = self.fuser(input_1, input_2, condition_tensors)
        output = self.transformer(inputs_embeds=fused_input1, 
                                  use_cache=self._is_streaming, 
                                  past_key_values=self._streaming_state.get('past_key_values_1', None))
        if self._is_streaming:
            self._streaming_state['past_key_values_1'] = output.past_key_values
        logits = output.logits # [B, S, card]
        logits = logits.unsqueeze(1) # [B, 1, S, card]
             
        # if self.out_norm:
        #     out = self.out_norm(out.to(self.out_norm.weight.data.dtype))
        if K > 1:
            fused_input2 = torch.cat([fused_input2, output.hidden_states], dim=-1)
            fused_input2 = self.mlp(fused_input2)
            output2 = self.transformer2(inputs_embeds=fused_input2, 
                                           use_cache=self._is_streaming, 
                                           past_key_values=self._streaming_state.get('past_key_values_2', None))
            if self._is_streaming:
                self._streaming_state['past_key_values_2'] = output2.past_key_values
            
            res_logits = torch.stack([self.linears[k](output2.hidden_states) for k in range(K - 1)], dim=1)  # [B, K, S, card] # [B, K, S, card]
            logits = torch.cat([logits, res_logits], dim=1)  # [B, K, S, card]
        
        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:, :]

        return logits  # [B, K, S, card]

    def compute_predictions(self, 
                            codes: torch.Tensor,
                            condition_tensors: tp.Optional[ConditionTensors] = None,
                            **kwargs,
                            ):  # this function is called during training
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=False
        )
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=False
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        
        return LMOutput(logits, logits_mask)   
    
    @torch.no_grad()
    def generate(self, #
                #  conditions: tp.List[ConditioningAttributes] = [],
                 texts = None,
                 descriptions = None,
                 audio_qt_embs = None,
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 check: bool = False,        
                 record_tokens: bool = True,
                 record_window: int = 150
                 ) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, K, T].
            conditions_tensors (list of ConditioningAttributes, optional): List of conditions.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coeff (float, optional): Classifier-free guidance coefficient.
            check (bool): Whether to apply further checks on generated sequence.
            callback (Callback, optional): Callback function to report generation progress.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        
        # 1) Check input shapes are consistent 
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif texts:            
            possible_num_samples.append(len(texts))
        elif audio_qt_embs:            
            possible_num_samples.append(len(audio_qt_embs))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]
        condition_tensors = self.prepare_condition_tensors(batch_size=1, text=texts, descriptions=descriptions, audio_qt_emb=audio_qt_embs, prepare_null_condition=True)
        # 3) Prepare token pool
        record_token_pool = None
        if record_tokens:
            record_token_pool = []
            
        # 4) set up startoff patterns
        start_offset = 0
        assert start_offset < max_gen_len, f"{start_offset}, {max_gen_len}"
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1
        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        B = num_samples
        gen_codes = torch.full((B, self.code_depth, max_gen_len), 
                               unknown_token, dtype=torch.long, device=device)
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        output_codes = torch.full_like(gen_sequence, self.code_size)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None
        is_end = torch.zeros((B, self.code_depth, 1)).bool().to(device)
        ignore_tokens = audio_qt_embs[0][0]
        ignore_tokens = ignore_tokens[ignore_tokens < 16384]
        # 5) auto-regressive sampling
        with self.streaming():
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
            prev_offset = 0
            for offset in tqdm(range(start_offset_sequence, gen_sequence_len)):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                    # check coherence between mask and sequence
                    assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                    # should never happen as gen_sequence is filled progressively
                    assert not (curr_sequence == unknown_token).any()
                # sample next token from the model, next token shape is [B, K, 1]
                next_token = self._sample_next_token(
                    curr_sequence, condition_tensors, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, 
                    sampled_token_pool=record_token_pool[-record_window:] if record_tokens else None,
                    ignore_tokens = ignore_tokens
                    )
                # ensure the tokens that should be masked are properly set to special_token_id
                # as the model never output special_token_id
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                # 检查eos id
                next_token[is_end] = self.special_token_id
                is_end = is_end | (next_token == self.eos_token_id)
                # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                # (then mask tokens should be left as is as well, which is correct)
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1])
                
                # record sampled tokens in a window
                if record_tokens:
                    record_token_pool.append(next_token.squeeze())
                if torch.all(is_end):
                    gen_sequence = gen_sequence[..., :offset+1]
                    break
                prev_offset = offset
                
        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        max_gen_len = gen_sequence.shape[-1]
        output_codes[..., :max_gen_len] = gen_sequence
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        # assert (gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, 
        #                                 self.special_token_id)
        # ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(output_codes, special_token=unknown_token)
        # sanity checks over the returned codes and corresponding masks
        assert (out_codes != unknown_token).all()
        assert (out_mask == 1).all()
        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.code_size).all()
        return out_codes      
    
    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           condition_tensors: ConditionTensors,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           sampled_token_pool: tp.Optional[list] = None,
                           ignore_tokens: tp.Optional[torch.tensor] = torch.tensor([])) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        # import pdb; pdb.set_trace()
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        
        # Preparing for CFG, predicting both conditional and unconditional logits.
        sequence = torch.cat([sequence, sequence], dim=0)
        all_logits = model(sequence, condition_tensors=condition_tensors)
        cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]
        
        # add punishment to pre-sampled tokens
        if sampled_token_pool is not None and len(sampled_token_pool) > 0:
            sampled_token_pool = torch.stack(sampled_token_pool, -1) # [K, T]
            for q in range(self.code_depth):
                # q_count = torch.bincount(sampled_token_pool)
                q_count = torch.bincount(torch.unique(sampled_token_pool[q]))
                tmp = min(q_count.shape[-1], self.code_size - 1) 
                logits[:, q, :tmp] /= (1.1 ** q_count[:tmp])

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if(ignore_tokens is not None and len(ignore_tokens) > 0):
            logits[0][0][ignore_tokens.to(torch.int)] = float('-inf')
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token_first = sample_top_k(probs[:,[0],:], k=top_k)
                next_token_res = sample_top_k(probs[:,1:,:], k=1)
                next_token = torch.cat([next_token_first,next_token_res], dim = 1)
            else:
                next_token = multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token
