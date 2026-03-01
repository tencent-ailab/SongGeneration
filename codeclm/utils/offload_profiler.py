import torch
from torch.func import functional_call
import queue
import threading
from typing import Dict, List, Any
import omegaconf
from pydantic import BaseModel, validator
from typing import Optional
from functools import wraps

def _callable_once(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        method_called_flag = f"_called_once_{func.__name__}"
        if getattr(self, method_called_flag, False):
            raise RuntimeError(f"{func.__name__} can only be called once.")
        setattr(self, method_called_flag, True)
        return func(self, *args, **kwargs)
    return wrapper

class OffloadCleanCacheWrapperParam(BaseModel):
    module: Any 
    method_name: str
    diff_mem_gb_thre: float

class OffloadParam(BaseModel):
    offload_module: Any 
    cpu_mem_gb: float
    pre_copy_step: Optional[int] = None
    clean_cache_after_forward: Optional[bool] = None
    dtype: Optional[str] = None 
    offload_layer_dict: Dict[str, int] = {}
    ignore_layer_list: List[str] = []
    clean_cache_wrapper: Optional[OffloadCleanCacheWrapperParam] = None
    debug: Optional[bool] = None

    @validator('dtype')
    def parse_dtype(cls, value):
        if value is None:
            return None
        dtype_map = {
            'torch.float16': torch.float16,
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.int64': torch.int64,
        }
        if value not in dtype_map:
            raise ValueError(f"Unsupported dtype: {value}")
        return dtype_map[value]
    
    def init_param_dict(self):
        param_dict = {}
        param_dict['cpu_mem_gb'] = self.cpu_mem_gb
        if self.pre_copy_step is not None:
            param_dict['pre_copy_step'] = self.pre_copy_step
        if self.clean_cache_after_forward is not None:
            param_dict['clean_cache_after_forward'] = self.clean_cache_after_forward
        if self.debug is not None:
            param_dict['debug'] = self.debug
        
        return param_dict
        
    def offload_layer_param_dict(self):
        param_dict = {}
        param_dict['module'] = self.offload_module
        param_dict['offload_layer_dict'] = self.offload_layer_dict
        param_dict['ignore_layer_list'] = self.ignore_layer_list
        param_dict['dtype'] = self.dtype

        return param_dict
    
    def clean_cache_param_dict(self):
        param_dict = {}
        if self.clean_cache_wrapper is not None:
            param_dict['module'] = self.clean_cache_wrapper.module
            param_dict['method_name'] = self.clean_cache_wrapper.method_name
            param_dict['diff_mem_gb_thre'] = self.clean_cache_wrapper.diff_mem_gb_thre

        return param_dict
    
    @staticmethod
    def recursive_print(model, indent=0):
        for field_name, field_info in model.__fields__.items():
            field_value = getattr(model, field_name)
            print(" " * indent + f"{field_name}:")

            if issubclass(type(field_value), BaseModel):
                print(" " * (indent + 2) + f"--- Nested model: {field_value.__class__.__name__}")
                OffloadParam.recursive_print(field_value, indent + 4) 
            else:
                print(" " * (indent + 2) + f"class: {field_value.__class__.__name__}")
                if isinstance(field_value, torch.nn.Module):
                    pass
                else:
                    print(" " * (indent + 2) + f"value: {field_value}")

    def show(self):
        print("-"*20 + "[OffloadParam]" + "-"*20)
        OffloadParam.recursive_print(self)
        print("-"*40)


class OffloadParamParse:
    def __init__(self):
        pass

    @staticmethod
    def _get_model(root_model: torch.nn.Module, model_dir: str):
        assert(model_dir.startswith("self")), f"model_dir {model_dir} must startswith `self`"
        model = root_model
        for layer in model_dir.split('.'):
            if layer == "self":
                continue
            assert(hasattr(model, layer)), f"model not has layer [{layer}]!"
            model = getattr(model, layer)
        return model

    @staticmethod
    def parse_config(root_model: torch.nn.Module, cfg: omegaconf.DictConfig)->OffloadParam:
        assert(hasattr(cfg, "offload_module") and hasattr(cfg, "cpu_mem_gb") and hasattr(cfg, "dtype"))
        
        offload_module = OffloadParamParse._get_model(root_model, cfg.offload_module)
        cpu_mem_gb = cfg.cpu_mem_gb
        dtype = cfg.dtype

        pre_copy_step = cfg.pre_copy_step \
            if hasattr(cfg, "pre_copy_step") else None

        clean_cache_after_forward = cfg.clean_cache_after_forward \
            if hasattr(cfg, "clean_cache_after_forward") else None
            
        offload_layer_dict = {k: v for k, v in cfg.offload_layer_dict.items()} \
            if hasattr(cfg, "offload_layer_dict") else {}

        ignore_layer_list = cfg.ignore_layer_list \
            if hasattr(cfg, "ignore_layer_list") else []
        
        debug = cfg.debug if hasattr(cfg, "debug") else None
        
        clean_cache_wrapper = None
        if hasattr(cfg, "clean_cache_wrapper"):
            clean_cache_cfg = cfg.clean_cache_wrapper
            cc_module = OffloadParamParse._get_model(root_model, clean_cache_cfg.module)
            cc_method_name = clean_cache_cfg.method_name
            diff_mem_gb_thre = clean_cache_cfg.diff_mem_gb_thre
            clean_cache_wrapper = OffloadCleanCacheWrapperParam(
                                        module=cc_module, 
                                        method_name=cc_method_name, 
                                        diff_mem_gb_thre=diff_mem_gb_thre)
        
        return OffloadParam(
            offload_module=offload_module,
            cpu_mem_gb=cpu_mem_gb,
            pre_copy_step=pre_copy_step,
            clean_cache_after_forward=clean_cache_after_forward,
            dtype=dtype,
            offload_layer_dict=offload_layer_dict,
            ignore_layer_list=ignore_layer_list,
            clean_cache_wrapper=clean_cache_wrapper,
            debug=debug
            )


class LayerParamStruct:
    def __init__(self):
        self.count = 0
        self.device_state = None


class OffloadProfiler:
    def __init__(self, device_index=0, cpu_mem_gb=-1, pre_copy_step=1, clean_cache_after_forward=False, debug=False):
        self.clean_cache_after_forward = clean_cache_after_forward
        self.cpu_mem_gb = cpu_mem_gb
        self.cpu_mem_b_count = 0
        self.device_index = device_index
        self.execution_order = []
        self.execution_order_idx = {} 
        self.pin_memory = False
        test_data = torch.rand(1,1, device='cpu')
        pin_data = test_data.pin_memory()
        self.pin_memory = pin_data.is_pinned()
        print(f"pin:{self.pin_memory}")
        self.copy_stream = torch.cuda.Stream() 
        self.copy_queue = queue.Queue() 
        self.layer_param:Dict[str, LayerParamStruct] = {} 
        self.model_map = {}
        self.stop_flag = False
        self.copy_condition = threading.Condition()
        self.queue_condition = threading.Condition()
        self.mem_line_b = 0

        self.copy_thread = threading.Thread(target=self._copy_thread_fun)
        self.copy_thread.daemon = True
        self.copy_thread.start()

        self.cur_copy_idx = 0 
        self.execute_over = False
        self.pre_copy_step = pre_copy_step

        self.tmp_state_list = []
        self.tmp_state_idx = 0
        for i in range(pre_copy_step + 2):
            self.tmp_state_list.append(None)

        self.debug = debug

    def stop(self):
        self.stop_flag = True
        with self.queue_condition:
            self.queue_condition.notify()
        self.copy_thread.join()

        del self.layer_param
        del self.model_map
        del self.copy_stream

    def _copy_thread_fun(self):
        while self.stop_flag == False:
            layer_name = "--"
            with self.queue_condition:
                while self.copy_queue.qsize() == 0 and self.stop_flag == False:
                    self.queue_condition.wait()
                if self.stop_flag == True:
                    break
                layer_name = self.copy_queue.get()
            with torch.cuda.stream(self.copy_stream):
                if layer_name in self.model_map:
                    model = self.model_map[layer_name]
                    self.tmp_state_list[self.tmp_state_idx] = {
                        k: v.to(torch.device(f"cuda:{self.device_index}"), non_blocking=False)
                        for k, v in model.state_dict().items()
                    }
                    self.copy_stream.synchronize()

                    device_state = self.tmp_state_list[self.tmp_state_idx]
                    self.tmp_state_idx = (self.tmp_state_idx + 1) % len(self.tmp_state_list)

                    with self.copy_condition:
                        if layer_name in self.layer_param:
                            self.layer_param[layer_name].count += 1
                        else:
                            self.layer_param[layer_name] = LayerParamStruct()
                            self.layer_param[layer_name].count = 1
                        self.layer_param[layer_name].device_state = device_state
                        self.copy_condition.notify()
                else:
                    print(f"get model error! {layer_name}")
        print("copy thread stop..")

    def _get_new_step_copy_begin_end(self, tag_name):
        
        pre_copy_step = self.pre_copy_step
        pre_copy_step = min(pre_copy_step, len(self.execution_order) // 2)
        
        cur_exe_idx = self.execution_order_idx[tag_name]
        copy_begin = self.cur_copy_idx
        copy_end = cur_exe_idx + pre_copy_step + 1
        if copy_end - copy_begin > len(self.execution_order):
            copy_end %= len(self.execution_order)
        if copy_end - copy_begin > pre_copy_step + 1 or copy_end - copy_begin < 0:
            # jump
            self.cur_copy_idx = cur_exe_idx
            copy_begin, copy_end = self._get_new_step_copy_begin_end(tag_name=tag_name)
        return copy_begin, copy_end
    
    def make_forward_wrapper(self, module, tag_name, ignore_layer_list=[]):
        original_forward = module.forward
        layer_param_size = 0
        for name, param in module.named_parameters():
            layer_param_size += param.data.numel() * param.data.element_size() / 1024 / 1024 #MB
        
        taget_cpu_mem_b = self.cpu_mem_gb * 1024 * 1024 * 1024
        offload = False
        for name, param in module.named_parameters():
            p_name = f"{tag_name}.{name}" if tag_name else name
            for i_layer in ignore_layer_list:
                if p_name.startswith(i_layer):
                    if self.debug:
                        print(f"ignore layer param: {p_name}")
                    continue

            if taget_cpu_mem_b >= 0 and self.cpu_mem_b_count >= taget_cpu_mem_b:
                break
            cpu_data = torch.empty_strided(size=param.data.size(),
                                        stride=param.data.stride(),
                                        dtype=param.data.dtype,
                                        layout=param.data.layout,
                                        device='cpu',
                                        pin_memory=self.pin_memory)
            cpu_data.copy_(param.data)
            param.data = cpu_data

            param_size = param.data.numel() * param.data.element_size()
            self.cpu_mem_b_count += param_size
            offload = True
        if self.debug:
            print(f"layer: {tag_name}, type: {module.__class__.__name__}, size(MB): {layer_param_size}, offload: {offload}, sum_offload_size(MB): {self.cpu_mem_b_count/1024/1024}")
        
        if offload:
            copy_condition = self.copy_condition
            queue_condition = self.queue_condition
            copy_queue = self.copy_queue
            layer_param = self.layer_param
            def forward_wrapper(*args, **kwargs):
                module.forward = original_forward

                execute_over = False if tag_name not in self.execution_order_idx else True
                if execute_over == False:
                    self.model_map[tag_name] = module
                    self.execution_order.append(tag_name)
                    self.execution_order_idx[tag_name] = len(self.execution_order) - 1
                    copy_queue.put(tag_name)
                    with queue_condition:
                        queue_condition.notify()
                else: 
                
                    copy_begin, copy_end = self._get_new_step_copy_begin_end(tag_name=tag_name)
                    if copy_end > copy_begin:
                        for idx in range(copy_begin, copy_end):
                            idx = idx % len(self.execution_order)
                            copy_tag_name = self.execution_order[idx]
                            copy_queue.put(copy_tag_name)
                            with queue_condition:
                                queue_condition.notify()

                        self.cur_copy_idx = copy_end % len(self.execution_order)
                
                run_state = None
                with self.copy_condition:
                    while tag_name not in self.layer_param:
                        copy_condition.wait()
                    run_state = self.layer_param[tag_name].device_state
                    self.layer_param[tag_name].count -= 1
                    
                module.eval()
                with torch.no_grad():
                    output = functional_call(module, run_state, args=args, kwargs=kwargs)
                with self.copy_condition:
                    if self.layer_param[tag_name].count == 0:
                        del self.layer_param[tag_name]
                diff_mem_b_thre = 1 * (1024 ** 3)
                if self.clean_cache_after_forward:
                    reserved = torch.cuda.memory_reserved()
                    if reserved > self.mem_line_b:
                        torch.cuda.empty_cache()
                        cur_reserved = torch.cuda.memory_reserved()
                        diff_mem = reserved - cur_reserved
                        if diff_mem > diff_mem_b_thre:
                            self.mem_line_b = cur_reserved + (reserved - cur_reserved) / 2 + 10
                        else:
                            self.mem_line_b = reserved + 10
                        if self.debug:
                            print(f"child mem line update, clean cache:{reserved/1024/1024}, cur mem: {cur_reserved/1024/1024}  new limit: {self.mem_line_b / 1024 / 1024}, child name: {tag_name}")
                    
                module.forward = forward_wrapper
                return output
            module.forward = forward_wrapper
        
        torch.cuda.empty_cache()
        return module
    
    def reset_empty_cache_mem_line(self):
        self.mem_line_b = 0
        torch.cuda.empty_cache()
    
    def clean_cache_wrapper(self, module, method_name='', diff_mem_gb_thre=1):
        if not hasattr(module, method_name) or not callable(getattr(module, method_name)):
            print(f"no this method {method_name}")
            return module
        
        original_fun = getattr(module, method_name)
        diff_mem_b_thre = diff_mem_gb_thre * (1024 ** 3)
        self.reset_empty_cache_mem_line()

        def clean_wrapper(*args, **kwargs):
            setattr(module, method_name, original_fun)
            output = original_fun(*args, **kwargs)
            reserved = torch.cuda.memory_reserved()
            if reserved > self.mem_line_b:
                torch.cuda.empty_cache()
                cur_reserved = torch.cuda.memory_reserved()
                diff_mem = reserved - cur_reserved
                if diff_mem > diff_mem_b_thre:
                    self.mem_line_b = cur_reserved + (reserved - cur_reserved) / 2 + 10
                else:
                    self.mem_line_b = reserved + 10

                if self.debug:
                    print(f"mem line update, clean cache:{reserved/1024/1024}, cur mem: {cur_reserved/1024/1024}  new limit: {self.mem_line_b / 1024 / 1024}")
            setattr(module, method_name, clean_wrapper)
            return output
        
        setattr(module, method_name, clean_wrapper)
        return module
    
    @_callable_once
    def offload_layer(self, module, offload_layer_dict={},  ignore_layer_list=[], dtype:torch.dtype = None):
        return self._offload_layer(
                                    module=module,
                                    tag="",
                                    offload_layer_dict=offload_layer_dict,
                                    ignore_layer_list=ignore_layer_list,
                                    dtype=dtype
                                    )
    
    def _offload_layer(self, module, tag="", offload_layer_dict={},  ignore_layer_list=[], dtype:torch.dtype = None):
        """
            Offload specific layers of a PyTorch model to a specified depth.
            A model can only be offloaded once.

            Args:
                module (torch.nn.Module): 
                    The PyTorch model containing the layers to offload. This is the model that will be modified in place.
                
                tag (str, optional): 
                    A string identifier for the model. 
                    Default is an empty string.
                
                offload_layer_dict (dict, optional): 
                    A dictionary where keys are layer names and values represent the depth at which the offloading should occur. 
                    For example, 
                    ```offload_layer_dict = {'cfm_wrapper': 5, 'hubert': 4}``` means that the `cfm_wrapper` layer should 
                    be offloaded at depth 5, and the `hubert` layer should be offloaded at depth 4.
                    Default is an empty dictionary.
                
                ignore_layer_list (list, optional): 
                    A list of layer names or parameter identifiers to be ignored during the offloading process. 
                    Layers in this list will not be offloaded, even if they are present in the `offload_layer_dict`. 
                     For example, 
                    ```ignore_layer_list = ['cfm_wrapper.estimator.h', 'cfm_wrapper.estimator.adaln_single']```
                    means that layers starting with `cfm_wrapper.estimator.h` or  'cfm_wrapper.estimator.adaln_single' will not be offload.
                    Default is an empty list.
                
                dtype (torch.dtype, optional): 
                    The data type (e.g., `torch.float16`, `torch.float32`) to which the offloaded layers should be converted. 
                    If `None`, the data type of the layers will remain unchanged. Default is `None`.

            Returns:
                None
        """
        for p in module._parameters.values():
            if p is not None:
                p.data = p.data.to(torch.device(f"cuda:{self.device_index}"))
                if dtype is not None:
                    p.data = p.data.to(dtype)
        for b in module._buffers.values():
            if b is not None:
                b.data = b.data.to(torch.device(f"cuda:{self.device_index}"))
                if dtype is not None:
                    b.data = b.data.to(dtype)
        for attr_name, attr in module.__dict__.items():
            if isinstance(attr, torch.Tensor) and not attr_name.startswith('_'):
                attr.data = attr.data.to(torch.device(f"cuda:{self.device_index}"))
                if dtype is not None:
                    attr.data = attr.data.to(dtype)

        for name, child in module.named_children():
            current_tag = f"{tag}.{name}" if tag else name
            child = child.to(torch.device(f"cuda:{self.device_index}"))
            if dtype is not None:
                child = child.to(dtype)

            torch.cuda.empty_cache()
            setattr(module, name, child)
            pre_name = current_tag.split('.')[0]
            if pre_name not in offload_layer_dict:
                param_size = 0
                for p in child.parameters():
                    param_size += p.data.numel() * p.data.element_size()
                param_size = param_size / 1024 / 1024
                if self.debug:
                    print(f"not offload layer {current_tag}, size: {param_size}MB")
                continue
            
            has_children = any(child.named_children())
            layer_count = current_tag.count('.') + 1
            
            layer_deep = offload_layer_dict[pre_name]
            if layer_count >= layer_deep:
                has_children = False 
            
            if has_children:
                self._offload_layer(module=child, 
                                   tag=current_tag, 
                                   offload_layer_dict=offload_layer_dict, 
                                   ignore_layer_list=ignore_layer_list,
                                   dtype=dtype)
                continue

            ignore = False
            for i_layer in ignore_layer_list:
                if current_tag.startswith(i_layer):
                    ignore = True
                    if self.debug:
                        print(f"ignore layer offload: {current_tag}")
                    break
    
            if hasattr(child, "forward") and not ignore:
                child = self.make_forward_wrapper(
                    child, current_tag, ignore_layer_list=ignore_layer_list
                )
        return module
    
    def get_execution_order(self):
        return self.execution_order
