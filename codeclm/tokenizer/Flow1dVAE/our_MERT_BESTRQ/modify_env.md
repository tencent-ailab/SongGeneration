cp -r fairseq/fairseq/model_parallel/megatron /opt/conda/envs/map/lib/python3.8/site-packages/fairseq/model_parallel/
vi /opt/conda/envs/map/lib/python3.8/site-packages/apex/amp/_initialize.py # string_classes = str
vi /opt/conda/envs/map/lib/python3.8/site-packages/fairseq/modules/layer_norm.py
vi /opt/conda/envs/map/lib/python3.8/site-packages/fairseq/distributed/utils.py # import datetime; timeout=datetime.timedelta(seconds=51200); logger.info("add nccl time to 51200")
