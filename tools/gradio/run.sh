export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

export USER=root
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
export NCCL_HOME=/usr/local/tccl
export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/":$PYTHONPATH


CKPT_PATH=$1
python3 tools/gradio/app.py $CKPT_PATH
