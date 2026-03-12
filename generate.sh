export USER=root
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
export NCCL_HOME=/usr/local/tccl
export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/":$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0


CKPT_DIR=$1
JSONL=$2
SAVE_DIR=$3
python3 generate.py \
    --input_jsonl $JSONL \
    --save_dir $SAVE_DIR \
    --ckpt_dir $CKPT_DIR
