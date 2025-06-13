export USER=root
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
export NCCL_HOME=/usr/local/tccl
export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/":$PYTHONPATH

CKPT_PATH=$1
JSONL=$2
SAVE_DIR=$3
python3 generate.py $CKPT_PATH $JSONL $SAVE_DIR
