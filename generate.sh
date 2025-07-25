export USER=root
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
export NCCL_HOME=/usr/local/tccl
export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/":$PYTHONPATH

CKPT_PATH=$1
JSONL=$2
SAVE_DIR=$3
USE_FLASH_ATTN="True"
LOW_MEM="False"
GENERATE_TYPE="mixed"
for arg in "$@"; do
    if [[ $arg == "--not_use_flash_attn" ]]; then
        USE_FLASH_ATTN="False"
    fi
done
for arg in "$@"; do
    if [[ $arg == "--low_mem" ]]; then
        LOW_MEM="True"
    fi
done
for arg in "$@"; do
    if [[ $arg == "--separate" ]]; then
        GENERATE_TYPE="separate"
    fi
done
for arg in "$@"; do
    if [[ $arg == "--bgm" ]]; then
        GENERATE_TYPE="bgm"
    fi
done
for arg in "$@"; do
    if [[ $arg == "--vocal" ]]; then
        GENERATE_TYPE="vocal"
    fi
done


if [ "$USE_FLASH_ATTN" == "True" ] && [ "$LOW_MEM" == "True" ]; then
    echo "Use Flash Attention + Low Memory Mode"
    python3 generate.py \
        --ckpt_path $CKPT_PATH \
        --input_jsonl $JSONL \
        --save_dir $SAVE_DIR \
        --generate_type $GENERATE_TYPE \
        --use_flash_attn \
        --low_mem 
elif [ "$USE_FLASH_ATTN" == "True" ] && [ "$LOW_MEM" == "False" ]; then
    echo "Use Flash Attention + Auto Memory Mode"
    python3 generate.py \
        --ckpt_path $CKPT_PATH \
        --input_jsonl $JSONL \
        --save_dir $SAVE_DIR \
        --generate_type $GENERATE_TYPE \
        --use_flash_attn 
elif [ "$USE_FLASH_ATTN" == "False" ] && [ "$LOW_MEM" == "False" ]; then
    echo "Not Use Flash Attention + Auto Memory Mode"
    python3 generate.py \
        --ckpt_path $CKPT_PATH \
        --input_jsonl $JSONL \
        --generate_type $GENERATE_TYPE \
        --save_dir $SAVE_DIR 
elif [ "$USE_FLASH_ATTN" == "False" ] && [ "$LOW_MEM" == "True" ]; then
    echo "Not Use Flash Attention + Low Memory Mode"
    python3 generate.py \
        --ckpt_path $CKPT_PATH \
        --input_jsonl $JSONL \
        --save_dir $SAVE_DIR \
        --generate_type $GENERATE_TYPE \
        --low_mem 
fi
