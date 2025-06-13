WORKER_RANK=${1:-$INDEX}
PLATFORM=${2:-'shef'} 
YAML_NAME_WITHOUT_EXT=${3:-'MERT_RVQ-VAE_CQT_95M'}
TRAINING_SETTING=${4:-'MERT_RVQ-VAE_CQT'}
MASTER_PROC_ADD=${5:-$CHIEF_IP}
DIST_PORT=${6:-'25520'}
# echo $PATH
# export PATH=$PATH:./
echo "worker rank ${WORKER_RANK}, master address ${MASTER_PROC_ADD}:${DIST_PORT}"

MAP_PROJ_DIR=$(pwd)
echo $MAP_PROJ_DIR

NNODS=1
BATCH_SIZE=12
NUM_WOKERS=6

run_command_prefix=' '
# Loading folders
# 1. tsv files for audio paths
# DATA_DIR=${MAP_PROJ_DIR}/data/audio_tsv
DATA_DIR=${MAP_PROJ_DIR}/data/music4all_sh #audio_manifest
# 2. working folder for saving checkpoints and loading config files
CONFIG_DIR=/${MAP_PROJ_DIR}/mert_fairseq/config/pretrain
# 3. clustering labels for training data
LABEL_ROOT_DIR=${MAP_PROJ_DIR}/data/encodec_labels/custom_audio_dataset

FAIRSEQ_PATH=${MAP_PROJ_DIR}/src/fairseq;
SAVE_DIR=${MAP_PROJ_DIR}/data/fairseq_savedir/

case $YAML_NAME_WITHOUT_EXT in
    EAT_pretraining_music_multinodes)
        NNODS=4
        NPROCES_PER_NODE=8
        LABEL_RATE=25
        BATCH_SIZE=12
        ;;
    *)
        echo "Unknown running config: ${$YAML_NAME_WITHOUT_EXT}"
        exit 1
        ;;
    esac

echo running $YAML_NAME_WITHOUT_EXT ..

mkdir -p ${SAVE_DIR}
echo "checkpoint save at: ${SAVE_DIR}"
cd ${SAVE_DIR}

DISTRIBUTED_WORLD_SIZE=`expr ${NNODS} \* ${NPROCES_PER_NODE}`
ACTUAL_WORKER_RANK=`expr ${WORKER_RANK} \* ${NPROCES_PER_NODE}`
echo "worker rank ${WORKER_RANK}, master address ${MASTER_PROC_ADD}:${DIST_PORT}, actual rank ${ACTUAL_WORKER_RANK}"

DATE_SUFFIX=`date +"%Y-%m-%d_%H-%M"`

OMP_NUM_THREADS=6 ${run_command_prefix} \
python -u ${FAIRSEQ_PATH}/fairseq_cli/hydra_train.py \
--config-dir ${CONFIG_DIR} --config-name ${YAML_NAME_WITHOUT_EXT} \
common.user_dir=${MAP_PROJ_DIR}/mert_fairseq \
common.tensorboard_logdir=${MAP_PROJ_DIR}/logs/pretrain_tb_${TRAINING_SETTING}_${YAML_NAME_WITHOUT_EXT}_multinodes${NNODS} \
checkpoint.save_dir=${SAVE_DIR}/ckpt_${TRAINING_SETTING}_multinodes${NNODS}_${DATE_SUFFIX}/${YAML_NAME_WITHOUT_EXT} \
distributed_training.distributed_rank=${ACTUAL_WORKER_RANK} \
distributed_training.distributed_world_size=${DISTRIBUTED_WORLD_SIZE}  \
distributed_training.distributed_num_procs=${DISTRIBUTED_WORLD_SIZE}  \
distributed_training.nprocs_per_node=${NPROCES_PER_NODE} \
distributed_training.distributed_init_method="tcp://${CHIEF_IP}:${DIST_PORT}" \
task.data=${DATA_DIR} \
dataset.num_workers=${NUM_WOKERS} \
dataset.batch_size=${BATCH_SIZE} \
dataset.disable_validation=true \

# pip install h5py timm -i https://mirrors.tencent.com/pypi/simple/