# the rank of distributed node worker
# If I use two nodes, 4 gpus per each, then WORKER_RANK for the two node should be 0, 4, i.e. the starting indice of the GPU.
WORKER_RANK=${1:-'0'}
PLATFORM=${2:-'shef'} 
YAML_NAME_WITHOUT_EXT=${3:-'MERT_RVQ-VAE_CQT_95M'}
TRAINING_SETTING=${4:-'MERT_RVQ-VAE_CQT'}
MASTER_PROC_ADD=${5:-'127.0.0.1'}
DIST_PORT=${6:-'39683'}

echo "worker rank ${WORKER_RANK}, master address ${MASTER_PROC_ADD}:${DIST_PORT}"

MAP_PROJ_DIR=$HOME/MERT

DISTRIBUTED_WORLD_SIZE=2
NPROCES_PER_NODE=2
MAX_TOKENS=1000000 # set for 80GB A100
NUM_WOKERS=6

run_command_prefix=' '
# Loading folders
# 1. tsv files for audio paths
DATA_DIR=${MAP_PROJ_DIR}/data/audio_tsv
# 2. working folder for saving checkpoints and loading config files
CONFIG_DIR=/${MAP_PROJ_DIR}/mert_fairseq/config/pretrain
# 3. clustering labels for training data
LABEL_ROOT_DIR=${MAP_PROJ_DIR}/data/labels


FAIRSEQ_PATH=${MAP_PROJ_DIR}/src/fairseq;
SAVE_DIR=${MAP_PROJ_DIR}/data/fairseq_savedir/

# set 75 for the RVQ-VAE model
LABEL_RATE=75

case $YAML_NAME_WITHOUT_EXT in
    MERT_RVQ-VAE_CQT_95M)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        DISTRIBUTED_WORLD_SIZE=8
        NPROCES_PER_NODE=1
        LABEL_RATE=75
        MAX_TOKENS=1800000
        ;;
    MERT_RVQ-VAE_CQT_330M)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        DISTRIBUTED_WORLD_SIZE=64
        NPROCES_PER_NODE=8
        LABEL_RATE=75
        MAX_TOKENS=920000
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

  ACTUAL_WORKER_RANK=`expr ${WORKER_RANK} \* ${NPROCES_PER_NODE}`
  echo "worker rank ${WORKER_RANK}, master address ${MASTER_PROC_ADD}:${DIST_PORT}, actual rank ${ACTUAL_WORKER_RANK}"

  OMP_NUM_THREADS=6 ${run_command_prefix} python -u ${FAIRSEQ_PATH}/fairseq_cli/hydra_train.py \
    --config-dir ${CONFIG_DIR} --config-name ${YAML_NAME_WITHOUT_EXT} \
    common.user_dir=${MAP_PROJ_DIR}/mert_faiseq \
    common.wandb_project=pretrain_${TRAINING_SETTING} \
    checkpoint.save_dir=${SAVE_DIR}/ckpt_${TRAINING_SETTING}/${YAML_NAME_WITHOUT_EXT} \
    distributed_training.distributed_rank=${ACTUAL_WORKER_RANK} \
    distributed_training.distributed_world_size=${DISTRIBUTED_WORLD_SIZE}  \
    distributed_training.nprocs_per_node=${NPROCES_PER_NODE} \
    distributed_training.distributed_init_method="tcp://${MASTER_PROC_ADD}:${DIST_PORT}" \
    task.data=${DATA_DIR} task.label_dir=${LABEL_DIR} \
    task.labels=${TASK_LABELS_POSTFIX} \
    dataset.num_workers=${NUM_WOKERS} \
    dataset.max_tokens=${MAX_TOKENS} \
    dataset.disable_validation=true \
    model.label_rate=${LABEL_RATE}