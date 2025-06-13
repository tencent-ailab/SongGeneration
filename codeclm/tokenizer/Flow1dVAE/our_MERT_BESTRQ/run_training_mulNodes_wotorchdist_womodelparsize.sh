# bash run_training_mulNodes_wotorchdist.sh 0 dummy MERT_RVQ-VAE_CQT_330M_multinodes
# bash run_training_mulNodes_wotorchdist.sh 1 dummy MERT_RVQ-VAE_CQT_330M_multinodes
# bash run_training_mulNodes_wotorchdist.sh 2 dummy MERT_RVQ-VAE_CQT_330M_multinodes
# bash run_training_mulNodes_wotorchdist.sh 3 dummy MERT_RVQ-VAE_CQT_330M_multinodes

# the rank of distributed node worker
# If I use two nodes, 4 gpus per each, then WORKER_RANK for the two node should be 0, 4, i.e. the starting indice of the GPU.
WORKER_RANK=${1:-$INDEX}
PLATFORM=${2:-'shef'} 
YAML_NAME_WITHOUT_EXT=${3:-'MERT_RVQ-VAE_CQT_95M'}
TRAINING_SETTING=${4:-'MERT_RVQ-VAE_CQT'}
MASTER_PROC_ADD=${5:-$CHIEF_IP}
DIST_PORT=${6:-'25520'}
DATASET_NAME=${7:-'dataindex'}
# echo $PATH
# export PATH=$PATH:./
echo "worker rank ${WORKER_RANK}, master address ${MASTER_PROC_ADD}:${DIST_PORT}"

MAP_PROJ_DIR=$(pwd)
echo $MAP_PROJ_DIR

NNODS=1
# MAX_TOKENS=1000000 # set for 80GB A100 batchsize
NUM_WOKERS=6

run_command_prefix=' '
# Loading folders
# 1. tsv files for audio paths
# DATA_DIR=${MAP_PROJ_DIR}/data/audio_tsv
DATA_DIR=${MAP_PROJ_DIR}/data/${DATASET_NAME} #audio_manifest
# 2. working folder for saving checkpoints and loading config files
CONFIG_DIR=/${MAP_PROJ_DIR}/mert_fairseq/config/pretrain
# 3. clustering labels for training data
LABEL_ROOT_DIR=${MAP_PROJ_DIR}/data/encodec_labels/custom_audio_dataset

FAIRSEQ_PATH=${MAP_PROJ_DIR}/src/fairseq;
SAVE_DIR=${MAP_PROJ_DIR}/data/fairseq_savedir/

# set 75 for the RVQ-VAE model
LABEL_RATE=75

case $YAML_NAME_WITHOUT_EXT in
    MERT_RVQ-VAE_CQT_95M)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=1
        LABEL_RATE=75
        MAX_TOKENS=1800000
        ;;
    MERT_RVQ-VAE_CQT_95M_mel_multinodes)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1200000
        ;;
    MERT_RVQ-VAE_CQT_95M_bestrq_multinodes)
        TASK_LABELS_POSTFIX='["rq_0"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1200000
        ;;
    MERT_RVQ-VAE_CQT_95M_bestrq_chroma_multinodes)
        TASK_LABELS_POSTFIX='["rq_0"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1600000
        ;;
    MERT_RVQ-VAE_CQT_95M_bestrq_norm_multinodes)
        TASK_LABELS_POSTFIX='["rq_0"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1600000
        ;;
    MERT_RVQ-VAE_CQT_95M_dac_multinodes)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1600000
        ;;
    MERT_RVQ-VAE_CQT_95M_groupbestrq_multinodes)
        TASK_LABELS_POSTFIX='["grq_0"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=1600000
        ;;
    MusicFM_95M_multinodes)
        TASK_LABELS_POSTFIX='["grq_0"]'
        NNODS=4
        LABEL_RATE=25
        NPROCES_PER_NODE=8
        MAX_TOKENS=4800000
        ;;
    MusicFM_95M_bestrvq_multinodes)
        TASK_LABELS_POSTFIX='["grq_0"]'
        NNODS=4
        LABEL_RATE=25
        NPROCES_PER_NODE=8
        MAX_TOKENS=4800000
        ;;
    MusicFM_95M_speech_multinodes)
        TASK_LABELS_POSTFIX='[]'
        NNODS=4
        LABEL_RATE=25
        NPROCES_PER_NODE=8
        MAX_TOKENS=1200000
        ;;
    MERT_RVQ-VAE_CQT_330M)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=1
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=720000
        ;;
    MERT_RVQ-VAE_CQT_330M_multinodes)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=4
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=600000
        ;;
    MERT_RVQ-VAE_CQT_330M_multinodes_debug2node)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=2
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=600000
        ;;
    MERT_RVQ-VAE_CQT_330M_multinodes_debug1node)
        TASK_LABELS_POSTFIX='["encodec_0","encodec_1","encodec_2","encodec_3","encodec_4","encodec_5","encodec_6","encodec_7"]'
        NNODS=1
        LABEL_RATE=75
        NPROCES_PER_NODE=8
        MAX_TOKENS=600000
        ;;
    *)
        echo "Unknown running config: ${$YAML_NAME_WITHOUT_EXT} = ${YAML_NAME_WITHOUT_EXT}"
        exit 1
        ;;
    esac

 echo running $YAML_NAME_WITHOUT_EXT ..

  mkdir -p ${SAVE_DIR}
  echo "checkpoint save at: ${SAVE_DIR}"
  cd ${SAVE_DIR}

  echo "NPROCES_PER_NODE is ${NPROCES_PER_NODE}"

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
    task.label_dir=${LABEL_DIR} \
    task.labels=${TASK_LABELS_POSTFIX} \
    dataset.num_workers=${NUM_WOKERS} \
    dataset.max_tokens=${MAX_TOKENS} \
    dataset.disable_validation=true \
    model.label_rate=${LABEL_RATE} \
