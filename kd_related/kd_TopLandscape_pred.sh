#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_TopLandscape`
DATADIR=${DATADIR_TopLandscape}
[[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
# set a comment via `COMMENT`
suffix=${COMMENT}

model=$1
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT_1" ]]; then
    modelopts="networks/benchmark_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "KDParT" ]]; then
    modelopts="networks/kd_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
    extraopts="--batch-size 256 --kd-mode --load-teacher-weights  /home/olympus/stqian/part_kd/particle_transformer/training/TopLandscape/ParT-FineTune/20230819-085506_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none --load-model-weights models/ParT_kin.pt --jit-output test_ParT_jit.pt"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none --load-model-weights models/ParticleNet_kin.pt"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096" 
elif [[ "$model" == "PFN_1" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 256" 
elif [[ "$model" == "KDPFN" ]]; then
    modelopts="networks/kd_PFN.py "
    lr="2e-2"
    extraopts="--batch-size 256 --kd-mode --load-teacher-weights  /home/olympus/stqian/part_kd/particle_transformer/training/TopLandscape/ParT-FineTune/20230819-085506_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt"
elif [[ "$model" == "KDPFN_1" ]]; then
    modelopts="networks/kd_PFN.py "
    lr="2e-2"
    extraopts="--batch-size 2048 --kd-mode --load-teacher-weights  /home/olympus/stqian/part_kd/particle_transformer/training/TopLandscape/ParT-FineTune/20230819-085506_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN_1" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 256"
elif [[ "$model" == "KDPCNN" ]]; then
    modelopts="networks/kd_PCNN.py "
    lr="2e-2"
    extraopts="--batch-size 256 --kd-mode --load-teacher-weights  /home/olympus/stqian/part_kd/particle_transformer/training/TopLandscape/ParT-FineTune/20230819-085506_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"
if [[ "${FEATURE_TYPE}" != "kin" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

weaver \
    --data-test "${DATADIR}/test_file.parquet" \
    --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory \
    # --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 20 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
    --predict \
    # --tensorboard KD_TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
    ${extraopts} "${@:3}"
