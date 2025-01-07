#!/bin/bash

# =================================================================================
# pass all arguments (both variables and constants) to rl-baselines3-zoo train.py 
# =================================================================================

cd $PBS_O_WORKDIR
source ../../venv/bin/activate
python -u train.py \
--algo ppo  --env ${game} --tensorboard-log "logs/tb_logs/${game}/${self_attn}/${seed}" --eval-freq 200000 --eval-episodes 5 \
--save-freq 500000 --log-folder "logs/exp_logs/${game}/${self_attn}/${seed}" --seed ${seed} --vec-env subproc \
--hyperparams policy_kwargs:"dict(features_extractor_class=SelfAttnCNNPPO, features_extractor_kwargs=dict(self_attn='''"${self_attn}"'''), net_arch=[])" \
--uuid