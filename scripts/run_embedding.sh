#!/bin/bash
#SBATCH --mem=4G
#SBATCH -c 2
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=bert_emb_%j.log

. /etc/profile.d/lmod.sh
. $HOME/condaenvs/tweet_covid/tweet_covid_env

python bert_embedding.py --chunk_id $1 --export /h/ypark/tweet_covid/hatespeech/output
