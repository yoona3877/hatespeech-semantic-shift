#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=tokenize_%j.log

. /etc/profile.d/lmod.sh
. $HOME/condaenvs/tweet_covid/tweet_covid_env

python -u mapping.py --days $1 --tweet_dir /h/ypark/tweet_covid/output --export /h/ypark/tweet_covid/output/unigrams --hatebase_dir /h/ypark/tweet_covid/hatespeech/hatebase

