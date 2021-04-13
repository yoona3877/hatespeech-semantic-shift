#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -p interactive
#SBATCH --qos=nopreemption
#SBATCH --output=countvec_%j.log

. /etc/profile.d/lmod.sh
. $HOME/condaenvs/tweet_covid/tweet_covid_env

python -u count_freq.py --days 30
