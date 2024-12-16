#! /bin/bash
#outdir="NumbER/experiments/number"
#config_path="NumbER/configs/number.yaml"
# eval "$(conda shell.bash hook)"
# source .env
export WANDB_API_KEY=$WANDB_API_KEY
function run_experiment {
	echo Doing it for $1
	echo Tagging with $2
	#conda activate py39
	#check if $3 is value wandb
	if [ "$3" == "wandb" ]; then
		echo "Using wandb"
		python3 evaluator/experimenter/experiment_script.py --scenario $1 --tag $2 --wandb
	else
		echo "Not using wandb"
		python3 evaluator/experimenter/experiment_script.py --scenario $1 --tag $2
	fi
	#python3 evaluator/experimenter/experiment_script.py --scenario $1 --tag $2 --wandb #> output_2.out
}

set -e
#eval "$(conda shell.bash hook)"
#cd /Users/lukaslaskowski/Documents/HPI/KG/ontology_mappings/rdb2ontology/

run_experiment $1 $2 $3