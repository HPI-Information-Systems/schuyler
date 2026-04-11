#! /bin/bash
export WANDB_API_KEY=$WANDB_API_KEY
function run_experiment {
	echo Doing it for $1
	echo Tagging with $2
	if [ "$3" == "wandb" ]; then
		echo "Using wandb"
		python3 evaluator/experimenter/experiment_script.py --scenario $1 --tag $2 --wandb
	else
		echo "Not using wandb"
		python3 evaluator/experimenter/experiment_script.py --scenario $1 --tag $2
	fi
}

set -e

run_experiment $1 $2 $3