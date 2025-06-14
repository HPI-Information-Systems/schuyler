# import faulthandler
# faulthandler.enable()
import numpy as np
import argparse
from schuyler.experimenter.ExperimentManager import ExperimentManager
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    # reda command line arguments
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--tag", type=str, help="Tag for the experiment")
    #parser.add_argument("--solution", type=str, help="Name of the solution to run")
    parser.add_argument("--scenario", type=str, help="Name of the scenario group to run")
    parser.add_argument("--wandb", action='store_true', help="Upload to wandb")
    args = parser.parse_args()

    experiment_manager = ExperimentManager(args.tag, use_wandb=args.wandb)
    experiment_manager.start_experiments(args.scenario)