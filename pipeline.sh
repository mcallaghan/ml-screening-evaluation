#!/bin/bash

#SBATCH --job-name=ml-screening-pipeline
#SBATCH --qos=short
#SBATCH --time=00:01:00
#SBATCH --output=out/pipeline.out
#SBATCH --error=out/pipeline.err

job1=$(sbatch --parsable -J "screen-simulation" --ntasks $1 run_prioritisation.sh $1 "$2")
#pred2=$(sbatch --parsable --dependency=afterok:$pred1 -J "$JNAME" run_predictions.sh $MODEL "$VAR")
	
