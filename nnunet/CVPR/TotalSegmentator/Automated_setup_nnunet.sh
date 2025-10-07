#!/bin/bash
# Usage: bash Automated_setup_nnunet.sh <dataset_id>

# Stop script if any command fails
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_id>"
    exit 1
fi

ID=$1

# Activate conda environment
# (if 'conda activate' doesn't work directly inside scripts, use 'source activate')
# source source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate nnunet_v2

# Set nnU-Net environment variables
export nnUNet_raw="/scratch/awias/data/nnUNet/nnUNet_raw"
export nnUNet_results="/scratch/awias/data/nnUNet/nnUNet_results"
export nnUNet_preprocessed="/scratch/awias/data/nnUNet/nnUNet_preprocessed"

# Step 5: Extract fingerprint
nnUNetv2_extract_fingerprint -d $ID -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres

# Step 6: Plan experiment
nnUNetv2_plan_experiment -d $ID -c 3d_fullres -pl nnUNetPlannerResEncL -np 4

# Step 7: Preprocess dataset
nnUNetv2_preprocess -d $ID -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8

exit