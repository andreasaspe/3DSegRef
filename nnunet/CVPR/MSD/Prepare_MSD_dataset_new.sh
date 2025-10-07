#!/bin/bash
# Usage: bash Automated_setup_nnunet.sh
set -e  # Stop on error

# conda activate might not work directly inside scripts, so we use source activate
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate nnunet_v2

# === Paths ===
export nnUNet_raw="/scratch/awias/data/nnUNet/nnUNet_raw"
export nnUNet_results="/scratch/awias/data/nnUNet/nnUNet_results"
export nnUNet_preprocessed="/scratch/awias/data/nnUNet/nnUNet_preprocessed"

MSD_DIR="/scratch/awias/data/MSD"


# === Convert to nnUNet format ===
echo "Converting MSD datasets..."
DATASETS=(
    "Task01_BrainTumour" 
    "Task02_Heart"
    "Task03_Liver"
    "Task04_Hippocampus"
    "Task05_Prostate"
    "Task06_Lung"
    "Task07_Pancreas"
    "Task08_HepaticVessel"
    "Task09_Spleen"
    "Task10_Colon"
)
IDS=(16 17 18 19 20 21 22 23)

# === Fingerprint, plan, and preprocess ===
echo "Processing datasets..."
for id in "${IDS[@]}"; do
    echo "Processing dataset ID $id ..."
    nnUNetv2_extract_fingerprint -d "$id" -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres
    nnUNetv2_plan_experiment -d "$id" -c 3d_fullres -pl nnUNetPlannerResEncL -np 4
    nnUNetv2_preprocess -d "$id" -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8
done

echo "✅ All MSD datasets converted and preprocessed successfully."

conda deactivate

exit 0
