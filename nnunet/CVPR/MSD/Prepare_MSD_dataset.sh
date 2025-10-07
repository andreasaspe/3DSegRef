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

# === Unpack all MSD datasets from .tar files ===
echo "Unpacking MSD datasets..."
for tarfile in "$MSD_DIR"/*.tar; do
    echo "Extracting $tarfile..."
    tar -xvf "$tarfile" -C "$MSD_DIR"
done

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
IDS=(14 15 16 17 18 19 20 21 22 23)

for i in "${!DATASETS[@]}"; do
    task="${DATASETS[$i]}"
    id="${IDS[$i]}"
    echo "Converting $task → Dataset $id..."
    nnUNetv2_convert_MSD_dataset -i "$MSD_DIR/$task" -overwrite_id "$id"
done

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
