#!/bin/bash
# Usage: bash Automated_setup_nnunet.sh <dataset_id>

# Stop script if any command fails
set -e

# Set nnU-Net environment variables
export nnUNet_raw="/scratch/awias/data/nnUNet/nnUNet_raw"
export nnUNet_results="/scratch/awias/data/nnUNet/nnUNet_results"
export nnUNet_preprocessed="/scratch/awias/data/nnUNet/nnUNet_preprocessed"

# Unpack all MSD datasets
tar -xvf /scratch/awias/data/MSD/Task01_BrainTumour-006.tar
tar -xvf /scratch/awias/data/MSD/Task02_Heart.tar
tar -xvf /scratch/awias/data/MSD/Task03_Liver-001.tar
tar -xvf /scratch/awias/data/MSD/Task04_Hippocampus.tar
tar -xvf /scratch/awias/data/MSD/Task05_Prostate.tar
tar -xvf /scratch/awias/data/MSD/Task06_Lung-004.tar
tar -xvf /scratch/awias/data/MSD/Task07_Pancreas-007.tar
tar -xvf /scratch/awias/data/MSD/Task08_HepaticVessel-005.tar
tar -xvf /scratch/awias/data/MSD/Task09_Spleen.tar
tar -xvf /scratch/awias/data/MSD/Task10_Colon-003.tar

# Convert to right data structure
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task01_BrainTumour-006.tar -overwrite_id 14
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task02_Heart.tar -overwrite_id 15
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task03_Liver-001.tar -overwrite_id 16
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task04_Hippocampus.tar -overwrite_id 17
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task05_Prostate.tar -overwrite_id 18
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task06_Lung-004.tar -overwrite_id 19
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task07_Pancreas-007 -overwrite_id 20
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task08_HepaticVessel-005.tar -overwrite_id 21
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task09_Spleen.tar -overwrite_id 22
nnUNetv2_convert_MSD_dataset -i /scratch/awias/data/MSD/Task10_Colon-003.tar -overwrite_id 23 

IDS=(14 15 16 17 18 19 20 21 22 23)

# Extract fingerprint, plan experiment, preprocess dataset for each MSD dataset
nnUNetv2_extract_fingerprint -d 14 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 14 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 14 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 15 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 15 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 15 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 16 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 16 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 16 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 17 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 17 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 17 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 18 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 18 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 18 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 19 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 19 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 19 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 20 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 20 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 20 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 21 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 21 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 21 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 22 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 22 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 22 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

nnUNetv2_extract_fingerprint -d 23 -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres # Extract fingerprint
nnUNetv2_plan_experiment -d 23 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4 # Plan experiment
nnUNetv2_preprocess -d 23 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8 # Preprocess dataset

exit