import argparse
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json
import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import label
from pathlib import Path
import os

def predict_with_nn_unet_on_filelist():
    print("Predict with NN U-Net")
    model_folder = "/scratch/awias/data/nnUNet/nnUNet_results/Dataset006_TotalSegmentatorGallbladder/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres"

    input_data_folder = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset006_TotalSegmentatorGallbladder/imagesTs"
    output_folder = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset006_TotalSegmentatorGallbladder/imagesTs/man_preds"

    os.environ["nnUNet_results"] = "/scratch/awias/data/nnUNet_dataset/nnUNet_results"
 
    os.makedirs(output_folder, exist_ok=True)

    in_files = []
    out_files = []
    
    for file in os.listdir(input_data_folder):
        if file.endswith("nii.gz"):
            subject = file.split(".nii.gz")[0]
            in_files.append([os.path.join(input_data_folder, file)])
            out_files.append(os.path.join(output_folder, subject + "_pred.nii.gz"))
        
    print(f"Initializing class")
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    print(f"Initializing from trained model folder")

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=[0],
        checkpoint_name='checkpoint_latest.pth',
    )

    print(f"Predicting from files")

    predictor.predict_from_files(in_files,
                                     out_files,
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)






if __name__ == '__main__':
    args = argparse.ArgumentParser(description='dtu-predict-with-nn-unet')
    # config = DTUConfig(args)
    # if config.settings is not None:
    #     # predict_with_nn_unet(config)

    predict_with_nn_unet_on_filelist()
