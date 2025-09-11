import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def entropy(probs):
    """Calculate entropy of probability maps."""
    # probs: (C,H,W,D)
    eps = 1e-10  # to avoid log(0)
    H = -np.sum(probs * np.log2(probs + eps), axis=0)  # sum over channel dimension
    return H  # shape: (H,W,D)


def process_subject(subject, path_stochastic, path_deterministic, path_img, path_entropy):
    """Process a single subject - calculate and save entropy maps."""
    try:
        # Load probability maps
        probs_stochastic = np.load(os.path.join(path_stochastic, subject + "_0000_pred.nii.gz.npz"))['probabilities']
        probs_deterministic = np.load(os.path.join(path_deterministic, subject + "_0000_pred.nii.gz.npz"))['probabilities']
        
        # Calculate entropy maps
        entropy_stochastic = entropy(probs_stochastic)
        entropy_deterministic = entropy(probs_deterministic)
        
        # Calculate difference (positive values indicate higher uncertainty in stochastic model)
        diff_entropy = entropy_stochastic - entropy_deterministic
        
        # Load original image for copying metadata
        img_sitk = sitk.ReadImage(os.path.join(path_img, subject + "_0000.nii.gz"))
        
        # Save entropy difference
        entropy_diff_sitk = sitk.GetImageFromArray(diff_entropy)
        entropy_diff_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(entropy_diff_sitk, os.path.join(path_entropy, subject + "_entropy-diff.nii.gz"))

        # Save stochastic entropy
        entropy_stochastic_sitk = sitk.GetImageFromArray(entropy_stochastic)
        entropy_stochastic_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(entropy_stochastic_sitk, os.path.join(path_entropy, subject + "_entropy-stochastic.nii.gz"))

        # Save deterministic entropy
        entropy_deterministic_sitk = sitk.GetImageFromArray(entropy_deterministic)
        entropy_deterministic_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(entropy_deterministic_sitk, os.path.join(path_entropy, subject + "_entropy-deterministic.nii.gz"))
        
        return f"Successfully processed {subject}"
        
    except Exception as e:
        return f"Error processing {subject}: {str(e)}"


def main():
    # Define paths
    path_stochastic = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_stochastic"
    path_deterministic = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_deterministic"
    path_img = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs"
    path_entropy = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/entropy"
    
    # Create output directory
    os.makedirs(path_entropy, exist_ok=True)
    
    # Get all subjects
    all_subjects = [x.split('_')[0] for x in os.listdir(path_img) if x.endswith('.nii.gz')]
    all_subjects = sorted(all_subjects, key=lambda x: int(x))
    
    print(f"Found {len(all_subjects)} subjects to process")
    
    # Determine number of processes to use
    n_processes = min(cpu_count(), 20)  # Don't use more processes than subjects
    print(f"Using {n_processes} parallel processes")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_subject,
        path_stochastic=path_stochastic,
        path_deterministic=path_deterministic,
        path_img=path_img,
        path_entropy=path_entropy
    )
    
    # Process subjects in parallel
    with Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, all_subjects),
            total=len(all_subjects),
            desc="Processing subjects"
        ))
    
    # Print results summary
    successful = sum(1 for r in results if "Successfully" in r)
    failed = len(results) - successful
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} subjects")
    print(f"Failed: {failed} subjects")
    
    # Print any error messages
    for result in results:
        if "Error" in result:
            print(result)


if __name__ == "__main__":
    main()