import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def process_subject(subject, path_deterministic, path_stochastic_PPT, path_stochastic_multibasis, path_img, outputpath):
    """Process a single subject's probability data"""
    try:
        # Load probability maps
        probs_deterministic = np.load(os.path.join(path_deterministic, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class
        probs_multibasis = np.load(os.path.join(path_stochastic_multibasis, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class
        probs_PPT = np.load(os.path.join(path_stochastic_PPT, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class

        img_sitk = sitk.ReadImage(os.path.join(path_img, subject + "_0000.nii.gz")) # Only for copying information
        
        # Calculate probability differences
        probs_difference_multibasis = probs_deterministic - probs_multibasis
        probs_difference_PPT = probs_deterministic - probs_PPT
        
        # Save deterministic probabilities
        probs_deterministic_sitk = sitk.GetImageFromArray(probs_deterministic)
        probs_deterministic_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(probs_deterministic_sitk, os.path.join(outputpath, subject + "_probs_deterministic.nii.gz"))
        
        # Save multibasis probabilities
        probs_multibasis_sitk = sitk.GetImageFromArray(probs_multibasis)
        probs_multibasis_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(probs_multibasis_sitk, os.path.join(outputpath, subject + "_probs_multibasis.nii.gz"))
        
        # Save PPT probabilities
        probs_PPT_sitk = sitk.GetImageFromArray(probs_PPT)
        probs_PPT_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(probs_PPT_sitk, os.path.join(outputpath, subject + "_probs_PPT.nii.gz"))
        
        # Save multibasis difference
        probs_difference_multibasis_sitk = sitk.GetImageFromArray(probs_difference_multibasis)
        probs_difference_multibasis_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(probs_difference_multibasis_sitk, os.path.join(outputpath, subject + "_probs_difference_multibasis.nii.gz"))
        
        # Save PPT difference
        probs_difference_PPT_sitk = sitk.GetImageFromArray(probs_difference_PPT)
        probs_difference_PPT_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(probs_difference_PPT_sitk, os.path.join(outputpath, subject + "_probs_difference_PPT.nii.gz"))
        
        # Create summary statistics for logging
        stats = {
            'deterministic': {'shape': probs_deterministic.shape, 'min': np.min(probs_deterministic), 
                            'max': np.max(probs_deterministic), 'mean': np.mean(probs_deterministic)},
            'multibasis': {'shape': probs_multibasis.shape, 'min': np.min(probs_multibasis), 
                         'max': np.max(probs_multibasis), 'mean': np.mean(probs_multibasis)},
            'PPT': {'shape': probs_PPT.shape, 'min': np.min(probs_PPT), 
                   'max': np.max(probs_PPT), 'mean': np.mean(probs_PPT)}
        }
        
        return f"SUCCESS: Processed {subject}", stats
        
    except Exception as e:
        return f"ERROR: Could not process {subject} - {str(e)}", None

def print_subject_stats(subject, stats):
    """Print statistics for a subject"""
    if stats:
        print(f"Subject {subject} statistics:")
        for method, data in stats.items():
            print(f"  {method}: shape={data['shape']}, min={data['min']:.6f}, max={data['max']:.6f}, mean={data['mean']:.6f}")
        print()

def main(num_processes=None, verbose=False):
    path_deterministic = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_deterministic"
    path_stochastic_PPT = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_stochastic_PPT"
    path_stochastic_multibasis = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_stochastic_multibasis"
    path_img = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs" # For copying image information
    outputpath = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/probs" # Output path

    os.makedirs(outputpath, exist_ok=True)

    all_subjects = [x.split('_')[0] for x in os.listdir(path_img) if x.endswith('.nii.gz')]
    all_subjects = sorted(all_subjects, key=lambda x: int(x))

    print(f"Found {len(all_subjects)} subjects to process")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    num_processes = min(num_processes, len(all_subjects))  # Don't use more processes than subjects
    print(f"Using {num_processes} processes")

    # Create partial function with fixed arguments
    process_func = partial(process_subject,
                          path_deterministic=path_deterministic,
                          path_stochastic_PPT=path_stochastic_PPT,
                          path_stochastic_multibasis=path_stochastic_multibasis,
                          path_img=path_img,
                          outputpath=outputpath)

    # Process subjects in parallel
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking with tqdm
        results = list(tqdm(
            pool.imap(process_func, all_subjects), 
            total=len(all_subjects),
            desc="Processing subjects"
        ))

    # Process results and print statistics if verbose
    successful = 0
    failed = 0
    
    for i, (result_msg, stats) in enumerate(results):
        subject = all_subjects[i]
        
        if result_msg.startswith("SUCCESS"):
            successful += 1
            if verbose and stats:
                print_subject_stats(subject, stats)
        else:
            failed += 1
            if verbose:
                print(result_msg)

    # Print summary
    print(f"\nProcessing complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Print error messages for failed subjects
    if failed > 0:
        print("\nError details:")
        for result_msg, _ in results:
            if result_msg.startswith("ERROR"):
                print(result_msg)

if __name__ == "__main__":
    # You can specify the number of processes and verbosity here
    # main(num_processes=4, verbose=True)  # Use 4 processes with verbose output
    main(verbose=False)  # Use automatic detection with minimal output