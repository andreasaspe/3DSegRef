import SimpleITK as sitk
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools

def get_direction_code(img_sitk):
    """ Get the direction of a SimpleITK image. """
    direction_code = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(img_sitk.GetDirection())
    return direction_code

def reorient_sitk(img_sitk, new_direction):
    """ Reorient a SimpleITK image to a new direction. """
    img_sitk_reoriented = sitk.DICOMOrient(img_sitk, new_direction)
    return img_sitk_reoriented

def process_single_subject(args):
    """Process a single subject - designed for multiprocessing"""
    subject, image_path, label_path, output_image_path, output_label_path, new_orientation_code = args
    
    # Input paths
    img_path = os.path.join(image_path, f"{subject}_0000.nii.gz")
    mask_path = os.path.join(label_path, f"{subject}.nii.gz")
    
    # Output paths
    output_img_path = os.path.join(output_image_path, f"{subject}_0000.nii.gz")
    output_mask_path = os.path.join(output_label_path, f"{subject}.nii.gz")
    
    try:
        # Check if input files exist
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            return f"Missing files for {subject}"
        
        # Load images
        image_sitk = sitk.ReadImage(img_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        # Get current direction
        direction_code = get_direction_code(image_sitk)
        
        # Reorient if needed
        if direction_code != new_orientation_code:
            image_sitk_reoriented = reorient_sitk(image_sitk, new_orientation_code)
            mask_sitk_reoriented = reorient_sitk(mask_sitk, new_orientation_code)
        else:
            image_sitk_reoriented = image_sitk
            mask_sitk_reoriented = mask_sitk
        
        # Convert mask to binary
        mask = sitk.GetArrayFromImage(mask_sitk_reoriented).astype(np.float32)
        mask[mask > 0] = 1  # Convert all non-zero labels to 1
        
        mask_sitk_reoriented_and_binary = sitk.GetImageFromArray(mask)
        mask_sitk_reoriented_and_binary.CopyInformation(mask_sitk_reoriented)
        
        # Save images to output paths
        sitk.WriteImage(image_sitk_reoriented, output_img_path)
        sitk.WriteImage(mask_sitk_reoriented_and_binary, output_mask_path)
        
        return f"Successfully processed {subject}"
        
    except Exception as e:
        return f"Error processing {subject}: {str(e)}"

def process_dataset_parallel(image_path, label_path, output_image_path, output_label_path, subjects, new_orientation_code='LAS', n_workers=None):
    """Process dataset in parallel"""
    
    if n_workers is None:
        n_workers = min(cpu_count(), len(subjects))  # Don't create more workers than subjects
    
    # Prepare arguments for multiprocessing
    args = [(subject, image_path, label_path, output_image_path, output_label_path, new_orientation_code) for subject in subjects]
    
    print(f"Processing {len(subjects)} subjects with {n_workers} workers...")
    
    # Use multiprocessing Pool
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_single_subject, args), total=len(subjects)))
    
    # Print results summary
    success_count = sum(1 for r in results if "Successfully processed" in r)
    error_count = len(results) - success_count
    
    print(f"Processing complete: {success_count} successful, {error_count} errors")
    
    # Print errors if any
    if error_count > 0:
        print("\nErrors:")
        for result in results:
            if "Error" in result or "Missing" in result:
                print(f"  {result}")

# Main execution
if __name__ == "__main__":
    # Input paths
    image_tr_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/before_preprocessing/imagesTr"
    label_tr_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/before_preprocessing/labelsTr"
    image_ts_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/before_preprocessing/imagesTs"
    label_ts_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/before_preprocessing/labelsTs"
    
    # Output paths
    output_image_tr_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/imagesTr"
    output_label_tr_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/labelsTr"
    output_image_ts_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/imagesTs"
    output_label_ts_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/labelsTs"
    
    # Create output directories
    os.makedirs(output_image_tr_path, exist_ok=True)
    os.makedirs(output_label_tr_path, exist_ok=True)
    os.makedirs(output_image_ts_path, exist_ok=True)
    os.makedirs(output_label_ts_path, exist_ok=True)
    
    # Get subjects
    all_subjects_tr = [x.split('_')[0] for x in os.listdir(image_tr_path) if x.endswith(".nii.gz")]
    all_subjects_ts = [x.split('_')[0] for x in os.listdir(image_ts_path) if x.endswith(".nii.gz")]
    
    # Process training set
    print("Processing training set...")
    process_dataset_parallel(
        image_tr_path, label_tr_path, 
        output_image_tr_path, output_label_tr_path, 
        all_subjects_tr, n_workers=8
    )
    
    # Process test set  
    print("\nProcessing test set...")
    process_dataset_parallel(
        image_ts_path, label_ts_path,
        output_image_ts_path, output_label_ts_path,
        all_subjects_ts, n_workers=8
    )