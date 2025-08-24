import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from scipy.ndimage import center_of_mass
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    """Reorients the nifti from its original orientation to another specified orientation
    
    Parameters:
    ----------
    img: nibabel image
    axcodes_to: a tuple of 3 characters specifying the desired orientation
    
    Returns:
    ----------
    newimg: The reoriented nibabel image 
    
    """
    aff = img.affine
    ornt_fr = nio.io_orientation(aff)
    axcodes_fr = nio.ornt2axcodes(ornt_fr)
    if axcodes_to == axcodes_fr:
        return img
    ornt_to = nio.axcodes2ornt(axcodes_to)
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def process_subject(subj, base_dir, savepath_root, new_orientation=('L','A','S')):
    """Process a single subject - this function will be parallelized"""
    try:
        ct_path = os.path.join(base_dir, subj, "ct.nii.gz")
        seg_path = os.path.join(base_dir, subj, "segmentations", "liver.nii.gz")

        # Check if files exist
        if not os.path.exists(ct_path) or not os.path.exists(seg_path):
            return f"Skipped {subj}: Missing files"

        seg_nib = nib.load(seg_path)
        seg_arr = seg_nib.get_fdata(dtype=np.float32)
        
        if np.max(seg_arr) != 1:
            return f"Skipped {subj}: Max segmentation value != 1"

        img_nib = nib.load(ct_path)
        arr = img_nib.get_fdata(dtype=np.float32)

        # Get voxel spacing
        zooms = img_nib.header.get_zooms()[:3]

        # Get orientation codes from affine
        ornt = nio.io_orientation(img_nib.affine)
        axcodes = nio.ornt2axcodes(ornt)

        # Build a clean affine: diagonal with voxel spacing
        new_affine = np.diag(list(zooms) + [1])

        # Optionally, you can enforce a pure identity orientation *with voxel spacing*:
        new_affine[:3, :3] = np.eye(3) * np.array(zooms)

        # Reset origin so all images start at the same place
        new_affine[:3, 3] = 0

        # Save new NIfTI with minimal clean header
        new_img = nib.Nifti1Image(arr, affine=new_affine)
        new_seg = nib.Nifti1Image(seg_arr, affine=new_affine)

        img_nib_reoriented = reorient_to(new_img, axcodes_to=new_orientation)
        seg_nib_reoriented = reorient_to(new_seg, axcodes_to=new_orientation)

        savepath_img = os.path.join(savepath_root, f"{subj}_img.nii.gz")
        savepath_seg = os.path.join(savepath_root, f"{subj}_liver.nii.gz")

        nib.save(img_nib_reoriented, savepath_img)
        nib.save(seg_nib_reoriented, savepath_seg)

        return f"Processed {subj}: Success"
        
    except Exception as e:
        return f"Error processing {subj}: {str(e)}"


def main():
    # Directories ubuntu
    # base_dir = "/home/awias/data/Totalsegmentator_dataset_v201"
    # savepath_root = "/home/awias/data/Totalsegmentator_dataset_v201_filtered_liver"

    # Directories titans
    base_dir = "/scratch/awias/data/Totalsegmentator_dataset_v201"
    savepath_root = "/scratch/awias/data/Totalsegmentator_dataset_v201_filtered_liver"

    os.makedirs(savepath_root, exist_ok=True)

    # Get all subject folders (s0000, s0001, ...)
    subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("s")])
    
    # For testing with a subset
    # subjects = ['s0001']

    # Determine number of processes to use (use all available CPUs minus 1, or at least 1)
    num_processes = max(1, cpu_count() - 1)
    print(f"Processing {len(subjects)} subjects using {num_processes} processes...")

    # Create a partial function with fixed arguments
    process_func = partial(process_subject, 
                          base_dir=base_dir, 
                          savepath_root=savepath_root)

    # Process subjects in parallel
    with Pool(processes=num_processes) as pool:
        # Use imap for better memory efficiency with large datasets
        results = list(tqdm(pool.imap(process_func, subjects), 
                           total=len(subjects), 
                           desc="Processing subjects"))

    # Print results summary
    success_count = sum(1 for r in results if "Success" in r)
    error_count = sum(1 for r in results if "Error" in r)
    skip_count = sum(1 for r in results if "Skipped" in r)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    
    # Print any error messages
    if error_count > 0:
        print("\nError details:")
        for result in results:
            if "Error" in result:
                print(f"  {result}")


if __name__ == "__main__":
    main()