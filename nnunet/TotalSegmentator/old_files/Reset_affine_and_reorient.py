import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from scipy.ndimage import center_of_mass
import os
import numpy as np
from tqdm import tqdm

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



# Directories ubuntu
# base_dir = "/home/awias/data/Totalsegmentator_dataset_v201"
# savepath_root = "/home/awias/data/Totalsegmentator_dataset_v201_filtered_liver"

# Directories titans
base_dir = "/scratch/awias/data/Totalsegmentator_dataset_v201"
savepath_root = "/scratch/awias/data/Totalsegmentator_dataset_v201_filtered_pancreas"

os.makedirs(savepath_root, exist_ok=True)

# Get all subject folders (s0000, s0001, ...)
subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("s")])

idx = 0

# subjects = ['s0001']

for subj in tqdm(subjects):
    ct_path = os.path.join(base_dir, subj, "ct.nii.gz")
    seg_path = os.path.join(base_dir, subj, "segmentations", "humerus.nii.gz")

    seg_nib = nib.load(seg_path)
    seg_arr = seg_nib.get_fdata(dtype=np.float32)
    
    if np.max(seg_arr) != 1:
        print(f"Skipping {subj}: Max segmentation value != 1")
        continue
    
    continue

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
    # new_img.header.set_xyzt_units('mm')
    # new_seg.header.set_xyzt_units('mm')

    new_orientation = ('L','A','S')  # Desired orientation

    img_nib_reoriented = reorient_to(new_img, axcodes_to=new_orientation)
    seg_nib_reoriented = reorient_to(new_seg, axcodes_to=new_orientation)

    savepath_img = os.path.join(savepath_root, f"{subj}_img.nii.gz")
    savepath_seg = os.path.join(savepath_root, f"{subj}_seg.nii.gz")

    nib.save(img_nib_reoriented, savepath_img)
    nib.save(seg_nib_reoriented, savepath_seg)
