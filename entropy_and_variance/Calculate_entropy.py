import numpy as np
import SimpleITK as sitk

path_entropy = "/home/awias/data/11_0000_pred.nii.gz.npz"
path_img = "/home/awias/data/11_0000.nii.gz"

data = np.load(path_entropy)

# print(data['probabilities'])
print(data['probabilities'].shape)


img_sitk = sitk.ReadImage(path_img)
img = sitk.GetArrayFromImage(img_sitk)
print(img.shape)


data_channel0 = data['probabilities'][0]
data_channel1 = data['probabilities'][1]

entropy = - (data_channel0 * np.log2(data_channel0 + 1e-10) + data_channel1 * np.log2(data_channel1 + 1e-10))

sums = data_channel0 + data_channel1
print(sums.min(), sums.max())