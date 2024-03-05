import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
voxels_list = [array.tolist() for array in np.array_split(arr, len(arr))]

print(voxels_list)
