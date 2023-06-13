import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import numpy as np

img1= Image.open("dog.jpg").convert("L")
img_arr= np.array(img1)

from mlxtend.preprocessing import MeanCenterer
mc = MeanCenterer().fit(img_arr)
arr= np.array(mc.transform(img_arr))

covmat= np.cov(arr, rowvar=False)

val, vect= np.linalg.eig(covmat)
index_array= np.argsort(val)[::-1]
val_r= np.real(val[index_array])
vect_r= np.real(vect[:, index_array])

vsum= np.abs(val_r.sum())
var_val=np.array([])
for elem in val_r:
    var_val= np.append(var_val, abs(elem)/vsum)

currsum=0
cumul_var_val= np.array([])
for elem in var_val:
    currsum+= elem
    cumul_var_val= np.append(cumul_var_val, currsum)

k=1
for elem in cumul_var_val:
    if elem>85:
        break
    k+=1

c=k
reduced_vect= vect_r[:, :c]
new_basis= np.dot(arr, reduced_vect)
reconstructed= np.matmul(new_basis ,np.transpose(reduced_vect))
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')
plt.show()
