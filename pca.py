from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np

img= Image.open("dog.jpg").convert("L")
img_arr= np.array(img)
avg = np.mean(img_arr, axis=0)

from mlxtend.preprocessing import MeanCenterer
mc = MeanCenterer().fit(img_arr)
img_arr= np.array(mc.transform(img_arr))

import matplotlib.pyplot as plt
plt.imshow(img_arr, cmap="gray")
plt.axis("off")
plt.show()

pca= PCA(n_components=19)
pca.fit(img_arr)

vectspace= pca.components_.T
new_basis= np.matmul(img_arr, vectspace)
recons= np.matmul(new_basis, vectspace.T)
recons = recons + avg

plt.imshow(recons, cmap="gray")
plt.axis("off")
plt.show()
