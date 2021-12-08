import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

dataset = datasets.load_digits()
digits = dataset.images.reshape(len(dataset.images), 64)

pca = PCA(n_components=64)
pca.fit(digits)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# plt.plot(range(0, 64), cumulative_explained_variance)
# plt.ylabel('cumulative explained variance')
# plt.xlabel('number of variables')
# plt.show()

pca40 = PCA(n_components=40)
pca40.fit(digits)
digits_40_components = pca40.transform(digits)
print(digits_40_components.shape)

#
# def get_gaussian_mixture_bic(n_components):
#     gm = GaussianMixture(n_components, random_state=1)
#     gm.fit(digits_40_components)
#     gm_bic = gm.bic(digits_40_components)
#     # print(n_components, gm_bic)
#     return gm_bic
#
#
# bics = [get_gaussian_mixture_bic(i) for i in range(1, 200, 5)]
# plt.plot(range(1, 200, 5), bics)
# plt.show()

gm63 = GaussianMixture(63, random_state=1)
gm63.fit(digits_40_components)
sample = gm63.sample(1)

sample_64 = pca40.inverse_transform(sample[0])
print(sample_64.shape)

img = np.reshape(sample_64, (8, 8))
print(img.shape)

plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

