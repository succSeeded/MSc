#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


SEED = 42
np.random.seed(SEED)


# In[ ]:


np.set_printoptions(suppress=True)


# In[ ]:


# !curl -L -o diamonds-price-dataset.zip https://www.kaggle.com/api/v1/datasets/download/amirhosseinmirzaie/diamonds-price-dataset
# !unzip diamonds-price-dataset.zip


# # Dataset & feature selection

# In[ ]:


N = 400
df = pd.read_csv('diamonds.csv').sample(n=N, random_state=SEED)

print("Shape:", df.shape)
df.head()


# Features chosen: `x`, `y`, `z`, `table`, `depth` and `carat`. All of these features describe the size of a diamond.

# In[ ]:


data = df[['x', 'y', 'z', 'table', 'depth', 'carat']]

data.head()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# # Data standardization

# In[ ]:


means = data.mean(axis=0)
means


# In[ ]:


stds = data.std(axis=0)
stds


# In[ ]:


ranges = data.max(axis=0) - data.min(axis=0)
ranges


# In[ ]:


# Z-score normalization
data_zs = (data - means) / stds

# Range normalization
data_rd = (data - means) / ranges

# Ranking normalization
data_rn = 100.0 * (data - data.min(axis=0)) / ranges


# In[ ]:


# Performing singular value decomposition
uzs, szs, vzs = np.linalg.svd(data_zs)
urd, srd, vrd = np.linalg.svd(data_rd)
urn, srn, vrn = np.linalg.svd(data_rn)
uns, snss, vns = np.linalg.svd(data)

print("Singular values for z-score normalization:", szs.round(1))
print("Singular values for range normalization:", srd.round(1))
print("Singular values for ranking normalization:", srn.round(1))
print("Singular values for no standardization:", snss.round(1))


# In[ ]:


contrib_zs = np.square(szs)
contrib_rd = np.square(srd)
contrib_rn = np.square(srn)
contrib_ns = np.square(snss)

print("PC natural contributions (z-score normalization):", contrib_zs.round(2))
print("PC natural contributions (range normalization):", contrib_rd.round(2))
print("PC natural contributions (ranking normalization):", contrib_rn.round(2))
print("PC natural contributions (no standardization):", contrib_ns.round(2))


# In[ ]:


ds_zs = np.square(data_zs).to_numpy().sum()
ds_rd = np.square(data_rd).to_numpy().sum()
ds_rn = np.square(data_rn).to_numpy().sum()
ds_ns = np.square(data).to_numpy().sum()

print("Data scatter by definition (z-score normalization): %.4f" % ds_zs)
print("Data scatter by definition (range normalization): %.4f" % ds_rd)
print("Data scatter by definition (ranking normalization): %.4f" % ds_rn)
print("Data scatter by definition (no standardization): %.4f" % ds_ns)


# In[ ]:


ds_contib_zs = contrib_zs.sum()
ds_contib_rd = contrib_rd.sum()
ds_contib_rn = contrib_rn.sum()
ds_contib_ns = contrib_ns.sum()

print("Data scatter from contributions (z-score normalization): %.4f" % ds_contib_zs)
print("Data scatter from contributions (range normalization): %.4f" % ds_contib_rd)
print("Data scatter from contributions (ranking normalization): %.4f" % ds_contib_rn)
print("Data scatter from contributions (no standardization): %.4f" % ds_contib_ns)


# From the results one can see that the calculated data scatter is the same for all versions of standardization, which is consistent with the theory.

# In[ ]:


contrib_zs_percent = 100.0 * contrib_zs / ds_zs
contrib_rd_percent = 100.0 * contrib_rd / ds_rd
contrib_rn_percent = 100.0 * contrib_rn / ds_rn
contrib_ns_percent = 100.0 * contrib_ns / ds_ns

print("PC percentage contributions (z-score normalization):", contrib_zs_percent.round(2))
print("PC percentage contributions (range normalization):", contrib_rd_percent.round(2))
print("PC percentage contributions (ranking normalization):", contrib_rn_percent.round(2))
print("PC percentage contributions (no standardization):", contrib_ns_percent.round(2))


# In[ ]:


for i in range(6):
    x_arrstr = np.char.mod('%.3f', vrn[i].round(3))
    #combine to a string
    print(" & ".join(x_arrstr))


# In[ ]:


vzs


# Here we see that the contrast between principle component contributions is  starker for range normalization - 78% PC1 contribution for r-standardization vs. 67% PC1 contribution for z-standardization.
# 
# Ranking normalization brings PC1 contribution even further up to 89%, closer to the non-standardizaed contribution distribution.

# # Principal components visualization
# 
# For the purpose of visualizing two first principal components and the differences between them, let us use two features to define two separate groups of dataset instances: `depth` and `carat`.
# 
# The first group are diamonds with small depth ratios (< 58.5, as shown below). The second group are diamonds of large weight (> 2.1 carats, as shown below).

# ## Defining instance groups

# In[ ]:


sns.histplot(data['depth']).set_title('Depth distribution')


# In[ ]:


# Diamonds with small depth ratios
depth_group = df[df['depth'] < 58.5]
depth_group.shape


# In[ ]:


sns.histplot(data['carat']).set_title('Carat distribution')


# In[ ]:


# Diamonds with large weight
carat_group = df[df['carat'] > 2.1]
carat_group.shape


# In[ ]:


depth_group.index.intersection(carat_group.index)


# As one can see, the two groups do not overlap.

# In[ ]:


carat_group_indices = df.index.get_indexer(carat_group.index.tolist())
depth_group_indices = df.index.get_indexer(depth_group.index.tolist())


# ## Calculating principal components

# ### Z-score normalized data PCs

# In[ ]:


pcs1_zs = vzs[0]
pcs1_zs


# Larger loading components are positive, thus the sign for factors remains unchanged.

# In[ ]:


pc1_zs = uzs[:, 0] * np.sqrt(szs[0])


# In[ ]:


pcs2_zs = vzs[1]
pcs2_zs


# Larger loading components are negative, thus the sign for factors is changed to negative.

# In[ ]:


pc2_zs = uzs[:, 1] * np.sqrt(szs[1])


# ### Range normalized data PCs

# In[ ]:


pcs1_rd = vrd[0]
pcs1_rd


# Larger loading components are positive, thus the sign for factors remains unchanged.

# In[ ]:


pc1_rd = urd[:, 0] * np.sqrt(srd[0])


# In[ ]:


pcs2_rd = vrd[1]
pcs2_rd


# Larger loading components are negative, thus the sign for factors is changed to negative.

# In[ ]:


pc2_rd = -urd[:, 1] * np.sqrt(srd[1])


# ## Visualization

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(10,5))

axs[0].set_title('PCs - z-scoring SVD')
sns.scatterplot(x=pc1_zs, y=pc2_zs, s=80, color="black", marker=".", alpha=0.4, ax=axs[0])
sns.scatterplot(x=pc1_zs[carat_group_indices], y=pc2_zs[carat_group_indices], label='Carat > 2.1', ax=axs[0])
sns.scatterplot(x=pc1_zs[depth_group_indices], y=pc2_zs[depth_group_indices], label='Depth < 58.5', ax=axs[0])

axs[1].set_title('PCs - range normalization SVD')
sns.scatterplot(x=pc1_rd, y=pc2_rd, s=80, color="black", marker=".", alpha=0.4, ax=axs[1])
sns.scatterplot(x=pc1_rd[carat_group_indices], y=pc2_rd[carat_group_indices], label='Carat > 2.1', ax=axs[1])
sns.scatterplot(x=pc1_rd[depth_group_indices], y=pc2_rd[depth_group_indices], label='Depth < 58.5', ax=axs[1])


# On the resulting plots one can see that overall the visualization for both normalization techniques share many similarities in the overall distribution of instances.
# 
# As for the groups, it seems that larger weight in carat coincides with larger values of PC1, while smaller depth ratios coincide with lesser PC2 values. This follows from the observation that both groups seem to denote the outliers in the respective axes.

# # Conventional PCA

# In[ ]:


# The covariance matrix
cov = (data_zs.T @ data_zs) / (data_zs.shape[0] - 1)
cov


# In[ ]:


# Performing spectral decomposition
la, c = np.linalg.eig(cov)


# In[ ]:


print('Eigenvalues:', la)


# In[ ]:


print('Eigenvectors:', c)


# The eigenvalues and the corresponding eigen vectors are already correctly ordered. Let us now compute the first two principal components.

# In[ ]:


c1 = c[:, 0]

print('Eigenvector 1:', c1)


# In[ ]:


c2 = c[:, 1]

print('Eigenvector 2:', c2)


# Largest components in both eigenvectors are positive, thus we don't need to invert the signs on the corresponding factors.

# In[ ]:


z1 = (data_zs @ c1) / np.sqrt(la[0] * (data_zs.shape[0] - 1))
z2 = (data_zs @ c2) / np.sqrt(la[1] * (data_zs.shape[0] - 1))


# In[ ]:


z1 = z1.to_numpy()
z2 = z2.to_numpy()


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(15,5))

axs[0].set_title('PCs - z-scoring conventional PCA')
sns.scatterplot(x=z1, y=z2, s=80, color=".2", marker=".", ax=axs[0], alpha=0.4)
sns.scatterplot(x=z1[carat_group_indices], y=z2[carat_group_indices], ax=axs[0], label='Carat > 2.1')
sns.scatterplot(x=z1[depth_group_indices], y=z2[depth_group_indices], ax=axs[0], label='Depth < 58.5')

axs[1].set_title('PCs - z-scoring SVD')
sns.scatterplot(x=pc1_zs, y=pc2_zs, s=80, color=".2", marker=".", ax=axs[1], alpha=0.4)
sns.scatterplot(x=pc1_zs[carat_group_indices], y=pc2_zs[carat_group_indices], ax=axs[1], label='Carat > 2.1')
sns.scatterplot(x=pc1_zs[depth_group_indices], y=pc2_zs[depth_group_indices], ax=axs[1], label='Depth < 58.5')

axs[2].set_title('PCs - range normalization SVD')
sns.scatterplot(x=pc1_rd, y=pc2_rd, s=80, color=".2", marker=".", ax=axs[2], alpha=0.4)
sns.scatterplot(x=pc1_rd[carat_group_indices], y=pc2_rd[carat_group_indices], ax=axs[2], label='Carat > 2.1')
sns.scatterplot(x=pc1_rd[depth_group_indices], y=pc2_rd[depth_group_indices], ax=axs[2], label='Depth < 58.5')


# The resulting figures are very similar, the differences in coordinates are mostly caused from the use of different normalization techniques.

# # Hidden ranking factor

# In[ ]:


pcs1_rn = vrn[0]
pcs1_rn


# Larger loading components are negative, thus the sign for factors is changed to negative.

# In[ ]:


pcs1_rn = -vrn[0]
pcs1_rn


# In[ ]:


pc1_rn = -urn[:, 0]


# In[ ]:


# Percentage scaling
pc1_rn_m = 100 * pc1_rn / np.max(pc1_rn)


# In[ ]:


sns.histplot(pc1_rn_m).set_title("Hidden factors, scaled to 0-100")


# Now, let's take the best instances in terms of their hidden factor:

# In[352]:


sub_df = df.loc[data.index.tolist()][['x', 'y', 'z', 'carat', 'table', 'depth']]


# In[353]:


sub_df['score'] = pc1_rn_m


# In[354]:


sub_df.sort_values('score', ascending=False)[:20]


# Looking at the table, it is easy to see that the score is naturally correlated with diamond's dimensions and weight. Intuitively it follows that the hidden factor describes how large a diamond is.
