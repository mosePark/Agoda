import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 1. Read the CSV data
# Reading the data from a CSV file into a pandas DataFrame.
data = pd.read_csv(".../0.7-all.csv")

# 2. Define the make_matrix function
# Converts a vector of strings (each representing a list of numbers) into a numeric NumPy array (matrix).
def make_matrix(xvec):
    n_rows = len(xvec)
    n_cols = 1536  # Assuming the embeddings have 1536 dimensions
    return_mat = np.empty((n_rows, n_cols))
    for i, x in enumerate(xvec):
        # Remove square brackets from the string
        x_clean = x.replace('[', '').replace(']', '')
        # Convert the string to a numeric array
        x_numeric = np.fromstring(x_clean, sep=',')
        return_mat[i, :] = x_numeric
    return return_mat

# 3. Sample indices and generate embeddings
# Randomly sampling 500 indices and extracting embeddings for 'ori_ebd', 'gen1_ebd', and 'gen2_ebd'.
np.random.seed(7)
sample_ind = np.random.choice(data.index, size=500, replace=False)

or_emb = make_matrix(data.loc[sample_ind, 'ori_ebd'])
gen1_emb = make_matrix(data.loc[sample_ind, 'gen1_ebd'])
gen2_emb = make_matrix(data.loc[sample_ind, 'gen2_ebd'])

# 4. Dimensionality reduction using SVD
# Reducing the dimensionality of each embedding to two dimensions using Singular Value Decomposition (SVD).
def reduce_dimensionality(emb):
    # Perform SVD
    U, s, Vt = np.linalg.svd(emb, full_matrices=False)
    # Multiply the embedding matrix by the first two right singular vectors
    red = np.dot(emb, Vt.T[:, :2])
    return red

red_or = reduce_dimensionality(or_emb)
red_gen1 = reduce_dimensionality(gen1_emb)
red_gen2 = reduce_dimensionality(gen2_emb)

# 5. Perform K-means clustering
# Clustering the reduced data into 4 clusters.
K = 4
or_clust = KMeans(n_clusters=K, random_state=0).fit(red_or)
gen1_clust = KMeans(n_clusters=K, random_state=0).fit(red_gen1)
gen2_clust = KMeans(n_clusters=K, random_state=0).fit(red_gen2)

# 6. Plot the clusters
# Plotting the two-dimensional data points and coloring them based on cluster labels.
plt.scatter(red_or[:, 0], red_or[:, 1], c=gen1_clust.labels_)
plt.title('Original Embeddings with gen1 Cluster Labels')
plt.show()

plt.scatter(red_gen1[:, 0], red_gen1[:, 1], c=gen1_clust.labels_)
plt.title('gen1 Embeddings with gen1 Cluster Labels')
plt.show()

plt.scatter(red_gen2[:, 0], red_gen2[:, 1], c=gen1_clust.labels_)
plt.title('gen2 Embeddings with gen1 Cluster Labels')
plt.show()

# 7. Compute adjusted Rand indices
# Measuring the similarity between clustering results using the Adjusted Rand Index.
ari_gen1_gen2 = adjusted_rand_score(gen1_clust.labels_, gen2_clust.labels_)
ari_gen1_or = adjusted_rand_score(gen1_clust.labels_, or_clust.labels_)
print(f"Adjusted Rand index between gen1 and gen2: {ari_gen1_gen2}")
print(f"Adjusted Rand index between gen1 and original: {ari_gen1_or}")

# 8. Perform permutation test
# Implementing a permutation test to assess the statistical significance of the observed clustering similarity.
def permutation_test(dat1, dat2, itrN=500, K=4, k_near=5):
    N = dat1.shape[0]
    # Get cluster labels
    dat1_cl = KMeans(n_clusters=K, random_state=0).fit(dat1).labels_
    dat2_cl = KMeans(n_clusters=K, random_state=0).fit(dat2).labels_
    # Compute pairwise distances
    dat1_dist = cdist(dat1, dat1)
    dat2_dist = cdist(dat2, dat2)
    # Set self-distances to infinity to avoid selecting the same point
    np.fill_diagonal(dat1_dist, np.inf)
    np.fill_diagonal(dat2_dist, np.inf)
    # Get indices of k_near nearest neighbors
    dat1_near_mat = np.argsort(dat1_dist, axis=1)[:, :k_near]
    dat2_near_mat = np.argsort(dat2_dist, axis=1)[:, :k_near]
    rand1 = np.empty(itrN)
    rand2 = np.empty(itrN)
    for itr in range(itrN):
        new_dat1 = dat1.copy()
        new_dat2 = dat2.copy()
        TF_ind = np.random.choice([True, False], size=N)
        for i in range(N):
            if TF_ind[i]:
                # Calculate the mixing coefficients
                mix_coef1 = dat1_dist[i, dat1_near_mat[i]]
                mix_coef1 = mix_coef1 / np.sum(mix_coef1)
                mix_coef2 = dat2_dist[i, dat2_near_mat[i]]
                mix_coef2 = mix_coef2 / np.sum(mix_coef2)
                # Calculate the weighted average
                new_dat2_point = np.sum(mix_coef1[:, None] * dat2[dat1_near_mat[i]], axis=0)
                new_dat1_point = np.sum(mix_coef2[:, None] * dat1[dat2_near_mat[i]], axis=0)
                new_dat1[i] = new_dat1_point
                new_dat2[i] = new_dat2_point
        # Cluster the new datasets
        new_cl_dat1 = KMeans(n_clusters=K, random_state=0).fit(new_dat1).labels_
        new_cl_dat2 = KMeans(n_clusters=K, random_state=0).fit(new_dat2).labels_
        # Compute adjusted Rand indices
        rand1[itr] = adjusted_rand_score(new_cl_dat1, dat2_cl)
        rand2[itr] = adjusted_rand_score(new_cl_dat2, dat1_cl)
    # Compute original adjusted Rand index
    rf = adjusted_rand_score(dat1_cl, dat2_cl)
    return {'ri1': rand1, 'ri2': rand2, 'rf': rf}

# Perform the permutation test
aa = permutation_test(red_gen1, red_gen2, itrN=500)

# 9. Compute the p-value
# Calculating the p-value based on the permutation test results.
p_value = np.mean((aa['ri1'] + aa['ri2']) * 0.5 < aa['rf'])
print(f"P-value: {p_value}")
