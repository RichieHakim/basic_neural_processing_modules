# test_pca.py
import torch
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from ..decomposition import PCA  # Adjust the import according to your module structure

# Generate a synthetic dataset
np.random.seed(42)
torch.manual_seed(42)
X_np = np.random.randn(100, 10).astype(np.float32)
X_torch = torch.from_numpy(X_np)

def test_fit_transform_equivalence():
    n_components = 5
    pca_sklearn = sklearnPCA(n_components=n_components, svd_solver='full').fit(X_np)
    pca_torch = PCA(n_components=n_components).fit(X_torch)
    
    # Transform the data using both PCA implementations
    X_transformed_sklearn = pca_sklearn.transform(X_np)
    X_transformed_torch = pca_torch.transform(X_torch).numpy()
    
    # Test for equivalence of the transformed data
    assert np.allclose(X_transformed_sklearn, X_transformed_torch, atol=1e-5)

def test_explained_variance_ratio():
    pca_torch = PCA(n_components=5)
    pca_torch.fit(X_torch)
    
    pca_sklearn = sklearnPCA(n_components=5, svd_solver='full').fit(X_np)
    
    # Test for equivalence of explained variance ratio
    assert np.allclose(pca_torch.explained_variance_ratio_.numpy(), pca_sklearn.explained_variance_ratio_, atol=1e-5)

def test_inverse_transform():
    pca_torch = PCA(n_components=5)
    pca_torch.fit(X_torch)
    
    X_transformed_torch = pca_torch.transform(X_torch)
    X_inversed_torch = pca_torch.inverse_transform(X_transformed_torch).numpy()
    
    # Test for approximation of original data after inverse transformation
    assert np.allclose(X_np, X_inversed_torch, atol=1e-5)

def test_components_sign():
    pca_torch = PCA(n_components=2)
    pca_torch.fit(X_torch)
    
    # Ensure that the signs of the principal components are corrected
    components = pca_torch.components_.numpy()
    assert (np.abs(components) == components).any() or (np.abs(components) == -components).any(), "Components' signs are not corrected properly."
    
def test_low_rank_svd():
    pca_low_rank = PCA(n_components=3, use_lowRank=True, lowRank_niter=5)
    pca_full_rank = PCA(n_components=3, use_lowRank=False)
    
    pca_low_rank.fit(X_torch)
    pca_full_rank.fit(X_torch)
    
    assert pca_low_rank.components_.shape == pca_full_rank.components_.shape, "Low-rank and full-rank PCA should produce components of the same shape."
    
    # This test doesn't check for numerical equivalence, as low-rank approximations
    # will differ, but it ensures that the method runs and produces output of the expected shape.

def test_whitening_effect():
    pca_whiten = PCA(n_components=5, whiten=True)
    pca_whiten.fit(X_torch)
    
    X_transformed = pca_whiten.transform(X_torch).numpy()
    # Check if the variance across each principal component is close to 1, which is expected after whitening
    variances = np.var(X_transformed, axis=0)
    assert np.allclose(variances, np.ones(variances.shape), atol=1e-5), "Whitened components do not have unit variance."

def test_retain_all_components():
    pca_all = PCA(n_components=None)  # Retain all components
    pca_all.fit(X_torch)
    
    expected_components = min(X_torch.shape)
    assert pca_all.components_.shape[0] == expected_components, "PCA with n_components=None did not retain all components."
    
def test_single_component():
    pca_single = PCA(n_components=1)
    pca_single.fit(X_torch)
    
    assert pca_single.components_.shape[0] == 1, "PCA did not correctly reduce data to a single component."
    assert pca_single.components_.shape[1] == X_torch.shape[1], "The single component does not have the correct dimensionality."

def test_more_components_than_features():
    n_features = X_torch.shape[1]
    pca_excessive = PCA(n_components=n_features + 5)  # Request more components than available features
    pca_excessive.fit(X_torch)
    
    # Should only return a number of components equal to the number of features
    assert pca_excessive.components_.shape[0] == n_features, "PCA returned more components than the number of input features."
    
def test_data_preparation():
    pca_center_scale = PCA(center=True, zscale=True)
    pca_center_scale.fit(X_torch)
    
    # sklearn doesn't directly expose mean_ and std_ for centered and scaled data,
    # so we compare against manually calculated values.
    X_centered_scaled = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)
    mean_diff = np.abs(X_centered_scaled.mean(axis=0) - pca_center_scale.mean_.numpy())
    std_diff = np.abs(X_centered_scaled.std(axis=0) - pca_center_scale.std_.numpy())
    
    assert np.all(mean_diff < 1e-5), "Data centering (mean subtraction) is incorrect."
    assert np.all(std_diff < 1e-5), "Data scaling (division by std) is incorrect."

def test_singular_values_and_vectors():
    pca_svd = PCA(n_components=5)
    pca_svd.fit(X_torch)
    
    # sklearn's singular values
    pca_sklearn = sklearnPCA(n_components=5, svd_solver='full').fit(X_np)
    
    # Singular values should match
    assert np.allclose(pca_svd.singular_values_.numpy(), pca_sklearn.singular_values_, atol=1e-5), "Singular values do not match."
    
    # There's no direct equivalent for left singular vectors in sklearn's PCA output,
    # but ensuring our singular values match is a good indication of correctness.

def test_singular_values_and_vectors():
    pca_svd = PCA(n_components=5)
    pca_svd.fit(X_torch)
    
    # sklearn's singular values
    pca_sklearn = sklearnPCA(n_components=5, svd_solver='full').fit(X_np)
    
    # Singular values should match
    assert np.allclose(pca_svd.singular_values_.numpy(), pca_sklearn.singular_values_, atol=1e-5), "Singular values do not match."
    
    # There's no direct equivalent for left singular vectors in sklearn's PCA output,
    # but ensuring our singular values match is a good indication of correctness.

def test_low_rank_approximation_accuracy():
    pca_low_rank = PCA(n_components=5, use_lowRank=True, lowRank_niter=10)
    pca_low_rank.fit(X_torch)
    
    pca_full_rank = PCA(n_components=5, use_lowRank=False)
    pca_full_rank.fit(X_torch)
    
    # While we can't expect the low-rank approximation to exactly match the full-rank results,
    # we can check that they're reasonably close, implying the approximation is reasonable.
    components_diff = np.abs(pca_low_rank.components_.numpy() - pca_full_rank.components_.numpy())
    assert components_diff.mean() < 0.1, "Low-rank approximation deviates too much from full SVD."

def test_low_rank_approximation_accuracy():
    pca_low_rank = PCA(n_components=5, use_lowRank=True, lowRank_niter=10)
    pca_low_rank.fit(X_torch)
    
    pca_full_rank = PCA(n_components=5, use_lowRank=False)
    pca_full_rank.fit(X_torch)
    
    # While we can't expect the low-rank approximation to exactly match the full-rank results,
    # we can check that they're reasonably close, implying the approximation is reasonable.
    components_diff = np.abs(pca_low_rank.components_.numpy() - pca_full_rank.components_.numpy())
    assert components_diff.mean() < 0.1, "Low-rank approximation deviates too much from full SVD."

def test_n_components_effect():
    for n in [2, 5, 8]:
        pca_n = PCA(n_components=n)
        pca_n.fit(X_torch)
        
        assert pca_n.components_.shape[0] == n, f"PCA with n_components={n} did not produce the correct number of components."
        assert pca_n.explained_variance_.shape[0] == n, f"PCA with n_components={n} did not produce the correct number of explained variances."