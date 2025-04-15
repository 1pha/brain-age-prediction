import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def linear_CKA_with_nan(X, Y):
    # NaN이 있는 위치 마스크 생성
    mask = ~(np.isnan(X) | np.isnan(Y))
    
    # 마스크 적용하여 NaN 제거
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    # 데이터가 너무 적으면 NaN 반환
    if len(X_filtered) < 2:
        return np.nan
    
    # 중심화
    X_centered = X_filtered - X_filtered.mean()
    Y_centered = Y_filtered - Y_filtered.mean()
    
    # CKA 계산
    XTX = X_centered.T @ X_centered
    YTY = Y_centered.T @ Y_centered
    XTY = X_centered.T @ Y_centered
    
    hsic = np.sum(XTY * XTY)
    normalization = np.sqrt(np.sum(XTX * XTX) * np.sum(YTY * YTY))
    
    return hsic / normalization


def kernel_CKA_with_nan(X, Y, sigma=1.0):
    # NaN이 있는 위치 마스크 생성
    mask = ~(np.isnan(X) | np.isnan(Y))
    
    # 마스크 적용하여 NaN 제거
    X_filtered = X[mask].reshape(-1, 1)
    Y_filtered = Y[mask].reshape(-1, 1)
    
    # 데이터가 너무 적으면 NaN 반환
    if len(X_filtered) < 2:
        return np.nan
    
    # 커널 계산
    K = rbf_kernel(X_filtered, gamma=1/(2*sigma**2))
    L = rbf_kernel(Y_filtered, gamma=1/(2*sigma**2))
    
    # 커널 중심화
    K_centered = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()
    L_centered = L - L.mean(axis=0) - L.mean(axis=1)[:, np.newaxis] + L.mean()
    
    # HSIC 계산
    hsic = np.sum(K_centered * L_centered)
    normalization = np.sqrt(np.sum(K_centered * K_centered) * np.sum(L_centered * L_centered))
    
    return hsic / normalization


def rsa_correlation_with_nan(X, Y):
    # NaN이 있는 위치 마스크 생성
    mask = ~(np.isnan(X) | np.isnan(Y))
    
    # 마스크 적용하여 NaN 제거
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    # 데이터가 너무 적으면 NaN 반환
    if len(X_filtered) < 3:  # 최소 3개 이상의 점이 있어야 의미 있는 RDM 계산 가능
        return np.nan, np.nan
    
    # NaN 없는 데이터로 RDM 계산
    rdm_X = squareform(pdist(X_filtered.reshape(-1, 1), metric='euclidean'))
    rdm_Y = squareform(pdist(Y_filtered.reshape(-1, 1), metric='euclidean'))
    
    # 상삼각행렬만 선택
    triu_indices = np.triu_indices_from(rdm_X, k=1)
    rdm_X_vec = rdm_X[triu_indices]
    rdm_Y_vec = rdm_Y[triu_indices]
    
    # Spearman 상관계수 계산
    corr, p = spearmanr(rdm_X_vec, rdm_Y_vec)
    return corr, p
