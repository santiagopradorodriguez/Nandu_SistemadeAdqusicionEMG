# ==============================================================================
# Archivo: EMG_desarrollo/core/latent_bci_utils.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Descripción: Rechazo de artefactos (Mahalanobis), alineación rápida inter-sesión
#              (Procrustes) y suavizado de trayectorias latentes (Kalman).
# ==============================================================================

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.linalg import orthogonal_procrustes


class LatentSpaceBCIGuard:
    """
    Rechaza pulsos fuera de distribución (deglución, tos, movimiento mandibular)
    usando el umbral Chi-cuadrado al 99% de la distancia de Mahalanobis respecto
    a la covarianza del cluster GMM ganador.
    """
    CHI2_99 = {2: 9.210, 3: 11.345, 4: 13.277, 8: 20.090}

    def __init__(self, latent_dim: int = 3):
        self.latent_dim = int(latent_dim)
        self.rot_matrix: np.ndarray = np.eye(self.latent_dim)
        self.translation: np.ndarray = np.zeros(self.latent_dim)

    def fit_few_shot_alignment(self, z_new_centroids: np.ndarray, z_ref_centroids: np.ndarray) -> None:
        mu_new = np.mean(z_new_centroids, axis=0)
        mu_ref = np.mean(z_ref_centroids, axis=0)
        R, _ = orthogonal_procrustes(z_new_centroids - mu_new, z_ref_centroids - mu_ref)
        self.rot_matrix = R
        self.translation = mu_ref - (mu_new @ R)

    def transform_point(self, z: np.ndarray) -> np.ndarray:
        return (z @ self.rot_matrix) + self.translation

    def check_mahalanobis_ood(
        self,
        z: np.ndarray,
        mean_k: np.ndarray,
        cov_k: np.ndarray,
        custom_threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        diff = z - mean_k
        try:
            inv_cov = np.linalg.pinv(cov_k)
            d2 = float(diff.T @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            d2 = float(np.sum(diff ** 2))

        limit = custom_threshold if custom_threshold is not None else self.CHI2_99.get(self.latent_dim, 11.345)
        return (d2 <= limit), d2


class LatentKalmanTracker:
    """Filtro de Kalman cinemático para suavizar coordenadas latentes en tiempo real."""
    def __init__(self, dim: int = 3, dt: float = 0.03, process_noise: float = 1e-2, measurement_noise: float = 5e-2):
        self.dim = int(dim)
        self.state = np.zeros(2 * self.dim)
        self.P = np.eye(2 * self.dim)
        self.F = np.eye(2 * self.dim)
        for i in range(self.dim):
            self.F[i, self.dim + i] = dt
        self.H = np.zeros((self.dim, 2 * self.dim))
        for i in range(self.dim):
            self.H[i, i] = 1.0
        self.Q = np.eye(2 * self.dim) * process_noise
        self.R = np.eye(self.dim) * measurement_noise

    def update(self, z_obs: np.ndarray) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z_obs - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.state = self.state + (K @ y)
        self.P = (np.eye(2 * self.dim) - K @ self.H) @ self.P
        return self.state[:self.dim].copy()
