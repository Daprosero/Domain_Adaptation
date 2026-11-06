"""Cauchy-Schwarz estimators of Yin et al. (UAI 2024, arXiv 2405.19978v1).

Eq. 11 (Prop. 5) is the marginal CS divergence between the bag
representations of the two domains; Eq. 13 (Prop. 6, the symmetric estimator
the appendix derives as Eq. 86 and repeats as Eq. 88) is the conditional CS
divergence between the predictive distributions p(y_hat | z). Both use the
Gaussian kernel exp(-||a - b||^2 / (2 sigma^2)) stated under Eq. 11. Its
normalising constants are omitted because they cancel: Eq. 11 combines its three
log terms with coefficients +1, +1, -2, and Eq. 13 its four with +1, +1, -1, -1.

Implementation choice the paper does not make: every sum of kernel values is
evaluated in the log domain (`torch.logsumexp` over the log-kernel), which is
the same number as the direct transcription wherever the direct sums do not
underflow, and stays finite where they do (two far-apart sets, whose
cross kernel underflows to 0 and whose log turns to -inf in the direct form).
See REPLICATION.md.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["7.1", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "cs_divergence_has_closed_form_on_single_points",
        "ccs_divergence_has_closed_form_on_single_points",
        "divergences_vanish_on_identical_samples",
        "cs_divergence_is_nonnegative",
        "ccs_estimator_matches_eq_13_transcription_and_is_symmetric",
        "divergences_stay_finite_where_direct_sums_underflow",
        "kernel_inputs_are_l2_normalised",
        "median_bandwidth_is_multiplier_times_positive_median",
    ],
}

#: The paper's sigma (Sec. 4.2), used only when a median is undefined.
PAPER_SIGMA = 1.0


def _check_width(sigma: float) -> float:
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"kernel width must be positive and finite, got {sigma!r}")
    return sigma


def normalise(x: torch.Tensor) -> torch.Tensor:
    """Sec. 4.2 "we normalize z and y_hat": rows to unit L2 norm, (n, d) -> (n, d)."""
    if x.ndim != 2:
        raise ValueError("rows of a (samples, features) matrix are normalised")
    return F.normalize(x, p=2, dim=1)


def log_gaussian_gram(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    """log kappa_sigma(a_j, b_i) = -||a_j - b_i||^2 / (2 sigma^2): (n, d), (m, d) -> (n, m).

    Squared distance from the difference itself, so its gradient is exactly
    zero (never NaN) on coincident points, unlike the derivative of cdist.
    """
    sigma = _check_width(sigma)
    return -(a[:, None, :] - b[None, :, :]).pow(2).sum(-1) / (2 * sigma ** 2)


def _log_mean(log_values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_values.reshape(-1), 0) - math.log(log_values.numel())


def cs_divergence(zs: torch.Tensor, zt: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. 11: (M, d), (N, d) -> scalar.

    log(1/M^2 sum K^s) + log(1/N^2 sum K^t) - 2 log(1/(MN) sum K^st).
    """
    if zs.ndim != 2 or zt.ndim != 2 or zs.shape[1] != zt.shape[1] or not len(zs) or not len(zt):
        raise ValueError("two non-empty (samples, d) sets of equal width required")
    return (_log_mean(log_gaussian_gram(zs, zs, sigma))
            + _log_mean(log_gaussian_gram(zt, zt, sigma))
            - 2 * _log_mean(log_gaussian_gram(zs, zt, sigma)))


def _log_ratio_sum(log_k_joint: torch.Tensor, log_l_joint: torch.Tensor,
                   log_k_left: torch.Tensor, log_k_right: torch.Tensor) -> torch.Tensor:
    """log sum_j [ sum_i K_ji L_ji / ((sum_i Kleft_ji)(sum_i Kright_ji)) ], row j fixed."""
    numerator = torch.logsumexp(log_k_joint + log_l_joint, 1)
    return torch.logsumexp(numerator - torch.logsumexp(log_k_left, 1)
                           - torch.logsumexp(log_k_right, 1), 0)


def ccs_divergence(zs: torch.Tensor, zt: torch.Tensor, ys: torch.Tensor, yt: torch.Tensor,
                   sigma_z: float, sigma_y: float) -> torch.Tensor:
    """Eq. 13 (= Eq. 86, Eq. 88): z (M, d), (N, d) and y_hat (M, C), (N, C) -> scalar.

    K are Gram matrices of z at sigma_z, L of y_hat at sigma_y; K^st is (M, N)
    with (K^st)_ji = kappa(z^s_j, z^t_i), K^ts its transpose.
    """
    if (zs.ndim != 2 or zt.ndim != 2 or ys.ndim != 2 or yt.ndim != 2
            or len(zs) != len(ys) or len(zt) != len(yt) or not len(zs) or not len(zt)
            or zs.shape[1] != zt.shape[1] or ys.shape[1] != yt.shape[1]):
        raise ValueError("paired non-empty (samples, d) features and (samples, C) predictions "
                         "required on both sides")
    ks, kt = log_gaussian_gram(zs, zs, sigma_z), log_gaussian_gram(zt, zt, sigma_z)
    kst, kts = log_gaussian_gram(zs, zt, sigma_z), log_gaussian_gram(zt, zs, sigma_z)
    ls, lt = log_gaussian_gram(ys, ys, sigma_y), log_gaussian_gram(yt, yt, sigma_y)
    lst, lts = log_gaussian_gram(ys, yt, sigma_y), log_gaussian_gram(yt, ys, sigma_y)
    self_s = _log_ratio_sum(ks, ls, ks, ks)        # denominator (sum_i K^s_ji)^2
    self_t = _log_ratio_sum(kt, lt, kt, kt)
    cross_st = _log_ratio_sum(kst, lst, ks, kst)   # (sum_i K^s_ji)(sum_i K^st_ji)
    cross_ts = _log_ratio_sum(kts, lts, kts, kt)   # (sum_i K^ts_ji)(sum_i K^t_ji)
    return self_s + self_t - cross_st - cross_ts


@torch.no_grad()
def median_bandwidth(points: torch.Tensor, multiplier: float) -> float:
    """Plan Sec. 7.7: multiplier x median of the positive pairwise Euclidean distances.

    `points` (n, d) are the already-normalised kernel inputs. No positive
    distance (n < 2 or a collapsed set): the paper's sigma = 1 (Sec. 4.2)
    stands in for the undefined median.
    """
    multiplier = _check_width(multiplier)
    if points.ndim != 2 or not torch.isfinite(points).all():
        raise ValueError("a finite (samples, d) matrix is required for the median")
    distance = torch.pdist(points.to(torch.float64))
    positive = distance[distance > 0]
    median = float(positive.median()) if positive.numel() else PAPER_SIGMA
    return multiplier * median
