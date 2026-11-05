"""Underwater image quality metrics, implemented from scratch (no skimage).

Full-reference : PSNR, SSIM
No-reference   : UCIQE (Yang & Sowmya, 2015), UIQM (Panetta et al., 2016)

Conventions: input images are BGR uint8 (H, W, 3).
"""

from __future__ import annotations

import cv2
import numpy as np

# --- UCIQE weights (Yang & Sowmya 2015, Table III) --------------------------
UCIQE_C1, UCIQE_C2, UCIQE_C3 = 0.4680, 0.2745, 0.2576
# --- UIQM weights (Panetta et al. 2016) ------------------------------------
UIQM_C1, UIQM_C2, UIQM_C3 = 0.0282, 0.2953, 3.5753


# ====================================================================== PSNR
def psnr(reference: np.ndarray, image: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (inf when identical)."""
    ref = reference.astype(np.float64)
    img = image.astype(np.float64)
    mse = np.mean((ref - img) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


# ====================================================================== SSIM
def _ssim_single(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM per channel: 11x11 Gaussian window, sigma=1.5, K1=0.01, K2=0.03."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).mean())


def ssim(reference: np.ndarray, image: np.ndarray) -> float:
    """Mean SSIM over channels."""
    return float(np.mean([
        _ssim_single(reference[..., c], image[..., c]) for c in range(3)
    ]))


# ===================================================================== UCIQE
def uciqe(image: np.ndarray) -> float:
    """Underwater Colour Image Quality Evaluation.

    UCIQE = c1*sigma_c + c2*con_l + c3*mu_s
      sigma_c : chroma std  sqrt(std(a*)^2 + std(b*)^2)   (CIELAB)
      con_l   : luminance contrast (std of L*)
      mu_s    : mean saturation (HSV S in [0, 1])
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_star, a_star, b_star = lab[..., 0], lab[..., 1], lab[..., 2]

    sigma_c = float(np.sqrt(np.var(a_star) + np.var(b_star)))
    con_l = float(np.std(l_star))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mu_s = float(hsv[..., 1].mean() / 255.0)

    return UCIQE_C1 * sigma_c + UCIQE_C2 * con_l + UCIQE_C3 * mu_s


# ====================================================================== UIQM
def _eme(gray: np.ndarray, block: int = 8, log_const: float = 20.0) -> float:
    """Enhancement Measure of Entropy: block-wise log of dynamic range.

    EME = (1/(k1*k2)) * sum_ij  log_const * log10((max+1)/(min+1))
    """
    gray = gray.astype(np.float64)
    h, w = gray.shape
    k1, k2 = max(1, h // block), max(1, w // block)
    gray = gray[: k1 * block, : k2 * block]

    total = 0.0
    for i in range(k1):
        for j in range(k2):
            patch = gray[i * block:(i + 1) * block, j * block:(j + 1) * block]
            pmax, pmin = float(patch.max()), float(patch.min())
            if pmax - pmin < 1e-6:
                continue
            total += log_const * np.log10((pmax + 1.0) / (pmin + 1.0))
    return total / (k1 * k2)


def _alpha_trimmed(values: np.ndarray, trim: float = 0.03) -> tuple[float, float]:
    """Two-sided alpha-trimmed mean and std (3% each side by default)."""
    v = np.sort(values.ravel().astype(np.float64))
    n = v.size
    cut = int(n * trim)
    v = v[cut: n - cut] if n - 2 * cut > 0 else v
    return float(v.mean()), float(v.std())


def uicm(image: np.ndarray) -> float:
    """UIQM colourfulness: opponent channels RG (red-green) and YB (yellow-blue)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b

    rg_mu, rg_std = _alpha_trimmed(rg)
    yb_mu, yb_std = _alpha_trimmed(yb)
    return float(np.sqrt(rg_mu**2 + rg_std**2) + np.sqrt(yb_mu**2 + yb_std**2))


def uism(image: np.ndarray) -> float:
    """UIQM sharpness: EME over the Sobel edge map of the HSV value channel."""
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float64)
    gx = cv2.Sobel(value, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(gx * gx + gy * gy)
    return _eme(edges, block=8)


def uiconm(image: np.ndarray) -> float:
    """UIQM contrast: EME over the log-luminance of the value channel."""
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float64)
    log_v = np.log1p(value)
    return _eme(log_v, block=8)


def uiqm(image: np.ndarray) -> float:
    """UIQM = c1*UICM + c2*UISM + c3*UIConM (Panetta et al., 2016)."""
    return UIQM_C1 * uicm(image) + UIQM_C2 * uism(image) + UIQM_C3 * uiconm(image)


# ================================================================= aggregate
def evaluate_enhancement(original: np.ndarray, enhanced: np.ndarray) -> dict[str, float]:
    """All four metrics for one (original, enhanced) pair.

    PSNR/SSIM compare enhanced against the original capture (restoration
    proximity); UCIQE/UIQM are no-reference quality scores of the enhanced
    image. ``uciqe_orig`` / ``uiqm_orig`` are included so relative gain can be
    reported (``delta_uiqm`` etc.).
    """
    return {
        "psnr": psnr(original, enhanced),
        "ssim": ssim(original, enhanced),
        "uciqe": uciqe(enhanced),
        "uiqm": uiqm(enhanced),
        "uciqe_orig": uciqe(original),
        "uiqm_orig": uiqm(original),
    }
