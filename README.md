# Zero-Shot Adversarial Patch Detection via Diffusion Priors

## 📌 Overview
This repository implements a **Bayesian optimization framework** for detecting and removing adversarial patches from images without prior training on specific attacks. Unlike traditional defense mechanisms that require adversarial training or heuristic edge detectors, this method utilizes a pre-trained, frozen **Diffusion Probabilistic Model (DDPM)** as a prior for natural image statistics.

We formulate the defense as a **Blind Inpainting** task solved via **Score Distillation Sampling (SDS)**. The system simultaneously optimizes:
1.  A **Clean Image Estimate ($\mathbf{x}$)**.
2.  A **Binary Patch Mask ($\mathbf{m}$)**.

By leveraging the gradients of a diffusion model, we can "scrub" adversarial patterns from an image while preserving the semantic integrity of the background.

---

## 📐 The Mathematical Model

We model the observed adversarial image $\mathbf{y}$ as a composition of the clean background and the adversarial patch:

$$ \mathbf{y} = (\mathbf{1} - \mathbf{m}) \odot \mathbf{x} + \mathbf{m} \odot \boldsymbol{\delta} $$

Where:
*   $\mathbf{x}$: The latent clean image.
*   $\mathbf{m}$: The binary mask ($0=$ background, $1=$ patch).
*   $\boldsymbol{\delta}$: The unknown adversarial patch.

Since $\boldsymbol{\delta}$ is unknown and arbitrary, we minimize the negative log-posterior of the clean image and the mask:

$$ \mathbf{x}^*, \mathbf{m}^* = \arg\min_{\mathbf{x}, \mathbf{m}} \left[ \mathcal{L}_{\text{fidelity}} + \lambda_{\text{sds}}\mathcal{L}_{\text{prior}} + \mathcal{L}_{\text{reg}} \right] $$

### 1. Fidelity Loss (Background Consistency)
The clean estimate $\mathbf{x}$ must match the observation $\mathbf{y}$ wherever the mask is **zero** (background).
$$ \mathcal{L}_{\text{fidelity}} = \| (\mathbf{1} - \mathbf{m}) \odot (\mathbf{y} - \mathbf{x}) \|_2^2 $$

### 2. The Diffusion Prior (SDS)
To ensure $\mathbf{x}$ lies on the manifold of natural images, we use **Score Distillation Sampling (SDS)**. We push the pixels of $\mathbf{x}$ against the gradients of a frozen DDPM U-Net.
$$ \nabla_{\mathbf{x}}\mathcal{L}_{\text{prior}} \propto \left( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon} \right) $$
*Key Technique:* We sample $t \sim \mathcal{U}(20, 980)$ continuously, utilizing the full frequency range of the diffusion model to repair both high-frequency textures and global structure simultaneously.

### 3. Mask Regularization
To prevent trivial solutions (e.g., masking the entire image), we apply:
*   **Sparsity ($L_1$):** Penalizes the total area of the patch.
*   **Total Variation (TV):** Penalizes noise in the mask, encouraging contiguous shapes.

---

## 🚀 Novelty & Key Contributions

This approach bridges the gap between **Adversarial Defense** and **Score-Based Inverse Imaging**.

### 1. Simultaneous Detection and Restoration (Smart Inpainting)
Most diffusion defenses (e.g., *DiffPure*) are sequential: they noise and denoise the whole image, often destroying safe background details.
*   **Our Method:** Optimizes $\mathbf{x}$ and $\mathbf{m}$ jointly. The Fidelity Loss anchors the background, while the Diffusion Prior hallucinates content *only* where the patch exists.

### 2. Optimization via SDS for Restoration
While SDS is popularized by *DreamFusion* (Text-to-3D), applying it to **2D pixel-space restoration** is a novel application. This allows us to dynamically "scrub" the adversarial pattern rather than relying on discrete denoising steps.

### 3. Zero-Shot & Model Agnostic
This is a generative defense. It requires **no adversarial training** and works on patches never seen before (e.g., QR codes, noise, cartoon patches).

### 4. Dynamic Timestep Sampling
Standard defenses rely on a fixed timestep $t^*$ (a tradeoff between removing noise and keeping identity). By sampling $t$ continuously, our method amortizes the defense, fixing global structure and local texture in the same optimization loop without hyperparameter tuning.

---

## ⚙️ Algorithm Implementation

The core logic is implemented in the optimization loop (`optimize_patch.py`):

1.  **Initialization:**
    *   $\mathbf{x}$: Initialized as the input image $\mathbf{y}$ + slight noise.
    *   $\mathbf{w}$: Mask logits initialized to negative values (mask $\approx 0$).
2.  **Optimization (Adam):**
    *   The **Fidelity Loss** constrains $\mathbf{x}$ to match $\mathbf{y}$.
    *   The **SDS Gradient** pushes $\mathbf{x}$ to look "natural."
    *   The mask $\mathbf{w}$ grows strictly where $\mathbf{x}$ cannot match $\mathbf{y}$ while satisfying the naturalness prior.
3.  **Exponential Moving Average (EMA):**
    *   We maintain shadow variables `x_ema` and `w_ema`.
    *   **Why?** SDS optimization is inherently chaotic/stochastic. EMA stabilizes the trajectory, yielding smooth masks and high-quality restorations.
4.  **Thresholding:**
    *   The final binary mask is extracted using **Otsu Thresholding** on the EMA probability map.

---

## 🌍 Universal Robustness (Extension)

While designed for patches, this framework acts as a **Unified Bayesian Defense**.

*   **Patch Attacks:** The mask converges to local regions ($\mathbf{m} \approx 1$ locally).
*   **Global Attacks (PGD/L-inf):** The adversarial noise is treated as "out-of-distribution" texture. The SDS gradient cleans the pixels, while the mask acts as a dynamic uncertainty map.

**Future Architectures:**
The code supports swapping the DDPM backbone for **Stable Diffusion + LoRA**. This enables:
*   Defense at high resolutions ($512 \times 512$ or $1024 \times 1024$).
*   Prompt-guided purification (e.g., "A photo of a face").

---

## 💻 Installation & Usage

### Dependencies
```bash
pip install torch torchvision numpy matplotlib diffusers scikit-image pillow
```

### Running the Defense
Ensure you have an image path ready. The script handles image loading, patch corruption (for testing), and restoration.

```python
# In your Python script
from optimize_patch import load_image_from_path, add_noisy_block, optimize_patch_detection_ema

# 1. Load
clean_tensor = load_image_from_path("path/to/image.jpg")

# 2. Attack (Simulated)
corrupted_tensor, gt_mask = add_noisy_block(clean_tensor, size=(60, 60))

# 3. Defend
restored_img, detected_mask = optimize_patch_detection_ema(
    corrupted_tensor, 
    unet, 
    scheduler, 
    num_steps=1500
)
```

---

## 📜 Citation & Reference

If you use this method, please refer to the paper title:
> **"Simultaneous Adversarial Patch Detection and Restoration via Score Distillation Sampling"**
