import numpy as np
from skimage import io, transform
from scipy.stats import ttest_1samp, norm
import matplotlib.pyplot as plt
import math


def compute_percent_loss_with_se(flat_b, flat_a, assume_correlation=True, alpha=0.05):
    """
    Compute percent loss and 100*(1 - alpha)% confidence interval using error propagation.
    
    Parameters:
        flat_b (np.ndarray): Flattened baseline intensity array
        flat_a (np.ndarray): Flattened aged intensity array
        assume_correlation (bool): Whether to include covariance term in error propagation
        alpha (float): Significance level (default 0.05 for 95% CI)
    
    Returns:
        percent_loss, ci_low, ci_high
    """
    mu_b = np.mean(flat_b)
    mu_a = np.mean(flat_a)
    n_b = len(flat_b)
    n_a = len(flat_a)
    sigma_b = np.std(flat_b, ddof=1)
    sigma_a = np.std(flat_a, ddof=1)

    if mu_b == 0:
        return 0.0, 0.0, 0.0 

    percent_loss = 100 * (mu_b - mu_a) / mu_b

    var_mu_b = (sigma_b ** 2) / n_b
    var_mu_a = (sigma_a ** 2) / n_a

    if assume_correlation:
        corr = np.corrcoef(flat_b, flat_a)[0, 1]
        cov_mu = corr * (sigma_b / np.sqrt(n_b)) * (sigma_a / np.sqrt(n_a))
    else:
        cov_mu = 0

    # covar
    term1 = (mu_a / mu_b) ** 2 * var_mu_b
    term2 = var_mu_a
    term3 = -2 * (mu_a / mu_b) * cov_mu
    var_percent_loss = (100 / mu_b) ** 2 * (term1 + term2 + term3)
    se = np.sqrt(var_percent_loss)

    # z-score for two-tailed CI
    
    z = norm.ppf(1 - alpha / 2)
    ci_low = percent_loss - z * se
    ci_high = percent_loss + z * se

    return percent_loss, ci_low, ci_high

def bootstrap_percent_loss(flat_b, flat_a, n_boot=1000, alpha=0.05, seed=None):
    rng = np.random.default_rng(seed)
    percent_losses = []

    for _ in range(n_boot):
        sample_b = rng.choice(flat_b, size=len(flat_b), replace=True)
        sample_a = rng.choice(flat_a, size=len(flat_a), replace=True)

        mu_b = np.mean(sample_b)
        mu_a = np.mean(sample_a)

        if mu_b != 0:
            loss = 100 * (mu_b - mu_a) / mu_b
            percent_losses.append(loss)

    percent_losses = np.array(percent_losses)
    mean_loss = np.mean(percent_losses)
    ci_low = np.percentile(percent_losses, 100 * alpha / 2)
    ci_high = np.percentile(percent_losses, 100 * (1 - alpha / 2))

    return mean_loss, ci_low, ci_high

def plot_histogram_grid_with_loss(
    baseline_images, aged_images, labels, bin_edges,
    bins=500, cols=4, figsize=(20, 12), save_path=None
):
    """
    Plot histograms of baseline vs aged images with vertical lines at the MEAN
    and annotate percent loss with analytic confidence intervals.
    """
    n_samples = len(labels)
    rows = math.ceil(n_samples / cols)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, (img_b, img_a, label) in enumerate(zip(baseline_images, aged_images, labels)):
        ax = axes[i]
        flat_b = img_b.flatten()
        flat_a = img_a.flatten()

        hist_b, _ = np.histogram(flat_b, bins=bin_edges, density=False)
        hist_a, _ = np.histogram(flat_a, bins=bin_edges, density=False)

        percent_loss, ci_low, ci_high = compute_percent_loss_with_se(flat_b, flat_a)

        mean_b = np.mean(flat_b)
        mean_a = np.mean(flat_a)

        ax.plot(bin_centers, hist_b, label='Baseline', alpha=0.7)
        ax.plot(bin_centers, hist_a, label='Aged', alpha=0.7)
        ax.axvline(mean_b, color='blue', linestyle='--', alpha=0.6, label='Mean Baseline' if i == 0 else None)
        ax.axvline(mean_a, color='orange', linestyle='--', alpha=0.6, label='Mean Aged' if i == 0 else None)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Intensity", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

        ax.text(
            0.05, 0.82,
            f"{percent_loss:.1f}%\n[{ci_low:.1f}, {ci_high:.1f}]",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6)
        )

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', fontsize=10)
    fig.suptitle("Histograms with Mean and Percent Loss +- CI (Error Propagation)", fontsize=14)
    plt.subplots_adjust(right=0.87)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
