"""
Estimate the population distribution of z using behavioral game theory.

Recommended default:
  - Quantal Cognitive Hierarchy (QCH) / Truncated QRE style estimate
    inspired by Camerer, Ho, Chong (2004) and Rogers, Palfrey, Camerer (2009).
  - Optional heterogeneous QRE comparison for a more equilibrium-like
    late-stage field.

Primary references:
  - McKelvey, Palfrey (1995), "Quantal Response Equilibria for Normal Form Games"
  - Camerer, Ho, Chong (2004), "A Cognitive Hierarchy Model of Games"
  - Rogers, Palfrey, Camerer (2009), "Heterogeneous quantal response equilibrium
    and cognitive hierarchies"

Important modeling choice:
  The papers usually use a simple level-0 rule such as uniform randomization.
  For this z-game, that is too crude because actions have an obvious cost/rank
  tradeoff. So the default level-0 prior below is a mixture of:
    1. uniform "some people are noisy / unsophisticated"
    2. a cost-aware soft rule "some people optimize the private tradeoff but
       do not model the field deeply"
  That mixture is an inference for this domain, not a direct empirical estimate
  from the papers.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "r2manualplots"
N_PLAYERS = 300


def research(x: float) -> float:
    return 200_000 * np.log(1 + x) / np.log(101)


def scale(y: float) -> float:
    return 7 * y / 100.0


def solve_x_from_z(z: int) -> tuple[float, float]:
    lo, hi = 0.0, 100.0
    for _ in range(30):
        mid = (lo + hi) / 2
        y = (mid + 1) * np.log(mid + 1)
        if mid + y + z > 100:
            hi = mid
        else:
            lo = mid
    x = (lo + hi) / 2
    y = (x + 1) * np.log(x + 1)
    return x, y


def precompute_z_economics() -> tuple[np.ndarray, np.ndarray]:
    gross_by_z = np.zeros(101, dtype=float)
    cost_by_z = np.zeros(101, dtype=float)

    for z in range(101):
        x, y = solve_x_from_z(z)
        gross_by_z[z] = research(x) * scale(y)
        cost_by_z[z] = 50_000 * (x + y + z) / 100.0

    return gross_by_z, cost_by_z


def utility_vector(field_dist: np.ndarray, gross_by_z: np.ndarray, cost_by_z: np.ndarray) -> np.ndarray:
    utilities = np.zeros(101, dtype=float)
    tail_mass_above = 0.0

    for z in range(100, -1, -1):
        own_mass = field_dist[z]
        speed_multiplier = 0.9 - 0.8 * (tail_mass_above + 0.5 * own_mass)
        utilities[z] = gross_by_z[z] * speed_multiplier - cost_by_z[z]
        tail_mass_above += own_mass

    return utilities


def stable_softmax(values: np.ndarray, precision: float) -> np.ndarray:
    shifted = precision * (values - np.max(values))
    weights = np.exp(np.clip(shifted, -700, 700))
    return weights / weights.sum()


def normalize(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def poisson_level_weights(mean_level: float, max_level: int) -> np.ndarray:
    levels = np.arange(max_level + 1)
    log_weights = -mean_level + levels * np.log(mean_level) - np.array([math.lgamma(k + 1) for k in levels])
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


def default_level0_distribution(
    gross_by_z: np.ndarray,
    cost_by_z: np.ndarray,
    uniform_weight: float = 0.35,
    cost_aware_multiplier: float = 0.55,
    cost_aware_precision: float = 1 / 120_000,
) -> np.ndarray:
    uniform = np.full(101, 1.0 / 101.0)
    cost_aware_values = cost_aware_multiplier * gross_by_z - cost_by_z
    cost_aware = stable_softmax(cost_aware_values, cost_aware_precision)
    mixed = uniform_weight * uniform + (1.0 - uniform_weight) * cost_aware
    return mixed / mixed.sum()


def quantal_cognitive_hierarchy_distribution(
    gross_by_z: np.ndarray,
    cost_by_z: np.ndarray,
    mean_level: float = 3.25,
    max_level: int = 8,
    level0_uniform_weight: float = 0.35,
    level0_multiplier: float = 0.55,
    level0_precision: float = 1 / 120_000,
    precision_step: float = 1 / 18_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    QCH / TQRE-style estimator.

    Level-0 is non-strategic. Higher levels choose quantal best responses to the
    truncated distribution of lower levels.
    """

    level_weights = poisson_level_weights(mean_level, max_level)
    level_strategies = [
        default_level0_distribution(
            gross_by_z,
            cost_by_z,
            uniform_weight=level0_uniform_weight,
            cost_aware_multiplier=level0_multiplier,
            cost_aware_precision=level0_precision,
        )
    ]

    for level in range(1, max_level + 1):
        lower_mass = level_weights[:level].sum()
        lower_mix = sum(level_weights[j] * level_strategies[j] for j in range(level)) / lower_mass
        utilities = utility_vector(lower_mix, gross_by_z, cost_by_z)
        precision = level * precision_step
        level_strategies.append(stable_softmax(utilities, precision))

    field_dist = sum(level_weights[level] * level_strategies[level] for level in range(max_level + 1))
    return field_dist / field_dist.sum(), level_weights, np.asarray(level_strategies)


def heterogeneous_qre_distribution(
    gross_by_z: np.ndarray,
    cost_by_z: np.ndarray,
    lambdas: np.ndarray | None = None,
    lambda_weights: np.ndarray | None = None,
    damping: float = 0.35,
    tol: float = 1e-12,
    max_iters: int = 2_000,
) -> np.ndarray:
    """
    Heterogeneous-QRE fixed point.

    This is a more equilibrium-like benchmark than QCH: players differ in payoff
    responsiveness, but they respond to the actual aggregate distribution.
    """

    if lambdas is None:
        lambdas = np.array([1 / 80_000, 1 / 40_000, 1 / 25_000, 1 / 15_000], dtype=float)
    if lambda_weights is None:
        lambda_weights = np.array([0.15, 0.35, 0.30, 0.20], dtype=float)

    lambda_weights = lambda_weights / lambda_weights.sum()
    field_dist = np.full(101, 1.0 / 101.0)

    for _ in range(max_iters):
        utilities = utility_vector(field_dist, gross_by_z, cost_by_z)
        next_dist = np.zeros(101, dtype=float)
        for lam, weight in zip(lambdas, lambda_weights):
            next_dist += weight * stable_softmax(utilities, lam)

        updated = (1.0 - damping) * field_dist + damping * next_dist
        if np.max(np.abs(updated - field_dist)) < tol:
            field_dist = updated
            break
        field_dist = updated

    return field_dist / field_dist.sum()


def best_response(field_dist: np.ndarray, gross_by_z: np.ndarray, cost_by_z: np.ndarray) -> tuple[int, np.ndarray]:
    utilities = utility_vector(field_dist, gross_by_z, cost_by_z)
    return int(np.argmax(utilities)), utilities


def discretize_distribution(dist: np.ndarray, n_players: int) -> np.ndarray:
    raw = dist * n_players
    counts = np.floor(raw).astype(int)
    remainder = n_players - counts.sum()

    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        counts[order[: -remainder]] -= 1

    return counts


def kl_divergence(target: np.ndarray, approx: np.ndarray) -> float:
    safe_approx = np.maximum(approx, 1e-15)
    mask = target > 0
    return float(np.sum(target[mask] * np.log(target[mask] / safe_approx[mask])))


def truncated_normal_distribution(mu: float, sigma: float) -> np.ndarray:
    z_midpoints = np.arange(101, dtype=float) + 0.5
    weights = np.exp(-0.5 * ((z_midpoints - mu) / sigma) ** 2)
    return normalize(weights)


def beta_like_distribution(alpha: float, beta: float) -> np.ndarray:
    x = (np.arange(101, dtype=float) + 0.5) / 101.0
    log_beta_fn = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x) - log_beta_fn
    return normalize(np.exp(log_pdf))


def fit_best_unimodal_curve(target: np.ndarray) -> tuple[str, np.ndarray, dict[str, float], float]:
    target_mean = float(np.dot(np.arange(101), target))
    target_std = float(np.sqrt(np.dot((np.arange(101) - target_mean) ** 2, target)))

    best_normal_score = float("inf")
    best_normal_dist = None
    best_normal_params = None

    mu_grid = np.linspace(max(0.0, target_mean - 15.0), min(100.0, target_mean + 15.0), 241)
    sigma_grid = np.linspace(max(3.0, target_std * 0.5), min(25.0, target_std * 1.8), 161)

    for mu in mu_grid:
        for sigma in sigma_grid:
            candidate = truncated_normal_distribution(float(mu), float(sigma))
            score = kl_divergence(target, candidate)
            if score < best_normal_score:
                best_normal_score = score
                best_normal_dist = candidate
                best_normal_params = {"mu": float(mu), "sigma": float(sigma)}

    best_beta_score = float("inf")
    best_beta_dist = None
    best_beta_params = None

    alpha_grid = np.linspace(1.2, 12.0, 217)
    beta_grid = np.linspace(1.2, 12.0, 217)

    for alpha in alpha_grid:
        for beta in beta_grid:
            candidate = beta_like_distribution(float(alpha), float(beta))
            score = kl_divergence(target, candidate)
            if score < best_beta_score:
                best_beta_score = score
                best_beta_dist = candidate
                best_beta_params = {"alpha": float(alpha), "beta": float(beta)}

    if best_normal_score <= best_beta_score:
        return "truncated_normal", best_normal_dist, best_normal_params, best_normal_score

    return "beta_like", best_beta_dist, best_beta_params, best_beta_score


def describe_distribution(name: str, dist: np.ndarray, gross_by_z: np.ndarray, cost_by_z: np.ndarray) -> None:
    best_z, utilities = best_response(dist, gross_by_z, cost_by_z)
    mean = float(np.dot(np.arange(101), dist))
    std = float(np.sqrt(np.dot((np.arange(101) - mean) ** 2, dist)))
    top = np.argsort(dist)[::-1][:12]
    top_text = ", ".join(f"z={z}:p={dist[z]:.4f},u={utilities[z]:.0f}" for z in top)
    mid_mass = float(dist[35:61].sum())
    high_mass = float(dist[70:86].sum())
    low_mass = float(dist[:26].sum())

    print(
        f"{name} | mean={mean:.2f} std={std:.2f} | best response z={best_z} | "
        f"mass z<=25: {low_mass:.3f} | mass 35-60: {mid_mass:.3f} | mass 70-85: {high_mass:.3f}",
        flush=True,
    )
    print(f"  Top bins: {top_text}", flush=True)


def plot_distributions(qch_dist: np.ndarray, hqre_dist: np.ndarray) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(11, 5))
    z = np.arange(101)
    plt.plot(z, qch_dist, label="QCH / TQRE-style estimate", linewidth=2.3)
    plt.plot(z, hqre_dist, label="HQRE comparison", linewidth=2.0)
    plt.xlabel("z")
    plt.ylabel("Probability")
    plt.title("Estimated z Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "z_distribution_estimates.png"))
    plt.close()


def plot_distribution_with_fit(target: np.ndarray, fitted: np.ndarray, family_name: str) -> None:
    plt.figure(figsize=(11, 5))
    z = np.arange(101)
    plt.plot(z, target, label="Behavioral target (QCH)", linewidth=2.3)
    plt.plot(z, fitted, label=f"Best unimodal fit ({family_name})", linewidth=2.0)
    plt.xlabel("z")
    plt.ylabel("Probability")
    plt.title("Smooth Unimodal Approximation of the z Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "z_distribution_unimodal_fit.png"))
    plt.close()


def plot_rounded_sample(name: str, dist: np.ndarray, filename: str) -> None:
    counts = discretize_distribution(dist, N_PLAYERS)
    sample = np.repeat(np.arange(101), counts)

    plt.figure(figsize=(10, 4))
    plt.hist(sample, bins=np.arange(-0.5, 101.5, 3))
    plt.xlabel("z")
    plt.ylabel("Players")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def estimate_and_print_best_z() -> dict[str, object]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gross_by_z, cost_by_z = precompute_z_economics()

    qch_dist, level_weights, level_strategies = quantal_cognitive_hierarchy_distribution(
        gross_by_z,
        cost_by_z,
        mean_level=3.25,
        max_level=8,
        level0_uniform_weight=0.35,
        level0_multiplier=0.55,
        level0_precision=1 / 120_000,
        precision_step=1 / 18_000,
    )
    hqre_dist = heterogeneous_qre_distribution(gross_by_z, cost_by_z)

    print("Behavioral z-estimates for a smart but not perfectly rational field", flush=True)
    print(
        "QCH settings: mean level=3.25, max level=8, "
        "level-0 = 35% uniform + 65% cost-aware prior",
        flush=True,
    )
    print(
        "HQRE settings: lambda grid = [1/80000, 1/40000, 1/25000, 1/15000], "
        "weights = [0.15, 0.35, 0.30, 0.20]",
        flush=True,
    )
    print(
        f"QCH level weights: {', '.join(f'k={k}:{w:.3f}' for k, w in enumerate(level_weights))}",
        flush=True,
    )
    print("", flush=True)

    describe_distribution("QCH estimate", qch_dist, gross_by_z, cost_by_z)
    describe_distribution("HQRE comparison", hqre_dist, gross_by_z, cost_by_z)
    fitted_family, fitted_dist, fitted_params, fitted_score = fit_best_unimodal_curve(qch_dist)
    describe_distribution("Best unimodal fit", fitted_dist, gross_by_z, cost_by_z)

    qch_best, _ = best_response(qch_dist, gross_by_z, cost_by_z)
    hqre_best, _ = best_response(hqre_dist, gross_by_z, cost_by_z)
    fitted_best, _ = best_response(fitted_dist, gross_by_z, cost_by_z)
    blended = 0.7 * qch_dist + 0.3 * hqre_dist
    blended_best, _ = best_response(blended, gross_by_z, cost_by_z)

    print("", flush=True)
    if fitted_family == "truncated_normal":
        fitted_label = f"truncated normal(mu={fitted_params['mu']:.2f}, sigma={fitted_params['sigma']:.2f})"
    else:
        fitted_label = f"beta-like(alpha={fitted_params['alpha']:.2f}, beta={fitted_params['beta']:.2f})"
    print(
        f"Best smooth unimodal curve | family={fitted_label} | KL to QCH={fitted_score:.4f} | "
        f"best response z={fitted_best}",
        flush=True,
    )
    print(
        f"Practical recommendation | QCH best z={qch_best} | HQRE best z={hqre_best} | "
        f"70/30 blend best z={blended_best} | unimodal-fit best z={fitted_best}",
        flush=True,
    )
    print(
        "If you want one clean bell-shaped assumption, use the unimodal fit. "
        "If you want the more theory-faithful estimate, use QCH.",
        flush=True,
    )

    plot_distributions(qch_dist, hqre_dist)
    plot_distribution_with_fit(qch_dist, fitted_dist, fitted_family)
    plot_rounded_sample("Rounded QCH Sample", qch_dist, "qch_distribution_sample.png")
    plot_rounded_sample("Rounded HQRE Sample", hqre_dist, "hqre_distribution_sample.png")

    return {
        "qch_best_z": qch_best,
        "hqre_best_z": hqre_best,
        "blend_best_z": blended_best,
        "unimodal_fit_best_z": fitted_best,
        "fitted_family": fitted_family,
        "fitted_params": fitted_params,
        "fitted_score": fitted_score,
        "qch_dist": qch_dist,
        "hqre_dist": hqre_dist,
        "fitted_dist": fitted_dist,
    }


def main() -> None:
    estimate_and_print_best_z()


if __name__ == "__main__":
    main()
