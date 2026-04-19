import os

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# PARAMETERS
# ----------------------------
N_PLAYERS = 300
OUTPUT_DIR = "r2manualplots"
IMPROVEMENT_TOL = 1e-9
MASS_TOL = 1e-12
BISECTION_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# FUNCTIONS
# ----------------------------
def research(x):
    return 200_000 * np.log(1 + x) / np.log(101)


def scale(y):
    return 7 * y / 100.0


def speed_multiplier_from_counts(my_z, opp_counts):
    higher = np.sum(opp_counts[my_z + 1 :])
    equal = opp_counts[my_z]
    n_players = np.sum(opp_counts) + 1

    if n_players <= 1:
        return 0.9

    avg_rank = (2 * higher + equal + 2) / 2
    return 0.9 - 0.8 * (avg_rank - 1) / (n_players - 1)


def speed_multiplier(my_z, opp_zs):
    counts = np.bincount(np.asarray(opp_zs, dtype=int), minlength=101)
    return speed_multiplier_from_counts(my_z, counts)


def tie_bin_summary(z_pop):
    counts = np.bincount(np.asarray(z_pop, dtype=int), minlength=101)
    n_players = len(z_pop)

    if n_players <= 1:
        avg_ranks = np.ones(101)
        multipliers = np.full(101, 0.9)
        return avg_ranks, multipliers, counts

    avg_ranks = np.full(101, np.nan)
    multipliers = np.full(101, np.nan)
    higher_counts = np.cumsum(counts[::-1])[::-1] - counts
    occupied_bins = np.flatnonzero(counts)

    for z in occupied_bins:
        avg_rank = (2 * higher_counts[z] + counts[z] + 1) / 2
        avg_ranks[z] = avg_rank
        multipliers[z] = 0.9 - 0.8 * (avg_rank - 1) / (n_players - 1)

    return avg_ranks, multipliers, counts


def pnl(x, y, z, sm):
    gross = research(x) * scale(y) * sm
    cost = 50_000 * (x + y + z) / 100.0
    return gross - cost


def solve_x_from_z(z):
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


def precompute_z_economics():
    gross_by_z = np.zeros(101, dtype=float)
    cost_by_z = np.zeros(101, dtype=float)

    for z in range(101):
        x, y = solve_x_from_z(z)
        gross_by_z[z] = research(x) * scale(y)
        cost_by_z[z] = 50_000 * (x + y + z) / 100.0

    return gross_by_z, cost_by_z


def expected_payoff_by_z(z_probs, gross_by_z, cost_by_z):
    payoffs = np.zeros(101, dtype=float)
    tail_mass_above = 0.0

    for z in range(100, -1, -1):
        payoffs[z] = gross_by_z[z] * (0.9 - 0.8 * (tail_mass_above + 0.5 * z_probs[z])) - cost_by_z[z]
        tail_mass_above += z_probs[z]

    return payoffs


def build_distribution_for_value(value, gross_by_z, cost_by_z):
    z_probs = np.zeros(101, dtype=float)
    tail_mass_above = 0.0
    top_rank_payoff = 0.9 * gross_by_z - cost_by_z

    # Descending water-fill:
    # if an empty z-bin would beat the common value level, it must absorb enough
    # mass to push its own payoff back down to that same level.
    for z in range(100, -1, -1):
        empty_bin_payoff = top_rank_payoff[z] - 0.8 * gross_by_z[z] * tail_mass_above
        if empty_bin_payoff <= value + IMPROVEMENT_TOL:
            continue

        z_mass = (empty_bin_payoff - value) / (0.4 * gross_by_z[z])
        z_mass = max(0.0, z_mass)

        z_probs[z] = z_mass
        tail_mass_above += z_mass

    return z_probs, tail_mass_above


def solve_rank_equilibrium(gross_by_z, cost_by_z):
    low = float(np.min(0.1 * gross_by_z - cost_by_z)) - 1.0
    high = float(np.max(0.9 * gross_by_z - cost_by_z)) + 1.0
    _, low_mass = build_distribution_for_value(low, gross_by_z, cost_by_z)
    _, high_mass = build_distribution_for_value(high, gross_by_z, cost_by_z)

    if low_mass <= 1.0 or high_mass >= 1.0:
        raise RuntimeError("Failed to bracket the rank-equilibrium value.")

    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        _, total_mass = build_distribution_for_value(mid, gross_by_z, cost_by_z)

        if total_mass > 1.0:
            low = mid
        else:
            high = mid

    value = (low + high) / 2
    z_probs, total_mass = build_distribution_for_value(value, gross_by_z, cost_by_z)

    if total_mass <= MASS_TOL:
        raise RuntimeError("Failed to build a non-empty equilibrium distribution.")

    z_probs /= z_probs.sum()
    return value, z_probs


def discretize_distribution(z_probs, n_players):
    raw_counts = z_probs * n_players
    counts = np.floor(raw_counts).astype(int)
    remainder = n_players - counts.sum()

    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw_counts - counts)
        counts[order[: -remainder]] -= 1

    return counts


def payoff_from_counts(z, opp_counts, gross_by_z, cost_by_z):
    sm = speed_multiplier_from_counts(z, opp_counts)
    return gross_by_z[z] * sm - cost_by_z[z]


def discrete_regret_summary(counts, gross_by_z, cost_by_z):
    max_regret = 0.0
    avg_regret = 0.0
    worst_from = None
    worst_to = None

    for z in np.flatnonzero(counts):
        opp_counts = counts.copy()
        opp_counts[z] -= 1

        current = payoff_from_counts(z, opp_counts, gross_by_z, cost_by_z)
        candidate_payoffs = np.array(
            [payoff_from_counts(candidate_z, opp_counts, gross_by_z, cost_by_z) for candidate_z in range(101)]
        )
        best_to = int(np.argmax(candidate_payoffs))
        regret = float(candidate_payoffs[best_to] - current)

        avg_regret += counts[z] * regret
        if regret > max_regret:
            max_regret = regret
            worst_from = int(z)
            worst_to = best_to

    return max_regret, avg_regret / counts.sum(), worst_from, worst_to


# ----------------------------
# PLOTTING
# ----------------------------
def plot_distribution(z_pop, filename_stem, title):
    plt.figure()
    plt.hist(z_pop, bins=np.arange(-0.5, 101.5, 5))
    plt.xlabel("z (Speed)")
    plt.ylabel("Players")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_stem}.png"))
    plt.close()


def plot_probability_curve(z_probs):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(101), z_probs, width=0.9)
    plt.xlabel("z (Speed)")
    plt.ylabel("Probability Mass")
    plt.title("Rank Equilibrium Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rank_equilibrium_distribution.png"))
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
np.random.seed(0)

gross_by_z, cost_by_z = precompute_z_economics()
equilibrium_value, z_probs = solve_rank_equilibrium(gross_by_z, cost_by_z)
expected_payoffs = expected_payoff_by_z(z_probs, gross_by_z, cost_by_z)

count_profile = discretize_distribution(z_probs, N_PLAYERS)
z_pop = np.repeat(np.arange(101), count_profile)
np.random.shuffle(z_pop)

plot_distribution(z_pop, "equilibrium_sample", "Rounded Rank Equilibrium Sample")
plot_probability_curve(z_probs)

avg_ranks, multipliers, rounded_counts = tie_bin_summary(z_pop)
support = np.flatnonzero(z_probs > 1e-10)
top_bins = support[np.argsort(z_probs[support])[::-1][:12]]
top_bin_text = ", ".join(
    (
        f"z={z}:p={z_probs[z]:.4f},count={rounded_counts[z]},"
        f"rank={avg_ranks[z]:.1f},k={multipliers[z]:.3f},u={expected_payoffs[z]:.2f}"
    )
    for z in top_bins
)

max_regret, avg_regret, worst_from, worst_to = discrete_regret_summary(count_profile, gross_by_z, cost_by_z)
mean_z = float(np.dot(np.arange(101), z_probs))
std_z = float(np.sqrt(np.dot((np.arange(101) - mean_z) ** 2, z_probs)))

print(
    f"Rank equilibrium solved | value={equilibrium_value:.2f} | "
    f"mean={mean_z:.2f} std={std_z:.2f} | "
    f"support={len(support)} bins",
    flush=True,
)
print(f"Top bins: {top_bin_text}", flush=True)
print(
    f"Rounded {N_PLAYERS}-player profile regret | max={max_regret:.2f} avg={avg_regret:.2f} | "
    f"worst move: z={worst_from} -> z={worst_to}",
    flush=True,
)
print(f"\nPlots saved in '{OUTPUT_DIR}'", flush=True)
