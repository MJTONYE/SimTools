import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------
SEED = 42

params = {
    "sA": 0.8,
    "sJ": 0.4,
    "f": 2.0,
    "sex_ratio": 0.5,
    "T": 10,
    "threshold": 250
}

# Unknown juveniles at t=0 -> conservative assumption:
J_TOTAL = 0


# -----------------------------
# HELPERS
# -----------------------------
def split_juveniles(J_total: int):
    """Split juveniles into (J0,J1,J2) with sum = J_total."""
    base = J_total // 3
    r = J_total % 3
    j0 = base + (1 if r > 0 else 0)
    j1 = base + (1 if r > 1 else 0)
    j2 = base
    return (j0, j1, j2)


# -----------------------------
# SIMULATION (one path)
# -----------------------------
def simulate_path_state(A0: int, J0: tuple[int, int, int], params: dict, rng: np.random.Generator):
    """
    Simulate one population path for T years.
    Returns final adults, final juvenile age-classes, and final total N_T.
    """
    A = int(A0)
    j0, j1, j2 = map(int, J0)

    sA = params["sA"]
    sJ = params["sJ"]
    f = params["f"]
    sex_ratio = params["sex_ratio"]
    T = params["T"]

    for _ in range(T):
        SA  = rng.binomial(A,  sA)
        SJ0 = rng.binomial(j0, sJ)
        SJ1 = rng.binomial(j1, sJ)
        SJ2 = rng.binomial(j2, sJ)

        B = rng.poisson(f * (sex_ratio * A))  # births

        A  = SA + SJ2
        j2 = SJ1
        j1 = SJ0
        j0 = B

    N_T = A + j0 + j1 + j2
    return {"A_T": A, "J0_T": j0, "J1_T": j1, "J2_T": j2, "N_T": N_T}


def simulate_path(A0: int, J0: tuple[int, int, int], params: dict, rng: np.random.Generator) -> int:
    """Same simulation, but returns only N_T (used for Monte-Carlo)."""
    return simulate_path_state(A0, J0, params, rng)["N_T"]


# -----------------------------
# MONTE-CARLO: estimate p_hat(A0)
# -----------------------------
def estimate_p_hat(A0: int, J_total: int, params: dict, R: int = 2000, seed: int = SEED):
    """
    Estimate p(A0) = P(N_T >= threshold).
    Juveniles at t=0 fixed to J_total. N0 is not fixed: N0_model = A0 + J_total.
    """
    rng = np.random.default_rng(seed)
    threshold = params["threshold"]
    J0 = split_juveniles(J_total)

    successes = 0
    for _ in range(R):
        N_T = simulate_path(A0, J0, params, rng)
        successes += (N_T >= threshold)

    p_hat = successes / R
    se = np.sqrt(p_hat * (1 - p_hat) / R)
    ci95 = (float(max(0.0, p_hat - 1.96 * se)), float(min(1.0, p_hat + 1.96 * se)))

    return {
        "A0": A0,
        "J_total": J_total,
        "J0_init": J0,
        "N0_model": A0 + J_total,
        "R": R,
        "p_hat": float(p_hat),
        "ci95": ci95
    }


# -----------------------------
# SEARCH: minimal A0 for target probability
# -----------------------------
def find_min_A0(params: dict, J_total: int, target: float = 0.95, R: int = 2000, seed: int = SEED,
               start_A: int = 50, step: int = 25, maxA: int = 5000, use_ci_lower: bool = True):
    """
    Find smallest A0 such that:
      - strict: lower 95% CI >= target  (use_ci_lower=True)
      - or:     p_hat >= target         (use_ci_lower=False)
    Juveniles at t=0 fixed to J_total. N0 is not fixed.
    """
    def passes(res):
        return (res["ci95"][0] >= target) if use_ci_lower else (res["p_hat"] >= target)

    # 1) Find an upper bound that passes
    A_high = start_A
    res_high = estimate_p_hat(A_high, J_total, params, R=R, seed=seed)
    while not passes(res_high):
        A_high += step
        if A_high > maxA:
            raise RuntimeError("No solution up to maxA. Increase maxA or change parameters/target.")
        res_high = estimate_p_hat(A_high, J_total, params, R=R, seed=seed)

    # 2) Find a lower bound (ideally failing)
    A_low = max(1, A_high - step)
    res_low = estimate_p_hat(A_low, J_total, params, R=R, seed=seed)
    while passes(res_low) and A_low > 1:
        A_high, res_high = A_low, res_low
        A_low = max(1, A_low - step)
        res_low = estimate_p_hat(A_low, J_total, params, R=R, seed=seed)

    # 3) Binary search
    lo, hi = A_low, A_high
    best = res_high
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        res_mid = estimate_p_hat(mid, J_total, params, R=R, seed=seed)
        if passes(res_mid):
            best = res_mid
            hi = mid
        else:
            lo = mid

    best["target"] = target
    best["criterion"] = "ci95_lower" if use_ci_lower else "p_hat"
    return best


# -----------------------------
# RUN (Display the answers cleanly)
# -----------------------------
print("One example path (shows juveniles are created by births even if J_total=0):")
rng = np.random.default_rng(SEED)
print(simulate_path_state(A0=100, J0=split_juveniles(J_TOTAL), params=params, rng=rng))

print("\nBaseline A0=100 (probability to reach threshold=250 after 10 years):")
print(estimate_p_hat(A0=100, J_total=J_TOTAL, params=params, R=1000, seed=SEED))

print("\nMinimal A0 for target probability (J_total=0):")
best = find_min_A0(params=params, J_total=J_TOTAL, target=0.95, R=1000, seed=SEED,
                   start_A=50, step=25, maxA=500, use_ci_lower=True)
print(best)




