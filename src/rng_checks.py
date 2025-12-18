import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

seed = 42
rng = np.random.default_rng(seed)

def check_uniform(N=100_000, out_prefix="figures/uniform", rng=rng):
    u = rng.random(N) # U(0,1)
    mean_u = u.mean()
    var_u  = u.var(ddof=1)
    rel_err_mean = abs(mean_u - 0.5) / 0.5
    rel_err_var  = abs(var_u - (1/12)) / (1/12)
    
    print(f"[Uniform] N={N} mean={mean_u:.5f} var={var_u:.5f}")
    print(f" (rel.err {100*rel_err_mean:.2f}%)")
    print(f" (rel.err {100*rel_err_var:.2f}%)")
    
    # Histogramm + theoretische Dichte
    plt.figure()
    plt.hist(u, bins=50, density=True, edgecolor="black", alpha=0.7)
    plt.hlines(1.0, 0, 1, linestyles="--")  # Dichte U(0,1) = 1
    plt.title(f"Uniform(0,1) Histogramm, N={N}")
    plt.xlabel("u")
    plt.ylabel("Dichte")
    plt.savefig(f"{out_prefix}_hist.png", dpi=150)
    plt.close()
    
    # QQ-Plot gegen Uniform: sortierte u vs. theoretische Quantile
    p = (np.arange(1, N+1) - 0.5) / N
    theo_q = p  # PPF der Uniform(0,1) ist Identität
    u_sorted = np.sort(u)
    
    plt.figure()
    plt.plot(theo_q, u_sorted, ".", ms=2)
    plt.plot([0,1], [0,1], "k--")
    plt.title(f"QQ-Plot Uniform vs. Theorie, N={N}")
    plt.xlabel("theoretische Quantile")
    plt.ylabel("empirische Quantile")
    plt.savefig(f"{out_prefix}_qq.png", dpi=150) 
    plt.close()


def check_normal(N=100_000, out_prefix="figures/normal", rng=rng):
    z = rng.standard_normal(N)
    mean_z = z.mean()
    var_z  = z.var(ddof=1)
    print(f"[Normal] N={N} mean={mean_z:.5f}, var={var_z:.5f}")
    
    # Histogramm + Standardnormal-Dichte
    xs = np.linspace(-4, 4, 401)
    pdf = stats.norm.pdf(xs)
    
    plt.figure()
    plt.hist(z, bins=60, density=True, edgecolor="black", alpha=0.7)
    plt.plot(xs, pdf, "k--")
    plt.title(f"N(0,1) Histogramm, N={N}")
    plt.xlabel("z")
    plt.ylabel("Dichte")
    plt.savefig(f"{out_prefix}_hist.png", dpi=150)
    plt.close()
    
    # QQ-Plot gegen Standardnormal (scipy probplot)
    plt.figure()
    stats.probplot(z, dist="norm", plot=plt)
    plt.title(f"QQ-Plot vs. N(0,1), N={N}")
    plt.savefig(f"{out_prefix}_qq.png", dpi=150)
    plt.close()