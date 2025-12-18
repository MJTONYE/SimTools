import numpy as np
import pandas as pd
from time import perf_counter

seed = 42
rng = np.random.default_rng(seed)

def estimate_pi(n_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """
    Schätzt pi mittels Viertelkreis-Monte-Carlo und liefert (pi_hat, se) zurück.

    Idee
    ----
    Wir ziehen n_samples Punkte (X_i, Y_i) gleichverteilt auf dem Einheitsquadrat
    [0, 1] x [0, 1]. Die Wahrscheinlichkeit, dass ein Punkt im Viertelkreis
    x^2 + y^2 <= 1 liegt, ist p = (Fläche Viertelkreis) / (Fläche Quadrat) = pi/4.
    Damit ist der MC-Schätzer für pi: pi_hat = 4 * p_hat.

    Parameter
    ---------
    n_samples : int
        Anzahl der Zufallspunkte (Stichprobengröße N).
    rng : np.random.Generator
        Zufallszahlengenerator (z. B. np.random.default_rng(seed)).

    Returns
    -------
    pi_hat : float
        Schätzer für pi (4 * Trefferquote).
    se : float
        Standardfehler von pi_hat aus der Binomialnäherung:
        SE(pi_hat) = 4 * sqrt(p_hat * (1 - p_hat) / N).
    """
    # 1) Ziehe N Punkte im Quadrat [0,1]^2 (vektorisiert)
    x = rng.random(n_samples)
    y = rng.random(n_samples)

    # 2) Prüfe, ob die Punkte im Viertelkreis liegen
    inside = x**2 + y**2 <= 1.0

    # 3) Trefferquote p_hat
    hits = inside.sum()
    p_hat = hits / n_samples

    # 4) MC-Schätzer für pi
    pi_hat = 4.0 * p_hat

    # 5) Standardfehler aus Binomial-Näherung
    se = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / n_samples)

    return pi_hat, se
print("Implementiere deinen Pi-Schätzer hier!")
pass

    


seed = 42
rsseq = np.random.SeedSequence(seed)

def replicate(
    Ns=(10**3, 10**4, 10**5),
    R=20,
    rsseq=rsseq,
    estimator=estimate_pi,          # <- dein Schätzer
    **estimator_kwargs,            
):
    """
    Repliziert Pi-Schätzungen und aggregiert laufend, ohne großes Zwischen-DataFrame.
    Gibt (df, summ) zurück; df ist None, außer keep_rows=True.

    - mu: Mittelwert von pi_hat über R Replikate (pro N)
    - sd: Stichproben-StdAbw. (ddof=1)
    - se_of_mean: sd/sqrt(R)
    - time_s: mittlere Laufzeit pro Replikat
    - ci_lo/ci_hi: 95%-CI um den Mittelwert
    """


    summ_rows = []

    for N in Ns:
        print(N)
        # einmalig R unabhängige Seeds erzeugen (effizienter als in innerer Schleife)
        children = rsseq.spawn(R)

        
        mean_pi = 0.0
        m2 = 0.0         # Summe der quadrierten Abweichungen
        mean_time = 0.0

        for r, child in enumerate(children):
            rng = np.random.default_rng(child)
            t0 = perf_counter()
            pi_hat, se = estimator(N, rng, **estimator_kwargs)
            dt = perf_counter() - t0

            k = r + 1
            # Online-Update für Mittelwert/Varianz von pi_hat
            delta = pi_hat - mean_pi
            mean_pi += delta / k
            m2 += delta * (pi_hat - mean_pi)

            # Online-Update für mittlere Zeit
            mean_time += (dt - mean_time) / k

        sd = np.sqrt(m2 / (R - 1)) if R > 1 else np.nan
        se_of_mean = sd / np.sqrt(R) if R > 0 else np.nan

        summ_rows.append({
            "N": N,
            "mu": mean_pi,
            "sd": sd,
            "se_of_mean": se_of_mean,
            "time_s": mean_time,
            "ci_lo": mean_pi - 1.96 * se_of_mean,
            "ci_hi": mean_pi + 1.96 * se_of_mean,
        })

    summ = pd.DataFrame(summ_rows)
    return summ