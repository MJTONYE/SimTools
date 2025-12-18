import numpy as np

#uebung 4
def g1(x):
    return np.sqrt(x)

def g2(x):
    return np.exp(-x**2)

def g3(x):
    out = np.sin(np.pi * x) / (np.pi * x)
    out = np.where(x == 0, 1.0, out)
    return out

def uniform_sampler(N, rng):
    return rng.random(N)
  

def mc_expectation(g, sampler, N, rng):
    """
    Schätzt E[g(X)] = Integral of g(x) f(x) dx via Monte Carlo.
    
    Returns
    -------
    mu : float
        MC-Schätzer (Mittelwert der g(X_i)).
    se : float
        Standardfehler (sd/sqrt(n)).
    """
    x = sampler(N, rng)
    gx = g(x)
    mu = gx.mean()
    se = gx.std(ddof=1) / np.sqrt(N)
    return mu, se
