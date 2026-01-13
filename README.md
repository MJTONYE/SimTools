# Lynx Population Viability (Monte Carlo)

This project contains a Monte Carlo simulation studying the viability of a lynx population.

## Objective
Estimate how many adult lynx must be released at time t = 0 so that the population reaches at least 250 individuals after 10 years with a probability of at least 95%.

## Model overview
- Annual, discrete-time simulation
- Stochastic survival and reproduction
- Age-structured population (adults + juveniles)
- No juveniles present at t = 0 (conservative assumption)
- Closed population (no immigration)
- Extinction risk threshold: 250 individuals

## Code
- `src/lynx_simulation.py`: main simulation script

## Requirements
- Python 3
- numpy

## Run
```bash
python src/lynx_simulation.py

Add README
