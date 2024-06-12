# COALA

COALA (COnservation study ALgorithm using Agent-based modelling) is a package for modelling the adoption of conservation initiatives.

## Files

The Python scripts presented in this repository are

- **example.py**: This script provides an example of how to run an end-to-end analysis of a data set using COALA.

- **COALA.py**: This script contains the code describing both of our agent-based models: a stochastic and a deterministic version.

- **mcmc.py**: This script can be used to run runs the MCMC using the [emcee package](https://emcee.readthedocs.io/en/stable/). It's called by example.py. In our paper presenting COALA, we also use a sparse inference tool, [SLInG](https://github.com/ASoelvsten/SLInG). The source code for [SLInG](https://github.com/ASoelvsten/SLInG) can be found on GitHub.

- **posteriorpredictive.py**: This script performs the posterior predictive tests, producing summary plots as well as plots for projections into the future.

- **mockdata.py**: To run COALA, one needs data. This script creates synthetic data for adoption and saves it as summary\_mock\_XD.npy (this is the standard name of the file that COALA looks for).

The folder MOCK01 contains the output obtained from running the analysis based on the synthetic data in **summary\_mock\_XD.npy** and **truth.npy**.



