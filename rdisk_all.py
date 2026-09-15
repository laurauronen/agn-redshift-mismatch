from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pymultinest.solve import solve
from pymultinest import Analyzer
import json
import os
import functools
from concurrent.futures import ProcessPoolExecutor
import time
from scipy.special import logsumexp

outdir = 'outdir_rdisk_all'

try:
    os.mkdir(outdir)
except OSError:
    pass

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def v(rat):
    # rat = r/Rs
    return 1 / np.sqrt(2 * (rat - 1))


def gamma(v):
    return 1 / np.sqrt(1 - v**2)


def z_rel(r, theta):
    # r in Rs
    vel = v(r)
    gamm = gamma(vel)
    zrel = gamm * (1 + vel * np.cos(theta)) - 1
    return zrel


def z_grav(r):
    # r in Rs
    inv = 1/r
    return np.sqrt(1 - inv) - 1


def coeff(r, theta):
    # redshift coefficient
    return (1 + z_rel(r, theta)) * (1 + z_grav(r))


def log_l(theta, sigma, zs, gw_posterior, r):
    # create model at AGN z_s + r
    dl_agn = cosmo.luminosity_distance(zs).value
    dl_eff = coeff(theta[0], np.pi) * dl_agn
    r.append(coeff(theta[0], np.pi))

    log_likelihood = -0.5 * (
        (dl_eff - np.mean(gw_posterior)) / np.std(gw_posterior)
    )**2

    if not np.isfinite(log_likelihood):
        return -1e100
    return log_likelihood


def prior(theta, min_r, max_r, min_tilt, max_tilt):
    # uniform prior on radius and theta
    return np.array([
        min_r * (max_r / min_r)**theta[0],
        #theta[0] * (max_r - min_r) + min_r,
        theta[1] * (max_tilt - min_tilt) + min_tilt,
    ])


n_params = 1

# set redshift
zs = 1.5

# set maximum and minimum radius of disk
rdisk = 1000
min_r = 0.1

# set maximum gas radius
gas_r = 50000

# set theta prior
max_tilt = np.pi
min_tilt = -np.pi

# set priors on the bbh distance
agn_prior = functools.partial(prior,
                              min_r=min_r,
                              max_r=rdisk,
                              min_tilt=min_tilt,
                              max_tilt=max_tilt)
gas_prior = functools.partial(prior,
                              min_r=rdisk,
                              max_r=gas_r,
                              min_tilt=min_tilt,
                              max_tilt=max_tilt)

ngrid = 10
pos = np.logspace(start=np.log10(3), stop=np.log10(rdisk), num=ngrid)
sigs = np.linspace(start=0.01, stop=0.5, num=ngrid)
tilts = np.linspace(start=-np.pi, stop=np.pi, num=ngrid)
dl_agn = cosmo.luminosity_distance(zs).value


def run_grid_point(args):
    i, j, k, rbbh, sig, tilt = args

    # set DL posterior width as percentage
    sigma_dl = sig * dl_agn

    # create posterior according to AGN redshift
    gas_posterior = np.random.normal(dl_agn, sigma_dl, 2000)

    # redshift according to grav/rel distortion
    gw_posterior = coeff(rbbh, tilt) * gas_posterior

    # blank ratio bank
    r = []

    # set functions for likelihood/prior
    wrapped_logl = functools.partial(log_l,
                                     sigma=sigma_dl,
                                     zs=zs,
                                     gw_posterior=gw_posterior,
                                     r=r)

    # set outdirs
    agndir = f'{outdir}/run_{i}_{j}_{k}_agn'
    gasdir = f'{outdir}/run_{i}_{j}_{k}_gas'
    os.mkdir(agndir)
    os.mkdir(gasdir)
    agn_prefix = f'{agndir}/inf'
    gas_prefix = f'{gasdir}/inf'

    # run MultiNest, disk prior
    agn_result = solve(
            LogLikelihood=wrapped_logl,
            Prior=agn_prior,
            n_dims=n_params,
            n_live_points=400,
            outputfiles_basename=agn_prefix,
            max_iter=10000,
            sampling_efficiency=0.3,
            evidence_tolerance=0.5,
            verbose=True,
            use_MPI=False
    )

    agn_analyzer = Analyzer(
            n_params=n_params,
            outputfiles_basename=agn_prefix
    )

    agn_stats = agn_analyzer.get_stats()
    agn_logZ = agn_stats['global evidence']
    agn_logZ_err = agn_stats['global evidence error']
    post = agn_analyzer.get_equal_weighted_posterior()
    r_samps = post[:, 0]
    t_samps = post[:, 1]
    coeff_samples = coeff(r_samps, t_samps)
    median_coeff = np.median(coeff_samples)

    # run MultiNest, gas prior
    gas_result = solve(
            LogLikelihood=wrapped_logl,
            Prior=gas_prior,
            n_dims=n_params,
            n_live_points=400,
            outputfiles_basename=gas_prefix,
            max_iter=10000,
            sampling_efficiency=0.3,
            evidence_tolerance=0.5,
            verbose=True,
            use_MPI=False
    )

    gas_analyzer = Analyzer(
            n_params=n_params,
            outputfiles_basename=gas_prefix
    )

    gas_stats = gas_analyzer.get_stats()
    gas_logZ = gas_stats['global evidence']
    gas_logZ_err = gas_stats['global evidence error']

    # compute and store bayes factor
    bayes = agn_logZ - gas_logZ
    err = np.sqrt(
            agn_logZ_err**2 +
            gas_logZ_err**2
            )

    return {
            'n': i * ngrid * 10 + j * ngrid + k + 1,
            'r_BBH': rbbh,
            'tilt': tilt,
            'sigma': sig,
            'logB_AGN_GAS': bayes,
            'logB_err': err,
            'median_R': median_coeff,
        }


if __name__ == '__main__':
    jobs = []
    start_time = time.perf_counter()

    for i, rbbh in enumerate(pos):
        for j, sig in enumerate(sigs):
            for k, tilt in enumerate(tilts):
            jobs.append((i, j, k, rbbh, sig, tilt))

    logb = {}

    with ProcessPoolExecutor(max_workers=8) as executor:
        for result in executor.map(run_grid_point, jobs):
            n = result.pop('n')
            logb[n] = result
            print(f"Finished {n}/{len(jobs)}")

    with open(f'{outdir}/results.json', 'w') as f:
        json.dump(logb, f, indent=4)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.6f} seconds")
