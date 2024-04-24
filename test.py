# %%

from jax.lib import xla_bridge
TF_CPP_MIN_LOG_LEVEL=0 
print(xla_bridge.get_backend().platform)
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# import arviz as az
# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
# import scipy

import jax.numpy as jnp
from jax import nn, lax, random
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_platform("gpu")
numpyro.set_host_device_count(2)
# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
rng_key = random.PRNGKey(2)
# %%
# Define model
def model(X,treatment, Y=None):
    bX = numpyro.sample('bX', dist.Normal(0,1).expand([X.shape[1]]))
    bTreat = numpyro.sample('bTreat', dist.Normal(0,1))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('Y',dist.Normal(X @ bX + bTreat * treatment,sigma), obs=Y)

# %%
# Generate artificial data
k = 4
N = 100000
X = np.random.normal(size=[N, k])
treatment = np.random.choice([0,1],N)

# Condition some coefficient to specific quantities
# Note: This is not required but useful in some cases
coefTrue = {'bTreat':1.}
condition_model = numpyro.handlers.condition(model, data=coefTrue)
prior_predictive = Predictive(condition_model, num_samples=1)

# Sample Y from model
rng_key, rng_key_ = random.split(rng_key)
prior_samples = prior_predictive(rng_key_, X=X, treatment=treatment)

Y = prior_samples['Y'][0,:]

true_bTreat = prior_samples['bTreat'][0]
print('true_bTreat:',true_bTreat)
# %%
# Use numpyro Predictive to calculate treatment effects
truePredictNoTreat = prior_predictive(rng_key_,X=X, treatment=np.zeros_like(treatment))
truePredictTreat = prior_predictive(rng_key_,X=X, treatment=np.ones_like(treatment))

# Calcualte treatment difference
trueTreatDiff = truePredictTreat['Y']-truePredictNoTreat['Y']
print('Mean sample treatment effect:',trueTreatDiff.mean())
# %%
# Estimate model using numpyro MCMC
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_samples=5000, num_warmup=1000, num_chains=2)
mcmc.run(rng_key_, X=X, treatment=treatment, Y=Y)
mcmc.print_summary()
# %%
