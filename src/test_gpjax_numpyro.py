#%%
# Based on: https://docs.jaxgaussianprocesses.com/_examples/numpyro_integration/
# from examples.utils import use_mpl_style
import gpjax as gpx
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
)

config.update("jax_enable_x64", True)

# use_mpl_style()
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

key = jr.key(123)
keys = jr.split(key, 4)
# %%
N = 200

x = jnp.sort(jr.uniform(keys[0], shape=(N, 1), minval=0.0, maxval=10.0), axis=0)

# True parameters for the linear trend
true_slope = 0.45
true_intercept = 1.5

# Structured residual signal captured by the GP
slow_period = 6.0
fast_period = 0.8
amplitude_envelope = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x / slow_period)
modulated_periodic = amplitude_envelope * jnp.sin(2 * jnp.pi * x / fast_period)
high_frequency_component = 0.3 * jnp.cos(2 * jnp.pi * x / 0.35)
localised_bump = 1.2 * jnp.exp(-0.5 * ((x - 7.0) / 0.45) ** 2)

linear_trend = true_slope * x + true_intercept
residual_signal = modulated_periodic + high_frequency_component + localised_bump
signal = linear_trend + residual_signal

# Observations with homoscedastic noise
observation_noise = 0.7
y = signal + observation_noise * jr.normal(keys[1], shape=x.shape)

D = gpx.Dataset(X=x, y=y)

fig, ax = plt.subplots()
ax.plot(x, y, "o", label="Observations", color=cols[0])
ax.plot(x, signal, "--", label="True Signal", color=cols[1])
ax.legend()
# %%
# Priors are defined as NumPyro distributions and sampled directly inside the model
# function below. GPJax parameter constructors accept raw JAX arrays from
# numpyro.sample, so no special registration step is needed.
def model(X, Y, X_new=None):
    slope = numpyro.sample("slope", dist.Normal(0.0, 2.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 2.0))
    linear_component = slope * X + intercept

    residuals = Y - linear_component

    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    period = numpyro.sample("period", dist.LogNormal(0.0, 0.5))
    obs_noise = numpyro.sample("obs_noise", dist.LogNormal(0.0, 1.0))

    stationary_component = gpx.kernels.RBF(
        lengthscale=lengthscale, variance=variance
    )
    periodic_component = gpx.kernels.Periodic(
        lengthscale=lengthscale, period=period
    )
    kernel = stationary_component * periodic_component

    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=N, obs_stddev=obs_noise)
    posterior = prior * likelihood

    D_resid = gpx.Dataset(X=X, y=residuals)
    mll = gpx.objectives.conjugate_mll(posterior, D_resid)

    numpyro.factor("gp_log_lik", mll)

    if X_new is not None:
        latent_dist = posterior.predict(X_new, train_data=D_resid)
        f_new = numpyro.sample("f_new", latent_dist)
        f_new = f_new.reshape((-1, 1))

        y_noise = numpyro.sample(
            "y_noise",
            dist.Normal(0.0, obs_noise).expand(f_new.shape).to_event(f_new.ndim),
        )

        total_prediction = slope * X_new + intercept + f_new + y_noise
        numpyro.deterministic("y_pred", total_prediction)
        return total_prediction
#%%
nuts_kernel = NUTS(model)
# In practice, one should run more samples from multiple chains.
mcmc = MCMC(
    nuts_kernel,
    num_warmup=500,
    num_samples=500,
)
mcmc.run(keys[2], x, y)
mcmc.print_summary()
# %%
samples = mcmc.get_samples()
predictive = Predictive(
    model,
    posterior_samples=samples,
    return_sites=["y_pred"],
)

x_test = jnp.linspace(-0.5, 10.5, 200).reshape(-1, 1)
predictions = predictive(keys[3], x, y, X_new=x_test)
y_pred = predictions["y_pred"]

mean_prediction = jnp.mean(y_pred, axis=0)
lower, upper = jnp.percentile(y_pred, jnp.array([2.5, 97.5]), axis=0)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.5, label="Observations", color=cols[0])
ax.plot(x, signal, "--", label="True Signal", color=cols[0])

ax.plot(x_test, mean_prediction, "-", label="Posterior Mean", color=cols[1])
ax.fill_between(
    x_test.flatten(),
    lower.flatten(),
    upper.flatten(),
    color=cols[1],
    alpha=0.2,
    label="95% Credible Interval",
)
ax.legend()

# %% 
# %load_ext watermark
# %watermark -n -u -v -iv -w -a "Thomas Pinder"
# %%
