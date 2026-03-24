"""Intertemporal shallow lake model for EMA Workbench labs.

Wraps ema_workbench.examples.lake_models.lake_problem_intertemporal so that
the 100 annual release decisions are received as individual keyword arguments
l0, l1, ..., l99 (matching the lever names defined in the notebooks).
"""

import numpy as np
from ema_workbench.examples.lake_models import lake_problem_intertemporal


def lake_problem(
    b: float = 0.42,
    q: float = 2.0,
    mean: float = 0.02,
    stdev: float = 0.001,
    delta: float = 0.98,
    alpha: float = 0.41,
    nsamples: int = 150,
    **kwargs,
):
    """EMA Workbench function model for the intertemporal lake problem.

    Uncertain parameters
    --------------------
    b      : decay rate for P in lake
    q      : recycling exponent
    mean   : mean of natural inflows
    stdev  : std dev of natural inflows
    delta  : future utility discount rate

    Constants
    ---------
    alpha    : utility weight on pollution control
    nsamples : Monte Carlo samples for natural inflow

    Levers
    ------
    l0 … l99 : annual anthropogenic phosphorus release (each in [0, 0.1])
    """
    # Collect lever values in order
    n_years = 100
    decisions = np.array([kwargs.get(f"l{i}", 0.0) for i in range(n_years)])

    max_p, utility, inertia, reliability = lake_problem_intertemporal(
        b=b,
        q=q,
        mean=mean,
        stdev=stdev,
        delta=delta,
        alpha=alpha,
        n_samples=nsamples,
        decisions=decisions,
    )

    return max_p, utility, inertia, reliability
