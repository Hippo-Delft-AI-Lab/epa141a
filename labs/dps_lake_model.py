"""DPS (Direct Policy Search) version of the shallow lake model for Lab 6.

The policy is parameterised by two radial basis functions (RBFs) with parameters
c1, c2, r1, r2, w1.  These are the levers optimised during MORDM.
"""

import numpy as np
from ema_workbench.examples.lake_models import lake_problem_dps


def lake_model(
    b: float = 0.42,
    q: float = 2.0,
    mean: float = 0.02,
    stdev: float = 0.001,
    delta: float = 0.98,
    alpha: float = 0.4,
    n_samples: int = 100,
    myears: int = 100,
    c1: float = 0.25,
    c2: float = 0.25,
    r1: float = 0.5,
    r2: float = 0.5,
    w1: float = 0.5,
):
    """EMA Workbench function model for the DPS lake problem.

    Uncertain parameters
    --------------------
    b      : decay rate for P in lake
    q      : recycling exponent
    mean   : mean of natural inflows
    stdev  : std dev of natural inflows
    delta  : future utility discount rate

    Levers (RBF policy parameters)
    --------------------------------
    c1, c2 : RBF centres
    r1, r2 : RBF radii
    w1     : weight of first RBF  (w2 = 1 - w1)
    """
    max_p, utility, inertia, reliability = lake_problem_dps(
        b=b,
        q=q,
        mean=mean,
        stdev=stdev,
        delta=delta,
        alpha=alpha,
        n_samples=n_samples,
        myears=myears,
        c1=c1,
        c2=c2,
        r1=r1,
        r2=r2,
        w1=w1,
    )

    return max_p, utility, inertia, reliability
