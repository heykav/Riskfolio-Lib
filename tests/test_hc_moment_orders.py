"""Regression tests: HRP/HERC must honour p_em / p_esm in cluster risk."""

import numpy as np
import pandas as pd
import pytest
import riskfolio as rp


@pytest.mark.parametrize(
    "rm, kw, func",
    [("EM", "p_em", rp.EvenMoment), ("ESM", "p_esm", rp.EvenSemiMoment)],
)
def test_hrp_uses_moment_order(rm, kw, func):
    rng = np.random.default_rng(0)
    Y = pd.DataFrame(rng.standard_t(4, (200, 2)) * [0.01, 0.02], columns=["A", "B"])

    port = rp.HCPortfolio(returns=Y, **{kw: 3})
    w = port.optimization(model="HRP", rm=rm, linkage="ward")

    a, b = (func(Y[c].values, p=3) for c in Y)
    expected = np.array([b / (a + b), a / (a + b)])
    np.testing.assert_allclose(w["weights"].values, expected, rtol=1e-8)
