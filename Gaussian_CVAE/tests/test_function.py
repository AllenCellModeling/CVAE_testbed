#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

from Gaussian_CVAE.datasets.dataloader import make_synthetic_data


def test_make_synthetic_data():
    """Make some synthetic data"""
    model_kwargs = {'x_dim': 2}
    X, C, ind = make_synthetic_data(2, 10, model_kwargs, shuffle = True)
    assert (X.size()[-1]*2 == C.size()[-1])
    assert (X.size()[-1] == model_kwargs['x_dim'])
