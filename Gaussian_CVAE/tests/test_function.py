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

from Gaussian_CVAE.datasets.synthetic import SyntheticDataset
import subprocess
import os

def test_synthetic_data():
    """Test synthetic data size"""
    model_kwargs = {'x_dim': 2}
    X, C, ind = SyntheticDataset(2, 10, model_kwargs, shuffle = True).get_all_items()
    assert (X.size()[-1]*2 == C.size()[-1])
    assert (X.size()[-1] == model_kwargs['x_dim'])

def test_run():
    """
    Test that synthetic data can pass through network
    """
    wd = os.getcwd()
    os.chdir(wd + '/Gaussian_CVAE/tests/')

    subprocess.call(['./example.sh'], shell=True)
    # os.system("sh example.sh")

