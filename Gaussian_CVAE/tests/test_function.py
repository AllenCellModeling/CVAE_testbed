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
from Gaussian_CVAE.main_train import get_model, str_to_object, get_args, make_plot_encoding
import torch
from torch import optim

def test_synthetic_data():
    """Test synthetic data size"""
    model_kwargs = {'x_dim': 2}
    X, C, ind = SyntheticDataset(2, 10, model_kwargs, shuffle = True, train=False).get_all_items()
    assert (X.size()[-1]*2 == C.size()[-1])
    assert (X.size()[-1] == model_kwargs['x_dim'])

def test_synthetic_baseline():
    """
    Test that synthetic data can pass through network
    """
    args = get_args()
    load_data = str_to_object(args.dataloader)
    X_train, C_train, Cond_indices_train,_ = load_data(args.num_batches, args.batch_size, args.model_kwargs, corr=False, train=True, mask=True).get_all_items()
    X_test, C_test, Cond_indices_test = load_data(args.num_batches, args.batch_size, args.model_kwargs, corr=False, train=False, mask=True).get_all_items()
    run = str_to_object('Gaussian_CVAE.run_models.run_synthetic.run_synthetic')

    device = (
        torch.device('cuda', args.gpu_id) if torch.cuda.is_available()
        else torch.device('cpu')
    )

    model = get_model(args.model_fn, args.model_kwargs).to(device)
    # print('model', model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # print('optimizer', opt)
    loss_fn = str_to_object(args.loss_fn)
    stats, stats_per_dim = run(args, X_train, C_train, Cond_indices_train, X_test, C_test, Cond_indices_test,
                    args.n_epochs, args.loss_fn, model, opt, args.batch_size, args.gpu_id, args.model_kwargs)

    make_plot_encoding(args, model, stats)

