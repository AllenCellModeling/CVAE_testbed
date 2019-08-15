=====================
CVAE research testbed
=====================

.. image:: https://travis-ci.org/AllenCellModeling/Gaussian_CVAE.svg?branch=master
        :target: https://travis-ci.org/AllenCellModeling/Gaussian_CVAE
        :alt: Build Status
        
.. image:: https://readthedocs.org/projects/gaussian-cvae/badge/?version=latest
        :target: https://gaussian-cvae.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/AllenCellModeling/Gaussian_CVAE/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/AllenCellModeling/Gaussian_CVAE
        :alt: Codecov Status


A research testbed on conditional variational autoencoders using Gaussian distributions as an input.


* Free software: Allen Institute Software License

* Documentation: https://Gaussian-CVAE.readthedocs.io.

Organization
--------

The project has the following structure::

    Gaussian_CVAE/
      |- README.rst
      |- setup.py
      |- requirements.txt
      |- tox.ini
      |- Makefile
      |- MANIFEST.in
      |- HISTORY.rst
      |- CHANGES.rst
      |- AUTHORS.rst
      |- LICENSE
      |- docs/
         |- ...
      |- Gaussian_CVAE/
         |- __init__.py
         |- main_train.py
         |- baseline_kwargs.json
         |- mnist_kwargs.json
         |- tests/
            |- __init__.py
            |- test_function.py
            |- example.sh
         |- datasets/
            |- __init__.py
            |- dataloader.py
            |- synthetic.py
         |- losses/
            |- __init__.py
            |- ELBO.py
         |- metrics/
            |- __init__.py
            |- blur.py
            |- calculate_fid.py
            |- inception.py
            |- visualize_encoder.py
         |- models/
            |- __init__.py
            |- CVAE_baseline.py
            |- CVAE_first.py
            |- sample.py
         |- run_models/
            |- __init__.py
            |- generative_metric.py
            |- run_synthetic.py
            |- run_test_train.py
            |- test.py
            |- train.py
         |- scripts/
            |- __init__.py
            |- baseline.sh
            |- mnist.sh
            |- compare_models.py
         |- utils/
            |- __init__.py
            |- compare_plots.py

Example run
--------

* Create conda environment

.. code-block:: bash

    $ conda create --name cvae python=3.7

* Activate conda environment :

.. code-block:: bash

    $ conda activate cvae

* Install requirments in setup.py

.. code-block:: bash

    $ pip install -e .[all]

* Run baseline model

.. code-block:: bash

    $ cd scripts

.. code-block:: bash

    $ ./baseline.sh

* View results in outputs/baseline_results folder. 

* Run compare_models.py to compare results across output folders

To-do list
----------

- [x] Projecttion of synthetic data
- [x] Blog post on usage
- [ ] Mask projected data with no conditions
- [ ] Mask projected data with all conditions
- [ ] Model loader and plotting 
- [ ] Repo name change
- [ ] train real data

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
