=====================
CVAE_research_testbed
=====================

.. image:: https://travis-ci.com/ritvikvasan/Gaussian_CVAE.svg?branch=master
        :target: https://travis-ci.com/AllenCellModeling/Gaussian_CVAE
        :alt: Build Status

.. image:: https://readthedocs.org/projects/Gaussian-CVAE/badge/?version=latest
        :target: https://Gaussian-CVAE.readthedocs.io/en/latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/ritvikvasan/Gaussian_CVAE/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/AllenCellModeling/Gaussian_CVAE
        :alt: Codecov Status


A research testbed on conditional variational autoencoders using Gaussian distributions as an input


* Free software: Allen Institute Software License

* Documentation: https://Gaussian-CVAE.readthedocs.io.


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

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
