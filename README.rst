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


A research testbed on conditional variational autoencoders using Gaussian distributions as an input. We are interested in arbitrary conditioning of a CVAE and finding the relationships between information passing through the latent dimension bottlneck and the input dimensions. Our goal is to generate a fully factoriazable probabilistic model of structures in a cell.

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

Tests
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

Usage
--------

* Run baseline model. 

.. code-block:: bash

    $ cd scripts

.. code-block:: bash

    $ ./baseline.sh

This model takes a set of independent Gaussian distributions as an input. Specify the number of input dimensions 'x_dim' in baseline_kwargs.json

Specifying 2 input dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63389668-53ff6000-c361-11e9-9bab-c1534c4bfb8b.png
   :width: 750px
   :scale: 100 %
   :align: center

This plot can be viewed in outputs/baseline_results. The first component is the train and test loss. The other 3 plots are encoding tests of the model in the presence of different sets of conditions. 0 (blue) implies that no conditions are provided, and thus the model uses 2 latent dimensions in order to encode the information. 1 (orange) implies that one condition is provided, meaning the model needs only 1 latent dimension to encode the information. Finally, 2 (green) means that both conditions are provided, implying that the model needs no dimensions to encode the information, i.e all the information about the input data has been provided via the condition. 

Similarly, specifying 4 input dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63390327-8e69fc80-c363-11e9-93e0-219b6044774d.png
   :width: 750px
   :scale: 100 %
   :align: center

specifying 6 input dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63390404-d7ba4c00-c363-11e9-99db-663530743e3e.png
   :width: 750px
   :scale: 100 %
   :align: center

and so on.

* Run swiss roll baseline model. This model will take the swiss roll dataset as an input. 

.. code-block:: bash

    $ ./baseline_swissroll.sh

The swiss roll dataset is parametrized as:

.. math:: x = \phi \cos(\phi)
.. math:: y = \phi \sin(\phi)
.. math:: z = \psi

Despite having 3 dimensions, it is parametrized by 2 dimensions. Running this script gives

.. image:: https://user-images.githubusercontent.com/40371793/63390553-6333dd00-c364-11e9-9d41-c3c13a2c049b.png
   :width: 750px
   :scale: 100 %
   :align: center

This plot can be viewed in outputs/baseline_results_swissroll. We observe that on providing 0 conditions (blue), the model uses only 2 dimensions in the latent space, indicating


   
* View results in outputs/baseline_results folder. 



* Run projected baseline model. This model will take a set of independent Gaussian distributions as an input and project to a higher dimension. Specify the number of input dimensions 'x_dim' and number of projected dimensions 'projection_dim' in baseline_kwargs_proj.json

.. code-block:: bash

    $ ./baseline_projected.sh

* View results in outputs/baseline_results_projected folder. 

* Run swiss roll baseline model. This model will take the swiss roll dataset as an input. 

.. code-block:: bash

    $ ./baseline_swissroll.sh

* View results in outputs/baseline_results_swissroll folder. 

* Run compare_models.py to compare results across output folders

To-do list
----------

- [ ] Repo name change
- [ ] train real data

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
