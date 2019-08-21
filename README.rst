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

* Run projected baseline model. This model will take a set of independent Gaussian distributions as an input and project it to a higher dimension. Specify the number of input dimensions 'x_dim' and number of projected dimensions 'projection_dim' in baseline_kwargs_proj.json

.. code-block:: bash

    $ ./baseline_projected.sh

Projecting 2 dimensions to 4 dimensions gives 

.. image:: https://user-images.githubusercontent.com/40371793/63390404-d7ba4c00-c363-11e9-99db-663530743e3e.png
   :width: 750px
   :scale: 100 %
   :align: center

This plot can be viewed in outputs/baseline_results_projected. The model uses only 2 dimensions in the latent space to encode information from a 4 dimensional input dataset. 

Similarly, projecting 2 dimensions to 8 dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63446020-53f97180-c3ee-11e9-8f26-5ea0489f68a4.png
   :width: 750px
   :scale: 100 %
   :align: center

projecting 4 dimensions to 8 dimensions gives 

.. image:: https://user-images.githubusercontent.com/40371793/63446173-9327c280-c3ee-11e9-95c9-ed04fdab0522.png
   :width: 750px
   :scale: 100 %
   :align: center

and so on. 

* Run projected baseline model with a mask. This model will take a set of independent Gaussian distributions, project it to a higher dimensional space and then mask a percentage of the input data during training. 

.. code-block:: bash

    $ ./baseline_projected_with_mask.sh

Here we need to update the loss function to not penalize masked data. Without doing this, projecting 2 dimensions to 8 dimensions with 50% of the input data masked gives 

.. image:: https://user-images.githubusercontent.com/40371793/63446885-dafb1980-c3ef-11e9-89cb-6389a38dfaca.png
   :width: 750px
   :scale: 100 %
   :align: center

After updating the loss, doing the same thing gives

.. image:: https://user-images.githubusercontent.com/40371793/63446987-10076c00-c3f0-11e9-9b99-72b67c3592fa.png
   :width: 750px
   :scale: 100 %
   :align: center

Despite 50% of the data being masked, the model uses 2 dimensions in the latent space.

* Run swiss roll baseline model. This model will take the swiss roll dataset as an input. 

.. code-block:: bash

    $ ./baseline_swissroll.sh

The swiss roll dataset is parametrized as:

.. math:: x = \phi \cos(\phi)
.. math:: y = \phi \sin(\phi)
.. math:: z = \psi

Despite having 3 dimensions, it is parametrized by 2 dimensions. Running this script gives

.. image:: https://user-images.githubusercontent.com/40371793/63444333-3ecf1380-c3eb-11e9-8597-e61c5e056744.png
   :width: 750px
   :scale: 100 %
   :align: center

This plot can be viewed in outputs/baseline_results_swissroll. We observe that given 0 conditions (blue), the model gets embedded into only dimensions in the latent space. Providing 1 condition (X) is no different then providing 2 conditions (X and Y) since both X and Y are parameterized by only 1 dimension. Finally, providing both conditions means that no information passes throught the bottleneck and the model encodes no information. 

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
