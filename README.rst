=====================
CVAE research testbed
=====================

.. image:: https://travis-ci.org/AllenCellModeling/CVAE_testbed.svg?branch=master
        :target: https://travis-ci.org/AllenCellModeling/CVAE_testbed
        :alt: Build Status
        
.. image:: https://readthedocs.org/projects/gaussian-cvae/badge/?version=latest
        :target: https://gaussian-cvae.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/AllenCellModeling/CVAE_testbed/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/AllenCellModeling/CVAE_testbed
        :alt: Codecov Status


A research testbed on conditional variational autoencoders using Gaussian distributions as an input. We are interested in arbitrary conditioning of a CVAE and finding the relationships between information passing through the latent dimension bottlneck and the input dimensions. Our goal is to generate a fully factoriazable probabilistic model of structures in a cell.

* Free software: Allen Institute Software License

* Documentation: https://Gaussian-CVAE.readthedocs.io.

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

Usage: Synthetic data
--------

* Run all synthetic data models:

.. code-block:: bash

    $ cd scripts

.. code-block:: bash

    $ ./run_all_synthetic_datasets.sh

This will run mutliple synthetic experiments on 2 GPU's simultaneously. Each script can also be run individually. Here, we go through these one by one:

* Run baseline model. 

.. code-block:: bash

    $ cd scripts

.. code-block:: bash

    $ ./baseline.sh

This model takes a set of independent Gaussian distributions as an input. Specify the number of input dimensions 'x_dim' in baseline_kwargs.json

Specifying 2 input dimensions gives

![baseline_2](./scripts/outputs/baseline_results/encoding_test_plots.png)

This plot can be viewed in outputs/baseline_results. The first component is the train and test loss. The other 3 plots are encoding tests of the model in the presence of different sets of conditions. 0 (blue) implies that no conditions are provided, and thus the model uses 2 latent dimensions in order to encode the information. 1 (orange) implies that one condition is provided, meaning the model needs only 1 latent dimension to encode the information. Finally, 2 (green) means that both conditions are provided, implying that the model needs no dimensions to encode the information, i.e all the information about the input data has been provided via the condition. 

Similarly, specifying 4 input dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63390327-8e69fc80-c363-11e9-93e0-219b6044774d.png
   :width: 750px
   :scale: 100 %
   :align: center

specifying 6 input dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63449614-4f848700-c3f5-11e9-842e-40b07271a5ed.png
   :width: 750px
   :scale: 100 %
   :align: center

and so on.

* Run projected baseline model. This model will take a set of independent Gaussian distributions as an input and project it to a higher dimension. Specify the number of input dimensions 'x_dim' and number of projected dimensions 'projection_dim' in baseline_kwargs_proj.json

.. code-block:: bash

    $ ./baseline_projected.sh

Projecting 2 dimensions to 8 dimensions gives 

![baseline_proj_2_8](./scripts/outputs/baseline_results_projected/encoding_test_plots.png)

This plot can be viewed in outputs/baseline_results_projected. The model uses only 2 dimensions in the latent space to encode information from a 4 dimensional input dataset. 

Similarly, projecting 2 dimensions to 4 dimensions gives

.. image:: https://user-images.githubusercontent.com/40371793/63447464-eac72d80-c3f0-11e9-86c9-26df0b5ed8da.png
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

![baseline_swissroll](scripts/outputs/baseline_results_swissroll/encoding_test_plots.png)

This plot can be viewed in outputs/baseline_results_swissroll. We observe that given 0 conditions (blue), the model gets embedded into only dimensions in the latent space. Providing 1 condition (X) is no different then providing 2 conditions (X and Y) since both X and Y are parameterized by only 1 dimension. Finally, providing both conditions means that no information passes throught the bottleneck and the model encodes no information. 

* Run sklearn datasets model. This model will take the sklearn datasets like circles, moons and blobs as an input. 

.. code-block:: bash

    $ ./baseline_circles_moons_blobs.sh

The type of dataset (i.e. circles, moons, blobs or an s_curve) is specified in "sklearn_data" in baseline_kwargs_circles_moons_blobs.json. Running this file for blobs gives 

![blobs](scripts/outputs/loop_models_blobs/encoding_test_plots.png)

Similarly for moons gives 

![moons](scripts/outputs/loop_models_moons/encoding_test_plots.png)

This is how the original data maps to the latent space

.. image:: https://user-images.githubusercontent.com/40371793/63801095-61b66780-c8c4-11e9-9b59-d51be918211f.png
   :width: 750px
   :scale: 100 %
   :align: center

Similarly for an s_curve gives 

![s_curve](scripts/outputs/loop_models_s_curve/encoding_test_plots.png)

And for circles gives

![circles](scripts/outputs/loop_models_circles/encoding_test_plots.png)

* Run compare_models.py to compare results across output folders

* Visualize individual model runs or multiple model runs using the notebooks in CVAE_testbed/notebooks

Usage: AICS feature data
--------

* Run aics feature model. Here we pass 159 features (1 binary, 102 real and 56 one-hot features) through the CVAE

.. code-block:: bash

    $ cd scripts

.. code-block:: bash

    $ ./aics_features_simple.sh

Here is what the encoding looks like for a beta of 1

![aics_159_features_beta_1](scripts/outputs/aics_159_features_beta_1/encoding_test_plots.png)

There is no information passing through the information bottleneck, i.e. the KL divergence term is near 0 and the model is close to the autoencoding limit. 

We can vary beta and compare ELBO and FID scores in order to find the best model. 

Organization
--------

The project has the following structure::

    CVAE_testbed/
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
      |- CVAE_testbed/
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

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
