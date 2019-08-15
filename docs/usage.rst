=====
Usage
=====

* To use Gaussian_CVAE in a project::

    import Gaussian_CVAE

Here, we provide a description of what Gaussian_CVAE can do:

* Train a conditional variational autoencoder (CVAE) on independent synthetic data. 

We generate synthetic data that is sampled from a multivairate normal
distribution such that each dimension is independent of the other. This synthetic data (X) is N dimensional and can be 
represented as a matrix (B*N) of the form 

    X11 X12 ... X1N
    ...
    XB1 XB2 ... XBN

where N is the number of dimensions of X and B is the number of mini-batches. 
Our condition matrix C (B*2N) is of the form

    X11 X12 ... X1N M11 M12 ... M1N
    ...             ...
    XB1 XB2 ... XBN MB1 MB2 ... MBN

where M11..M1N is a mask indicator that indicates whether the corresponding elements X11..X1N
have been masked or not. So, if N is 2, and B is 4, this is what that could look like

    0.4 0.8 1 1
    0 0.3 0 1
    0.7 0 1 0
    0 0 0 0

where row 1 has both conditions, row 2 has condition 2, row 3 has condition 1 and row 4 has no condition. 

When we train the neural network (see Gaussian_CVAE/models/CVAE_baseline.py), we ensure that all permutations 
of conditions are present in every mini-batch. The kwargs that we need to specify as an input to the neural
network can be found in

    baseline_kwargs.json

These kwargs include 'x_dim' (the number of dimensions of X), 'c_dim' (the number of dimensions of C (twice of X)), 
'enc_layers' (the size of each linear layer in the encoder network) and 'dec_layers' (the size of each linear layer in the decoder network).
We pass other parameters like batch size, number of batches, learning rate, loss function, model etc. as arguments via shell scripts. This can
be found in

    scripts/baseline.sh

Running this script will save the trained model and other metrics like distortion (MSE loss) and rate (KL divergence from a normal prior). 
Together, the rate and distortion make up the loss function called ELBO (Evidence Lower Bound Objective). The script will also
save images of the encoding latent space for every possible condition and KL divergence per latent dimension. These results can be found in 

    outputs/baseline_results

Providing independent Gaussian samples means that the CVAE only needs as many latent dimensions as there are input dimensions in 
order to learn the distribution of the data. Thus, if X contains 2 independent dimensions and we have 200 latent dimensions, the CVAE
will learn to use only 2 latent dimensions to reconstruct the input data. However, if we provide one of the dimensions of X as a condition 
, then the CVAE will only need to use 1 dimension in the latent space -- the other dimension is no longer required since the CVAE has been 
given that information via the condition. Similarly, if we give the network both dimensions as a condition, then the CVAE uses no dimension
in the latent space. 

This can be repeated for any number of dimensions by varying 'x_dim' in baseline_kwargs.json.

* Train a conditional variational autoencoder (CVAE) on synthetic data projected onto higher dimensional space. 

We generate synthetic data where we can control the dimensions that are independent of each other (or not) via projection to 
a higher dimensional space. For example, if our input data X (1*2) is 

    0.4 0.8

then, we can project this into higher dimensional space by first padding with 0's 

    0.4 0.8 0 0 

rotating it 

    0.4
    0.8 
    0 
    0

and then multiplying by a projection matrix (P) where P is 

    A1 0 0 0
    A2 0 0 0 
    0 B1 0 0
    0 B2 0 0

This will give

    A1*0.4
    A2*0.4
    B1*0.8
    B2*0.8

since our projection matrix (P) is linearly independent, our projected space also retains this linear independence. However, if P 
was to be rewritten as 

    A1 A3 0 0
    A2 0 0 0 
    0 B1 0 0
    0 B2 0 0

then the first row is no longer linearly independent. Such a projection operation can be done to any number of dimensions with any number
of correlations. These parameters can be set in 

    baseline_kwargs_proj.json

where 'x_dim' is the number of dimensions of X and 'projection_dim' is the number of dimensions of the new projected X. Run the script

    baseline2.sh

to train a model with projected synthetic data. If all of the dimensions in the projected space are independent, then the CVAE will still learn
that N latent dimensions (N is the dimension of X) are required to represent the distribution of the projected synthetic data. However, 
if some of the dimensions are correlated, then it will require <=N dimensions in the latent space. 





