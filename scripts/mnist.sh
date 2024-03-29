CVAE_train \
	--gpu_id 0 \
	--path_save_dir ./outputs/mnist_results/3 \
	--model_fn CVAE_testbed.models.CVAE_first.CVAE \
    --model_kwargs mnist_kwargs.json \
	--batch_size 50  \
    --n_epochs 10 \
    --data_type 'mnist' \
    --dataloader CVAE_testbed.datasets.dataloader.load_mnist_data \
    --loss_fn CVAE_testbed.losses.ELBO.calculate_loss \
    --lr 0.001