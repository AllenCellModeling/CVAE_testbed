cd ..
python ~/Github/Fully_factorizable_CVAE/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/mnist_results/3 \
	--model_fn models.CVAE_first.CVAE \
    --model_kwargs mnist_kwargs.json \
	--batch_size 50  \
    --n_epochs 10 \
    --data_type 'mnist' \
    --dataloader 'datasets.dataloader.load_mnist_data' \
    --loss_fn 'losses.ELBO.calculate_loss' \
    --lr 0.001