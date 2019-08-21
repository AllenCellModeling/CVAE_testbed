cd ..
python ~/Github/cookiecutter/Gaussian_CVAE/Gaussian_CVAE/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs_3.json \
	--batch_size 64  \
    --num_batches 500  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.synthetic.SyntheticDataset' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001