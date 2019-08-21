cd ..
python ~/Github/cookiecutter/Gaussian_CVAE/Gaussian_CVAE/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results_projected/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs_proj.json \
	--batch_size 64  \
    --num_batches 1000  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.synthetic_projected.ProjectedSyntheticDataset' \
    --loss_fn 'losses.ELBO.synthetic_loss_no_mask' \
    --lr 0.001