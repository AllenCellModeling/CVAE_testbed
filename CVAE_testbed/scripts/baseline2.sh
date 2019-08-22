cd ..
python ~/Github/cookiecutter/CVAE_testbed/CVAE_testbed/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results2/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs_proj.json \
	--batch_size 64  \
    --num_batches 1000  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.synthetic_projected.ProjectedSyntheticDataset' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001