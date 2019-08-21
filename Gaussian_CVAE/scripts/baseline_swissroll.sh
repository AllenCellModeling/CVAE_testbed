cd ..
python ~/Github/cookiecutter/Gaussian_CVAE/Gaussian_CVAE/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results_swissroll/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_swissroll_kwargs.json \
	--batch_size 64  \
    --num_batches 1000  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.swiss_roll.SwissRoll' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001