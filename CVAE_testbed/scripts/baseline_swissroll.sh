cd ..
python ~/Github/cookiecutter/CVAE_testbed/CVAE_testbed/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results_swissroll/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_swissroll_kwargs.json \
	--batch_size 64  \
    --num_batches 500  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.swiss_roll.SwissRoll' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001