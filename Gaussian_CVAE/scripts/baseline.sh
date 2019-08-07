cd ..
python ~/Github/Fully_factorizable_CVAE/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs.json \
	--batch_size 50  \
    --n_epochs 10 \
    --data_type 'synthetic' \
    --dataloader 'datasets.dataloader.make_synthetic_data' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001
