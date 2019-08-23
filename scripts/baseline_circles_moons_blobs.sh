CVAE_train \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results_circles_moons_blobs/ \
	--model_fn CVAE_testbed.models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs_circles_moons_blobs.json \
	--batch_size 64  \
    --num_batches 500  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader CVAE_testbed.datasets.circles_moons_blobs.CirclesMoonsBlobs \
    --loss_fn CVAE_testbed.losses.ELBO.synthetic_loss_no_mask \
    --lr 0.001