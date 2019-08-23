CVAE_train \
	--gpu_id 0 \
	--path_save_dir ./outputs/baseline_results_swissroll/ \
	--model_fn CVAE_testbed.models.CVAE_residual_extended.CVAE \
    --model_kwargs residual_extended_kwargs.json \
	--batch_size 64  \
    --num_batches 500  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader CVAE_testbed.datasets.swiss_roll.SwissRoll \
    --loss_fn CVAE_testbed.losses.ELBO.synthetic_loss_no_mask \
    --lr 0.0005
