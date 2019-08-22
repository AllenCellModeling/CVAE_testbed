#!/bin/sh
cd CVAE_testbed
python ~/Github/cookiecutter/CVAE_testbed/CVAE_testbed/main_train.py \
	--gpu_id 0 \
	--path_save_dir ./tests/test_results/ \
	--model_fn models.CVAE_baseline.CVAE \
    --model_kwargs baseline_kwargs.json \
	--batch_size 30  \
    --num_batches 3  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader 'datasets.synthetic.SyntheticDataset' \
    --loss_fn 'losses.ELBO.synthetic_loss' \
    --lr 0.001