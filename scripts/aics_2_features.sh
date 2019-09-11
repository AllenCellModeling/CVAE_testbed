CVAE_train \
    --gpu_id 1 \
    --path_save_dir ./outputs/aics_features/2_features/ \
    --model_fn CVAE_testbed.models.CVAE_baseline.CVAE \
    --model_kwargs '{"x_dim": 2, "c_dim": 4, "enc_layers": [2, 64, 64,64, 64, 64, 64, 512], "dec_layers": [512, 64, 64, 64, 64, 64, 64, 2]}'\
    --json_quilt_path '/home/ritvik.vasan/test/'\
    --config_path '/home/ritvik.vasan/config.json'\
    --binary_real_one_hot_parameters '{"binary_range": [0, 1],
            "binary_loss": "BCE",
            "real_range": [1, 103],
            "real loss": "MSE",
            "one_hot_range": [103, 159],
            "one_hot_loss": "CE"}'\
    --batch_size 64  \
    --beta_vae 1 \
    --num_batches 1000  \
    --n_epochs 30 \
    --data_type 'aics_features' \
    --dataloader CVAE_testbed.datasets.quilt_aics_features.QuiltAicsFeatures \
    --loss_fn CVAE_testbed.losses.ELBO.synthetic_loss_no_mask \
    --lr 0.001
