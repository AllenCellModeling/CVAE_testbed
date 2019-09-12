CVAE_train \
    --gpu_id 1 \
    --path_save_dir ./outputs/aics_159_features_beta_1/ \
    --model_fn CVAE_testbed.models.CVAE_baseline.CVAE \
    --model_kwargs '{"x_dim": 159, "c_dim": 318, "enc_layers": [159, 256, 256,256, 256, 256, 512, 512], "dec_layers": [512, 512, 256, 256, 256, 256, 256, 159]}'\
    --json_quilt_path '/home/ritvik.vasan/test/'\
    --config_path '/home/ritvik.vasan/config.json'\
    --binary_real_one_hot_parameters '{"binary_range": [0, 1],
            "binary_loss": "BCE",
            "real_range": [1, 103],
            "real loss": "MSE",
            "one_hot_range": [103, 159],
            "one_hot_loss": "CE"}'\
    --batch_size 64  \
    --num_batches 1000  \
    --beta_vae 1 \
    --n_epochs 30 \
    --data_type 'aics_features' \
    --dataloader CVAE_testbed.datasets.quilt_aics_features.QuiltAicsFeatures \
    --loss_fn CVAE_testbed.losses.ELBO.combined_loss \
    --lr 0.001
