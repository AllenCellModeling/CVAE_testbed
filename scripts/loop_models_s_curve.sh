CVAE_train \
	--gpu_id 1 \
	--path_save_dir ./outputs/loop_models_s_curve/ \
	--model_fn CVAE_testbed.models.CVAE_baseline.CVAE \
    --model_kwargs '{"x_dim": 3, "c_dim": 6, "enc_layers": [3, 64, 64, 64, 64, 64], "dec_layers": [64, 64, 64, 64, 64, 3], "sklearn_data": "s_curve"}' \
	--post_plot_kwargs '{"latent_space_colorbar": "no"}'\
    --beta_vae 1 \
    --C_vae 0 \
    --batch_size 64  \
    --num_batches 1000  \
    --n_epochs 30 \
    --data_type 'synthetic' \
    --dataloader CVAE_testbed.datasets.circles_moons_blobs.CirclesMoonsBlobs \
    --loss_fn CVAE_testbed.losses.ELBO.synthetic_loss_no_mask \
    --lr 0.001

# CVAE_train \
# 	--gpu_id 1 \
# 	--path_save_dir ./outputs/loop_models_s_curve/ \
# 	--model_fn CVAE_testbed.models.CVAE_dense.CVAE \
#     --model_kwargs '{"sklearn_data": "s_curve", 
#                 "enc_layers": [[9,32,32,32,128],
#                 [128,32,32,32,256],
#                 [256,32,32,32,512]],
#                 "vae_layers": [[512,32,32,32,512],
#                 [512,32,32,32,128]],
#                 "dec_layers": [[134,32,32,32,256],[256,32,32,32,128],[128,32,32,32,3]]}'\
# 	--post_plot_kwargs '{"latent_space_colorbar": "yes"}'\
#     --beta_vae 1 \
#     --C_vae 0 \
#     --batch_size 64  \
#     --num_batches 1000  \
#     --n_epochs 30 \
#     --data_type 'synthetic' \
#     --dataloader CVAE_testbed.datasets.circles_moons_blobs.CirclesMoonsBlobs \
#     --loss_fn CVAE_testbed.losses.ELBO.synthetic_loss_no_mask \
#     --lr 0.001