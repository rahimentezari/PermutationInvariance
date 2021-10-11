#!/bin/bash


### current
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
#caliban run --experiment_config config.json train.py

#caliban cloud --name train_mlp_L1_W8_1024_S1_5 --xgroup train_mlp_L1_W8_1024_S1_5 --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name train_mlp_mnist_L1_W8_1024_S1_5 --xgroup train_mlp_mnist_L1_W8_1024_S1_5 --experiment_config config.json --gpu_spec 1xP100 train.py
#caliban cloud --name train_mlp_L1_W2048_32768_S1_5 --xgroup train_mlp_L1_W2048_32768_S1_5 --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name train_mlp_mnist_L1_W2048_32768_S1_5 --xgroup train_mlp_mnist_L1_W2048_32768_S1_5 --experiment_config config.json --gpu_spec 1xP100 train.py

#caliban cloud --name train_mlp_mnist_high --xgroup train_mlp_mnist_high --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name train_mlp_mnist_low1 --xgroup train_mlp_mnist_low1 --experiment_config config.json --gpu_spec 1xP100 train.py
#caliban cloud --name train_mlp_high --xgroup train_mlp_high --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name train_mlp_mnist_low2 --xgroup train_mlp_mnist_low2 --experiment_config config.json --gpu_spec 1xP100 train.py
#caliban cloud --name train_mlp_w4 --xgroup train_mlp_w4 --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban run --experiment_config config.json train.py
#caliban cloud --name train_mlp_4K --xgroup train_mlp_4K --experiment_config config.json --gpu_spec 1xV100 train.py

#### width s_conv
#caliban run --experiment_config config.json train.py
#caliban cloud --name Sconv_width --xgroup Sconv_width --experiment_config config.json --gpu_spec 1xV100 train.py


##### width & Depth ResNet
#caliban run --experiment_config config.json train.py
#caliban cloud --name ResNet_widthDepth --xgroup ResNet_widthDepth --experiment_config config.json --gpu_spec 1xV100 train.py

##### width & Depth VGG
#caliban run --experiment_config config.json train.py
#caliban cloud --name VGG_width_depth --xgroup VGG_width_depth --experiment_config config.json --gpu_spec 1xV100 train.py


##### Depth
#caliban cloud --name depth_mlp_2layer --xgroup depth_mlp_2layer --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name depth_mlp_S1_6 --xgroup depth_mlp_S1_6 --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name depth_mlp --xgroup depth_mlp --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --name depth_sconv --xgroup depth_sconv --experiment_config config.json --gpu_spec 1xV100 train.py





##### Barrier
#caliban run --experiment_config config_barrier.json barrier.py
#caliban cloud --name after_mlp_width_barrier --xgroup after_mlp_width_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py
#caliban cloud --name after_sconv_width_barrier --xgroup after_sconv_width_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py
#caliban cloud --name after_sconv_width_barrier_512_1024 --xgroup after_sconv_width_barrier_512_1024 --experiment_config config_barrier.json --gpu_spec 1xP100 barrier.py
#caliban cloud --name after_mlp_depth_barrier --xgroup after_mlp_depth_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py
#caliban cloud --name after_sconv_depth_barrier --xgroup after_sconv_depth_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py

#### barrier: resent and vgg
#CUDA_VISIBLE_DEVICES=0 python barrier.py
#caliban run --experiment_config config_barrier.json barrier.py
#caliban cloud --name before_vgg_barrier --xgroup before_vgg_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py
#caliban cloud --name before_resnet_barrier --xgroup before_resnet_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py


#### Basin Barrier
#caliban run --experiment_config config_barrier.json barrier_instance_v2.py
#caliban cloud --name basin_before_barrier --xgroup basin_before_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier_instance_v2.py
## after
#caliban cloud --name basin_mlp_width_barrier --xgroup basin_mlp_width_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier_instance_v2.py
#caliban cloud --name basin_sconv_width_barrier --xgroup basin_sconv_width_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier_instance_v2.py
#caliban cloud --name basin_mlp_depth_barrier --xgroup basin_mlp_depth_barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier_instance_v2.py












##### FD
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_2layer.py
#caliban run --experiment_config config_barrier.json functional_diff_barrier_2layer.py
#caliban cloud --name mlp_width_barrier_FD --xgroup mlp_width_barrier_FD --experiment_config config_barrier.json --gpu_spec 1xP100 functional_diff_barrier_2layer.py


#### 1layer
#caliban run --experiment_config config_barrier.json functional_diff_barrier.py
#caliban cloud --name mlp_width_barrier_FD --xgroup mlp_width_barrier_FD --experiment_config config_barrier.json --gpu_spec 1xV100 functional_diff_barrier.py



##### Gaussian Noise
#CUDA_VISIBLE_DEVICES=0 python Ourmodel_GaussianPerturb.py
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_ourmodel.py
#caliban run --experiment_config config_barrier_FD.json functional_diff_barrier_ourmodel.py
##caliban cloud --name mlp_width_ourmodel_FD --xgroup mlp_width_ourmodel_FD --experiment_config config_barrier_FD.json --gpu_spec 1xV100 functional_diff_barrier_ourmodel.py

#caliban run --experiment_config config_barrier_FD.json FD_orthogonal_ourmodel.py
#caliban cloud --name mlp_width_ourmodel_FD --xgroup mlp_width_ourmodel_FD --experiment_config config_barrier_FD.json --gpu_spec 1xV100 FD_orthogonal_ourmodel.py



#### weight decay and Functional Difference
#caliban run --experiment_config config.json train.py
#caliban cloud --name train_mlp_WD_cifar10 --xgroup train_mlp_WD_cifar10 --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban run --experiment_config config_barrier.json functional_diff_barrier.py


#### alpha our model and real world
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_alpha_ourmodel_realworld.py


### scaler Noise
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_ScalerNoise.py
#caliban run --experiment_config config_fd_fool.json functional_diff_barrier_ScalerNoise.py
#caliban cloud --name FD_fool_mlp --xgroup FD_fool_mlp --experiment_config config_fd_fool.json --gpu_spec 1xV100 functional_diff_barrier_ScalerNoise.py




#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_GradientSimilarity.py
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_GradientSimilarity_hessian.py




#### deterministic training
#caliban run --experiment_config config_deterministic.json train_deterministic.py
#caliban cloud --name FD_fool_train --xgroup FD_fool_train --experiment_config config_deterministic.json --gpu_spec 1xV100 train_deterministic.py

#caliban run --experiment_config config_fd_fool_Gaussian_init.json functional_diff_barrier_Gaussian_init.py
#caliban cloud --name FD_fool_barrier --xgroup FD_fool_barrier --experiment_config config_fd_fool_Gaussian_init.json --gpu_spec 1xV100 functional_diff_barrier_Gaussian_init.py

### shared path
#caliban run --experiment_config config_sharedpath.json train_sharedpath.py
#caliban cloud --name FD_fool_train --xgroup FD_fool_train --experiment_config config_sharedpath.json --gpu_spec 1xV100 train_sharedpath.py


#caliban run --experiment_config config_fd_fool_shared_path.json functional_diff_barrier_SharedPath.py
#caliban cloud --name FD_fool_train --xgroup FD_fool_train --experiment_config config_fd_fool_shared_path.json --gpu_spec 1xV100 functional_diff_barrier_SharedPath.py


#### relaxed version of real world
#caliban run --experiment_config config_deterministic2.json train_deterministic2.py
#caliban cloud --name FD_fool_train --xgroup FD_fool_train --experiment_config config_deterministic2.json --gpu_spec 1xV100 train_deterministic2.py

#caliban run --experiment_config config_fd_fool_Gaussian_init.json functional_diff_barrier_Gaussian_init.py
#caliban cloud --name FD_fool_barrier --xgroup FD_fool_barrier --experiment_config config_fd_fool_Gaussian_init.json --gpu_spec 1xV100 functional_diff_barrier_Gaussian_init.py

### deterministic + scaling
#caliban run --experiment_config config_deterministic_scaled.json train_deterministic_scaled.py
#caliban cloud --name FD_train_scaled --xgroup FD_train_scaled --experiment_config config_deterministic_scaled.json --gpu_spec 1xV100 train_deterministic_scaled.py
#caliban run --experiment_config config_fd_fool_Gaussian_init_scaled.json functional_diff_barrier_Gaussian_init_scaled.py
#caliban cloud --name FD_barrier --xgroup FD_barrier --experiment_config config_fd_fool_Gaussian_init_scaled.json --gpu_spec 1xV100 functional_diff_barrier_Gaussian_init_scaled.py



######## FD scale
#### MLP 1layer
#caliban run --experiment_config config_fd_fool.json functional_diff_barrier_ScalerNoise_mlp1layer.py
#caliban cloud --name FD_fool_mlp1 --xgroup FD_fool_mlp1 --experiment_config config_fd_fool.json --gpu_spec 1xV100 functional_diff_barrier_ScalerNoise_mlp1layer.py
#caliban cloud --name FD_realworld_mlp1 --xgroup FD_realworld_mlp1 --experiment_config config_fd_fool.json --gpu_spec 1xV100 functional_diff_barrier_ScalerNoise_mlp1layer.py


### MLP 2layer
#caliban run --experiment_config config.json train.py
#caliban cloud --name mlp_depth_8k --xgroup mlp_depth_8k --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban run --experiment_config config_fd_fool.json functional_diff_barrier_ScalerNoise.py
#caliban cloud --name FD_fool_mlp --xgroup FD_fool_mlp --experiment_config config_fd_fool.json --gpu_spec 1xV100 functional_diff_barrier_ScalerNoise.py
## 8K
#caliban cloud --name FD_fool_mlp --xgroup FD_fool_mlp --experiment_config config_fd_fool.json --gpu_spec 2xV100 functional_diff_barrier_ScalerNoise.py


### sconv-2layer
#CUDA_VISIBLE_DEVICES=0 python functional_diff_barrier_ScalerNoise_sconv2layer.py
#caliban run --experiment_config config_fd_fool.json functional_diff_barrier_ScalerNoise_sconv2layer.py
#caliban cloud --name FD_sconv2_realworld --xgroup FD_sconv2_realworld --experiment_config config_fd_fool.json --gpu_spec 2xV100 functional_diff_barrier_ScalerNoise_sconv2layer.py

caliban cloud --name FD_sconv2_ourmodel --xgroup FD_sconv2_ourmodel --experiment_config config_fd_fool.json --gpu_spec 2xV100 functional_diff_barrier_ScalerNoise_sconv2layer.py