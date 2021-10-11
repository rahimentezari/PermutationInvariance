import statistics

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import itertools
import copy
from collections import OrderedDict
import numpy as np
import random
import argparse
import pickle
import time
from models import *
from simanneal import Annealer
from google.cloud import storage
start_time = time.time()

# settings
parser = argparse.ArgumentParser(description='LMC- before/after(best)/random permutation')

parser.add_argument('--datadir', default='datasets', type=str,
                    help='path to the directory that contains the datasets (default: datasets)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('--dataset', default='MNIST', type=str,
                    help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
parser.add_argument('--batchsize', default=64, type=int,
                    help='input batch size (default: 64)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=1024, type=int)
parser.add_argument('--steps', default=50000, type=int)
parser.add_argument('--tmax', default=25000, type=float)
parser.add_argument('--tmin', default=2.5, type=float)
parser.add_argument('--pair', default=1, type=int)
import timeit


def main():
    start = timeit.default_timer()  
    global args
    args = parser.parse_args()
    save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}'

    from google.cloud import storage
    # bucket_name = 'permutation-mlp'
    # source_file_name = f'args.pkl'
    # destination_blob_name = f'Neurips21/{save_dir}/barrier/original/{args.seed}/{source_file_name}'
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # pickle_out = pickle.dumps(args)
    # blob.upload_from_string(pickle_out)

    ########### download train/test data/labels to the bucket

    if (args.dataset == 'MNIST'):
        if "mlp" in args.arch:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST_Train_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Train_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
        else:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST3d_Train_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Train_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
    elif (args.dataset == 'CIFAR10'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'SVHN'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'CIFAR100'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    # elif (args.dataset == 'ImageNet'):

    ######## models
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100

    if args.nlayers == 1 and "mlp" in args.arch:
        model = MLP1_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 2 and "mlp" in args.arch:
        model = MLP2_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 4 and "mlp" in args.arch:
        model = MLP4_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 8 and "mlp" in args.arch:
        model = MLP8_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 16 and "mlp" in args.arch:
        model = MLP16_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)

    if args.nlayers == 18:
        model = ResNet18(nclasses, args.width, nchannels)
    if args.nlayers == 34:
        model = ResNet34(nclasses, args.width, nchannels)
    if args.nlayers == 50:
        model = ResNet50(nclasses, args.width, nchannels)

    if "vgg" in args.arch:
        model = vgg.__dict__[args.arch](nclasses)

    if "s_conv" in args.arch and args.nlayers == 2:
        model = s_conv_2layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 4:
        model = s_conv_4layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 6:
        model = s_conv_6layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 8:
        model = s_conv_8layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'

    print(model)

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )


    cudnn.benchmark = True

    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    # # ##################################################### LMC original models
    # # ############################# load selected models
    # # sd = []
    # # for j in [7,8,9,15,16]:
    # #     bucket_name = 'permutation-mlp'
    # #     destination_blob_name = 'model_best.th'
    # #     source_file_name = f'Neurips21/{save_dir}/Train/{j}/{destination_blob_name}'
    # #     download_blob(bucket_name, source_file_name, destination_blob_name)
    # #
    # #     checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # #
    # #     def key_transformation(old_key):
    # #         if 'module' in old_key:
    # #             return old_key[7:]
    # #         return old_key
    # #
    # #     new_state_dict = OrderedDict()
    # #     for key, value in checkpoint.items():
    # #         new_key = key_transformation(key)
    # #         new_state_dict[new_key] = value
    # #     checkpoint = new_state_dict
    # #
    # #     sd.append(checkpoint)
    # #
    # #
    # #
    # #
    # # w = []
    # # for i in range(5):
    # #     params = []
    # #     for key in sd[i].keys():
    # #         param = sd[i][key]
    # #         params.append(param.cpu().detach().numpy())
    # #     w.append(params)
    # #
    # # conv_arch = False
    # # for key in sd[0]:
    # #     print(key, sd[0][key].shape)
    # #     if "conv" in key or "running_mean" in key:
    # #         conv_arch = True
    # #
    # # pairs = list(itertools.combinations(range(5), 2))
    # # pair = 0
    # # barrier_test_basin_before = []
    # # for x in pairs:
    # #     pair = pair + 1
    # #     idx1 = x[0]
    # #     idx2 = x[1]
    # #     sd1_ = sd[idx1]
    # #     sd2_ = sd[idx2]
    # #     dict_after = get_barrier(model, sd1_, sd2_, train_inputs, train_targets, test_inputs, test_targets)
    # #
    # #
    # #     barrier_test = dict_after['barrier_test']
    # #     lmc_test = dict_after['test_lmc']
    # #
    # #     print("barrier_test_pairwise_original", barrier_test)
    # #     print("lmc_test_pairwise_original", lmc_test)
    # #     barrier_test_basin_before.append(barrier_test[0])
    # #
    # #     source_file_name = f'dict_before_{pair}.pkl'
    # #     # destination_blob_name = f'Neurips21/{save_dir}/barrier/SA/auto/{source_file_name}'
    # #     destination_blob_name = f'Neurips21/{save_dir}/barrier/SA_InstanceOptimized_v1/original/{source_file_name}'
    # #     pickle_out = pickle.dumps(dict_after)
    # #     upload_pkl(bucket_name, pickle_out, destination_blob_name)
    # # print()
    # # print("basin_mean_after", statistics.mean(barrier_test_basin_before))
    # # print("basin_std_after", statistics.stdev(barrier_test_basin_before))
    # ########################################## oracle barrier
    # bucket_name = 'permutation-mlp'
    # destination_blob_name = 'model_best.th'
    # source_file_name = f'Neurips21/{save_dir}/Train/{5}/{destination_blob_name}'
    # download_blob(bucket_name, source_file_name, destination_blob_name)
    #
    # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    #
    # def key_transformation(old_key):
    #     if 'module' in old_key:
    #         return old_key[7:]
    #     return old_key
    #
    # new_state_dict = OrderedDict()
    # for key, value in checkpoint.items():
    #     new_key = key_transformation(key)
    #     new_state_dict[new_key] = value
    # checkpoint = new_state_dict
    # sd1 = checkpoint
    #
    # w1 = []
    # for key in sd1.keys():
    #     param = sd1[key]
    #     w1.append(param.cpu().detach().numpy())
    # # create permutation list for mlp
    # if args.arch == 'mlp':
    #     len_perm = []
    #     for i in range(int(len(w1) / 2 - 1)):
    #         len_perm.append(args.width)
    # # create permutation list for conv nets
    # conv_arch = False
    # for key in sd1:
    #     print(key, sd1[key].shape)
    #     if "conv" in key or "running_mean" in key:
    #         conv_arch = True
    #
    # if conv_arch:
    #     params = []
    #     len_perm = []
    #     for key in sd1.keys():
    #         param = model.state_dict()[key]
    #         if "num_batches_tracked" not in key:
    #             params.append(param.cpu().detach().numpy())
    #             if len(param.shape) == 4:
    #                 len_perm.append(param.shape[0])
    #             if len(param.shape) == 2:
    #                 len_perm.append(param.shape[0])
    #
    # print("len_perm", len(len_perm))
    # print("len_perm", len_perm)
    #
    #
    # init_states = []
    # for i in range(1,6):
    #     random_permuted_index = []
    #     for z in len_perm:
    #         lst = [y for y in range(z)]
    #         random.seed(i)
    #         rnd = random.sample(lst, z)
    #         random_permuted_index.append(rnd)
    #     init_states.append(random_permuted_index)
    #
    #
    # # print(sd1["features.0.weight"][0:2])
    # permuted_oracle_sds = []
    # for i in range(5):
    #     permuted_oracle_sds.append(permute(args.arch, model, init_states[i], sd1, w1, nchannels, nclasses, args.width))
    # # print(permuted_oracle_sd1["features.0.weight"][0:2])
    #
    #
    # # # #### sanity check if permutation is done properly: L2 Gaussian Noise
    # # # ##################################################
    # # # # hooks = {}
    # # # # for name, module in model.named_modules():
    # # # #     hooks[name] = module.register_forward_hook(self, hook_fn)
    # # #
    # # #
    # # # activation = {}
    # # # def get_activation(name):
    # # #     def hook(model, input, output):
    # # #         activation[name] = output.detach()
    # # #
    # # #     return hook
    # # #
    # # # device = torch.device('cuda')
    # # # torch.manual_seed(1)
    # # # input_g = torch.randn(256, 1, 32, 32)
    # # # input_g = input_g.to(device)
    # # # # input_g = input_g.to(device).view(input_g.size(0), -1)
    # # # ######################### to model1
    # # # model.load_state_dict(sd1)
    # # # model.register_forward_hook(get_activation('layer4.1.bn2'))
    # # # output = model(input_g)
    # # # print(activation['layer4.1.bn2'].shape)
    # # # print(torch.transpose(sd1['linear.weight'], 0, 1).shape)
    # # # gaussian_out1 = torch.matmul(activation['layer4.1.bn2'], torch.transpose(sd1['linear.weight'], 0, 1))
    # # # ######################### to model2
    # # # model.load_state_dict(permuted_oracle_sd)
    # # # model.register_forward_hook(get_activation(['layer4.1.bn2']))
    # # # output = model(input_g)
    # # # gaussian_out2 = torch.matmul(activation['layer4.1.bn2'], torch.transpose(sd1['linear.weight'], 0, 1))
    # # #
    # # # dist = np.linalg.norm(gaussian_out1.cpu() - gaussian_out2.cpu())
    # # # print(f"L2 noise:", dist)
    # # # print('{0:4f}'.format(dist))
    #
    #
    # ##################################################
    # # bucket_name = 'permutation-mlp'
    # # destination_blob_name = 'model_best.th'
    # # source_file_name = f'Neurips21/{save_dir}/Train/{5}/{destination_blob_name}'
    # # download_blob(bucket_name, source_file_name, destination_blob_name)
    # #
    # # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    #
    # #
    # # def key_transformation(old_key):
    # #     if 'module' in old_key:
    # #         return old_key[7:]
    # #     return old_key
    # #
    # # new_state_dict = OrderedDict()
    # # for key, value in checkpoint.items():
    # #     new_key = key_transformation(key)
    # #     new_state_dict[new_key] = value
    # # checkpoint = new_state_dict
    # # sd1 = checkpoint
    # #
    # # for i in range(5):
    # #
    # #     dict_oracle = get_barrier(model, sd1, permuted_oracle_sds[i], train_inputs, train_targets, test_inputs, test_targets)
    # #     barrier_test = dict_oracle['barrier_test']
    # #     lmc_test = dict_oracle['test_lmc']
    # #
    # #     print("barrier_test_oracle", barrier_test)
    # #     print("lmc_test_oracle", lmc_test)
    # #
    # #     source_file_name = f'dict_oracle_{i}.pkl'
    # #     destination_blob_name = f'Neurips21/{save_dir}/barrier/SA_InstanceOptimized_v1/oracle/before/{source_file_name}'
    # #     pickle_out = pickle.dumps(dict_oracle)
    # #     upload_pkl(bucket_name, pickle_out, destination_blob_name)
    #
    #
    # pairs = list(itertools.combinations(range(5), 2))
    # pair = 0
    # barrier_test_basin_before = []
    # for x in pairs:
    #     pair = pair + 1
    #     idx1 = x[0]
    #     idx2 = x[1]
    #     sd1_ = permuted_oracle_sds[idx1]
    #     sd2_ = permuted_oracle_sds[idx2]
    #     dict_after = get_barrier(model, sd1_, sd2_, train_inputs, train_targets, test_inputs, test_targets)
    #
    #
    #     barrier_test = dict_after['barrier_test']
    #     lmc_test = dict_after['test_lmc']
    #
    #     print("barrier_test_pairwise_original", barrier_test)
    #     print("lmc_test_pairwise_original", lmc_test)
    #     barrier_test_basin_before.append(barrier_test[0])
    #
    #     source_file_name = f'dict_before_{pair}.pkl'
    #     # destination_blob_name = f'Neurips21/{save_dir}/barrier/SA/auto/{source_file_name}'
    #     destination_blob_name = f'Neurips21/{save_dir}/barrier/SA_InstanceOptimized_v1/oracle/before/{source_file_name}'
    #     pickle_out = pickle.dumps(dict_after)
    #     upload_pkl(bucket_name, pickle_out, destination_blob_name)
    # print()
    # print("basin_mean_after", statistics.mean(barrier_test_basin_before))
    # print("basin_std_after", statistics.stdev(barrier_test_basin_before))
    #
    # # ########################################## SA oracle: model1 and permuted model1
    # sd = permuted_oracle_sds
    #
    # w = []
    # for i in range(5):
    #     params = []
    #     for key in sd[i].keys():
    #         param = sd[i][key]
    #         params.append(param.cpu().detach().numpy())
    #     w.append(params)
    #
    # conv_arch = False
    # for key in sd[0]:
    #     print(key, sd[0][key].shape)
    #     if "conv" in key or "running_mean" in key:
    #         conv_arch = True
    #
    # # create permutation list for mlp
    # if args.arch == 'mlp':
    #     len_perm = []
    #     for i in range(int(len(w[0]) / 2 - 1)):
    #         len_perm.append(args.width)
    # # create permutation list for conv nets
    # if conv_arch:
    #     params = []
    #     len_perm = []
    #     for key in sd[0].keys():
    #         param = model.state_dict()[key]
    #         if "num_batches_tracked" not in key:
    #             params.append(param.cpu().detach().numpy())
    #             if len(param.shape) == 4:
    #                 len_perm.append(param.shape[0])
    #             if len(param.shape) == 2:
    #                 len_perm.append(param.shape[0])
    #
    # print("len_perm", len(len_perm))
    # print("len_perm", len_perm)
    #
    # init_state = []
    # for i in range(5):
    #     random_permuted_index = []
    #     for z in len_perm:
    #         lst = [y for y in range(z)]
    #         random.seed(1)
    #         rnd = random.sample(lst, z)
    #         random_permuted_index.append(rnd)
    #     init_state.append(random_permuted_index)
    #
    # exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    # winning_permutation = barrier_SA(args.arch, model, sd, w, init_state,
    #                                  args.tmax, args.tmin, args.steps,
    #                                  train_inputs, train_targets,
    #                                  nchannels, nclasses, args.width)
    # print("winning_permutation", winning_permutation)
    # # winning_permutation = [[[51, 395, 645, 240, 20, 262, 903, 50, 930, 293, 828, 319, 221, 970, 385, 985, 558, 179, 932, 947, 881, 740, 599, 874, 69, 369, 644, 322, 630, 407, 338, 693, 818, 598, 951, 441, 516, 413, 423, 542, 235, 831, 520, 901, 454, 316, 32, 784, 573, 40, 566, 201, 495, 45, 418, 259, 796, 194, 125, 822, 230, 657, 891, 483, 1010, 737, 7, 445, 767, 476, 416, 746, 255, 494, 844, 346, 621, 712, 703, 353, 791, 109, 409, 1007, 887, 959, 436, 639, 339, 198, 629, 8, 183, 782, 112, 946, 399, 12, 246, 1015, 983, 157, 53, 695, 26, 873, 232, 626, 641, 329, 330, 949, 102, 588, 606, 306, 685, 935, 617, 707, 332, 954, 682, 867, 940, 3, 656, 518, 883, 451, 260, 341, 167, 777, 73, 870, 735, 117, 545, 256, 199, 774, 442, 308, 701, 504, 24, 317, 347, 200, 435, 669, 692, 875, 85, 851, 858, 937, 44, 238, 34, 571, 666, 840, 854, 552, 754, 66, 667, 705, 397, 521, 1023, 1019, 188, 360, 897, 866, 957, 261, 439, 124, 661, 860, 908, 716, 182, 401, 525, 815, 801, 379, 447, 913, 460, 812, 914, 964, 575, 433, 215, 832, 83, 169, 489, 233, 530, 206, 759, 819, 321, 146, 381, 22, 468, 415, 60, 211, 855, 431, 537, 837, 921, 508, 234, 943, 218, 772, 506, 294, 562, 714, 1005, 594, 287, 608, 209, 612, 490, 471, 684, 27, 591, 747, 723, 625, 33, 826, 514, 761, 808, 786, 220, 29, 595, 61, 511, 266, 17, 806, 788, 104, 702, 243, 658, 274, 71, 539, 202, 655, 455, 64, 2, 771, 38, 523, 288, 497, 296, 650, 565, 652, 512, 532, 428, 659, 237, 729, 620, 673, 99, 636, 836, 586, 841, 178, 781, 165, 223, 54, 372, 531, 719, 265, 1021, 438, 663, 361, 1017, 88, 480, 474, 242, 925, 880, 488, 94, 884, 615, 279, 59, 528, 743, 817, 197, 6, 865, 103, 677, 250, 604, 992, 960, 839, 440, 152, 607, 756, 596, 36, 895, 541, 139, 670, 335, 805, 425, 299, 533, 337, 35, 568, 929, 792, 613, 450, 343, 509, 579, 89, 877, 963, 654, 522, 196, 14, 790, 285, 16, 680, 147, 479, 477, 462, 1, 734, 364, 192, 270, 484, 349, 574, 141, 660, 1018, 121, 843, 611, 1003, 333, 988, 268, 730, 633, 529, 151, 557, 668, 135, 696, 159, 168, 802, 592, 540, 271, 210, 105, 107, 721, 779, 72, 823, 382, 212, 195, 1000, 166, 384, 609, 43, 5, 155, 632, 678, 219, 519, 700, 400, 833, 581, 408, 548, 309, 70, 295, 189, 628, 113, 706, 948, 745, 144, 340, 63, 538, 448, 576, 998, 554, 527, 153, 838, 138, 857, 378, 93, 1004, 76, 907, 457, 357, 391, 98, 48, 1020, 749, 502, 825, 67, 464, 190, 927, 18, 924, 748, 971, 926, 203, 691, 236, 764, 421, 217, 0, 492, 972, 498, 597, 829, 708, 185, 944, 253, 482, 264, 859, 846, 688, 465, 635, 869, 30, 154, 942, 934, 148, 175, 473, 931, 778, 958, 114, 390, 286, 56, 922, 775, 58, 976, 249, 798, 731, 129, 605, 885, 412, 150, 894, 807, 344, 371, 191, 358, 458, 348, 827, 101, 324, 564, 15, 47, 753, 28, 741, 547, 787, 267, 466, 570, 799, 651, 380, 793, 478, 81, 886, 797, 704, 752, 757, 444, 681, 879, 783, 359, 304, 718, 856, 278, 578, 986, 585, 814, 847, 374, 810, 314, 181, 248, 389, 106, 91, 709, 672, 370, 459, 486, 387, 383, 991, 898, 149, 733, 583, 933, 864, 910, 193, 580, 132, 313, 254, 86, 214, 281, 213, 816, 247, 785, 342, 560, 4, 368, 821, 419, 300, 973, 551, 119, 1012, 429, 809, 283, 780, 919, 563, 732, 982, 161, 862, 133, 375, 999, 952, 405, 751, 559, 95, 619, 417, 961, 363, 830, 848, 163, 115, 968, 590, 569, 403, 122, 676, 298, 1011, 795, 257, 978, 131, 75, 646, 410, 683, 549, 472, 679, 292, 239, 720, 535, 493, 388, 824, 713, 57, 507, 912, 406, 301, 37, 272, 515, 128, 1002, 981, 835, 87, 710, 350, 510, 171, 980, 222, 593, 962, 853, 13, 939, 290, 915, 553, 602, 534, 11, 763, 715, 367, 136, 544, 916, 323, 543, 770, 328, 975, 648, 41, 446, 717, 145, 365, 727, 177, 918, 126, 251, 503, 393, 19, 334, 77, 505, 307, 616, 373, 674, 411, 849, 750, 331, 736, 755, 143, 561, 414, 434, 485, 62, 352, 603, 427, 291, 555, 724, 318, 158, 231, 229, 722, 687, 1001, 500, 773, 868, 690, 536, 872, 280, 989, 513, 969, 23, 90, 108, 803, 116, 899, 904, 310, 426, 996, 945, 863, 643, 276, 769, 228, 966, 640, 351, 226, 834, 974, 726, 184, 386, 634, 888, 463, 377, 302, 461, 21, 275, 876, 871, 258, 437, 241, 766, 882, 452, 845, 917, 244, 889, 46, 517, 1014, 993, 10, 671, 376, 601, 556, 404, 789, 902, 355, 394, 738, 526, 584, 953, 739, 420, 186, 744, 699, 1016, 142, 995, 811, 123, 647, 92, 79, 176, 698, 245, 906, 273, 392, 137, 277, 686, 140, 467, 728, 965, 938, 325, 65, 174, 110, 39, 315, 187, 923, 173, 424, 892, 850, 499, 941, 804, 74, 697, 282, 97, 52, 312, 180, 469, 890, 955, 852, 127, 765, 997, 896, 491, 760, 984, 9, 160, 1008, 637, 311, 204, 987, 622, 1013, 776, 572, 990, 893, 134, 567, 84, 900, 216, 68, 928, 82, 524, 156, 164, 224, 345, 78, 1022, 1006, 861, 55, 336, 909, 694, 305, 289, 263, 269, 25, 430, 920, 130, 354, 813, 624, 653, 356, 638, 979, 162, 642, 366, 208, 205, 327, 800, 1009, 689, 100, 496, 402, 842, 589, 80, 967, 118, 587, 432, 303, 950, 422, 172, 956, 627, 662, 631, 470, 170, 610, 225, 742, 675, 120, 31, 207, 994, 878, 600, 911, 546, 456, 481, 582, 725, 905, 711, 618, 501, 768, 49, 577, 820, 362, 665, 252, 977, 42, 936, 443, 649, 453, 758, 96, 623, 297, 794, 398, 475, 487, 111, 762, 227, 614, 550, 320, 284, 326, 449, 396, 664]], [[980, 939, 405, 139, 387, 13, 656, 546, 969, 931, 1004, 572, 904, 803, 989, 291, 601, 358, 505, 39, 424, 569, 378, 681, 440, 214, 622, 544, 471, 410, 217, 1022, 170, 742, 239, 985, 604, 368, 881, 485, 495, 85, 710, 422, 392, 445, 771, 376, 776, 144, 94, 576, 159, 416, 452, 894, 256, 281, 531, 327, 704, 350, 1016, 521, 693, 759, 299, 840, 185, 964, 791, 748, 114, 799, 918, 486, 160, 595, 517, 914, 425, 72, 884, 646, 589, 675, 986, 443, 533, 17, 502, 10, 464, 639, 773, 732, 833, 631, 588, 484, 86, 625, 57, 655, 462, 490, 211, 179, 215, 279, 56, 207, 265, 889, 975, 666, 176, 846, 71, 231, 873, 75, 95, 926, 988, 556, 698, 902, 643, 102, 101, 198, 828, 537, 623, 747, 872, 278, 103, 635, 874, 634, 616, 233, 686, 450, 116, 374, 89, 680, 527, 762, 491, 178, 540, 47, 1008, 936, 813, 208, 869, 458, 268, 570, 627, 557, 232, 100, 167, 285, 615, 621, 692, 560, 415, 370, 182, 507, 212, 301, 994, 1021, 390, 609, 343, 37, 730, 188, 674, 30, 696, 957, 757, 393, 446, 876, 853, 997, 162, 553, 867, 125, 930, 587, 536, 503, 726, 990, 1018, 722, 220, 709, 583, 414, 240, 61, 397, 115, 890, 109, 355, 247, 995, 868, 218, 255, 628, 302, 135, 558, 1019, 474, 190, 142, 903, 760, 832, 578, 259, 133, 437, 359, 763, 52, 143, 166, 412, 792, 795, 434, 946, 35, 573, 448, 455, 307, 999, 164, 465, 391, 744, 449, 644, 78, 399, 127, 701, 384, 234, 568, 400, 123, 44, 697, 54, 21, 934, 949, 626, 282, 435, 848, 377, 168, 346, 864, 338, 15, 352, 1007, 955, 672, 204, 756, 128, 460, 0, 993, 827, 403, 1001, 787, 727, 594, 5, 866, 51, 2, 252, 577, 981, 529, 64, 304, 630, 702, 597, 266, 784, 482, 1014, 967, 843, 790, 793, 555, 151, 428, 834, 493, 419, 538, 690, 401, 110, 953, 830, 470, 751, 688, 893, 295, 68, 952, 147, 652, 298, 328, 901, 602, 920, 411, 691, 202, 599, 339, 733, 274, 541, 472, 825, 624, 774, 617, 717, 919, 60, 24, 734, 859, 488, 535, 251, 366, 611, 183, 897, 332, 475, 516, 192, 305, 775, 966, 227, 267, 900, 564, 642, 1006, 303, 45, 345, 439, 571, 510, 224, 842, 845, 736, 837, 84, 907, 738, 306, 469, 1013, 479, 447, 620, 381, 324, 525, 645, 348, 360, 812, 849, 1010, 708, 290, 459, 550, 132, 716, 665, 637, 46, 600, 145, 528, 970, 506, 909, 641, 65, 805, 288, 720, 119, 362, 943, 801, 230, 911, 219, 32, 735, 134, 161, 396, 421, 679, 96, 586, 48, 406, 888, 983, 678, 108, 386, 79, 554, 58, 137, 777, 4, 140, 664, 148, 764, 152, 855, 783, 565, 1011, 877, 765, 26, 365, 413, 924, 606, 242, 711, 334, 118, 662, 581, 798, 719, 660, 542, 962, 561, 714, 728, 998, 636, 191, 921, 895, 640, 653, 216, 823, 806, 963, 603, 6, 563, 700, 420, 605, 451, 11, 847, 29, 682, 309, 916, 923, 811, 273, 984, 928, 892, 945, 494, 19, 590, 973, 66, 487, 330, 694, 383, 913, 77, 1, 607, 325, 592, 670, 31, 427, 835, 25, 254, 804, 466, 941, 354, 297, 530, 950, 695, 364, 703, 677, 389, 228, 959, 80, 156, 858, 908, 8, 982, 453, 978, 925, 683, 300, 654, 785, 117, 226, 467, 685, 322, 547, 996, 311, 136, 687, 337, 826, 886, 236, 316, 16, 289, 367, 347, 739, 69, 314, 809, 223, 852, 380, 817, 612, 1012, 28, 87, 741, 431, 885, 146, 122, 53, 778, 107, 264, 948, 574, 82, 875, 824, 810, 344, 292, 870, 712, 3, 9, 50, 272, 906, 715, 22, 74, 130, 619, 815, 937, 927, 173, 598, 34, 201, 1020, 442, 184, 831, 746, 880, 238, 197, 707, 947, 596, 1015, 320, 408, 49, 126, 280, 149, 915, 740, 124, 150, 478, 480, 839, 187, 36, 138, 584, 90, 407, 585, 896, 754, 349, 731, 539, 81, 283, 820, 59, 938, 385, 457, 423, 887, 650, 851, 871, 372, 141, 649, 992, 961, 62, 20, 270, 942, 509, 856, 780, 863, 258, 829, 514, 155, 879, 432, 321, 375, 743, 991, 318, 477, 157, 657, 222, 671, 814, 335, 899, 860, 426, 684, 725, 857, 199, 891, 165, 534, 468, 394, 543, 245, 73, 99, 905, 312, 629, 614, 174, 1017, 120, 632, 882, 854, 489, 647, 663, 786, 749, 418, 838, 315, 782, 689, 373, 794, 504, 713, 271, 808, 409, 97, 175, 515, 40, 968, 850, 960, 113, 193, 971, 356, 382, 163, 878, 706, 402, 456, 250, 974, 429, 800, 308, 758, 522, 524, 91, 284, 483, 575, 865, 169, 7, 241, 844, 562, 441, 417, 932, 331, 767, 944, 210, 189, 822, 633, 910, 753, 591, 551, 933, 755, 745, 253, 194, 249, 501, 816, 438, 463, 1003, 196, 548, 976, 235, 638, 929, 667, 243, 508, 724, 195, 203, 12, 293, 41, 737, 545, 42, 461, 129, 898, 593, 526, 351, 433, 1005, 956, 353, 336, 287, 33, 559, 92, 319, 121, 23, 669, 310, 171, 567, 518, 935, 549, 1009, 269, 750, 705, 779, 772, 979, 263, 532, 807, 388, 781, 912, 209, 200, 497, 661, 14, 221, 673, 275, 492, 342, 513, 104, 206, 294, 260, 262, 444, 404, 987, 496, 1023, 246, 954, 379, 579, 172, 83, 43, 821, 98, 277, 1002, 131, 552, 473, 333, 651, 67, 181, 476, 699, 158, 818, 658, 659, 676, 770, 500, 70, 769, 106, 244, 836, 329, 862, 105, 958, 313, 55, 63, 618, 861, 111, 237, 186, 768, 76, 205, 511, 608, 580, 180, 398, 213, 582, 357, 248, 340, 965, 668, 940, 752, 112, 819, 481, 154, 789, 613, 972, 883, 27, 93, 761, 38, 723, 802, 788, 296, 520, 369, 917, 326, 286, 796, 1000, 566, 363, 257, 276, 88, 371, 512, 718, 341, 610, 841, 153, 436, 766, 317, 18, 498, 430, 225, 922, 229, 523, 361, 797, 395, 454, 499, 323, 177, 519, 261, 729, 721, 951, 977, 648]], [[390, 793, 403, 728, 249, 97, 166, 731, 885, 231, 824, 844, 431, 264, 636, 108, 944, 118, 302, 415, 439, 420, 472, 372, 660, 268, 119, 120, 236, 722, 423, 335, 474, 106, 304, 452, 228, 859, 911, 307, 239, 826, 405, 412, 274, 1, 476, 881, 49, 748, 95, 186, 371, 345, 467, 740, 972, 833, 47, 918, 247, 743, 523, 508, 199, 901, 682, 481, 5, 89, 39, 280, 865, 569, 909, 778, 733, 642, 982, 711, 96, 384, 68, 355, 839, 628, 543, 102, 66, 990, 600, 293, 235, 514, 517, 558, 927, 134, 300, 51, 609, 190, 699, 836, 819, 858, 459, 203, 779, 210, 943, 310, 20, 201, 375, 594, 167, 554, 615, 828, 388, 316, 936, 694, 542, 922, 515, 422, 805, 690, 830, 299, 795, 330, 112, 442, 410, 449, 673, 953, 255, 613, 620, 114, 585, 997, 545, 874, 563, 212, 57, 1003, 132, 309, 875, 811, 708, 81, 348, 269, 534, 948, 646, 774, 105, 211, 866, 772, 786, 519, 757, 392, 1004, 182, 904, 533, 123, 910, 305, 829, 245, 331, 752, 800, 194, 347, 336, 436, 389, 857, 84, 684, 202, 612, 812, 926, 451, 700, 570, 441, 817, 288, 606, 539, 668, 477, 53, 457, 883, 967, 634, 432, 38, 894, 614, 222, 838, 907, 221, 692, 504, 993, 1006, 631, 712, 709, 616, 847, 447, 282, 1001, 787, 59, 25, 704, 75, 890, 238, 942, 99, 892, 667, 93, 400, 574, 209, 681, 229, 785, 861, 992, 263, 69, 965, 58, 395, 327, 306, 464, 618, 737, 326, 296, 659, 559, 82, 710, 233, 275, 790, 536, 912, 225, 855, 359, 421, 110, 393, 557, 44, 185, 385, 41, 363, 969, 661, 555, 178, 957, 562, 572, 617, 386, 56, 260, 402, 573, 643, 64, 906, 654, 344, 763, 87, 732, 924, 611, 958, 237, 29, 697, 144, 133, 598, 889, 168, 849, 321, 960, 940, 902, 241, 357, 527, 475, 509, 913, 242, 142, 398, 374, 683, 150, 92, 962, 356, 1011, 131, 320, 537, 490, 303, 625, 315, 917, 24, 214, 821, 794, 657, 744, 586, 207, 273, 42, 556, 512, 487, 160, 159, 792, 473, 314, 184, 141, 977, 32, 215, 549, 339, 635, 996, 548, 34, 653, 898, 810, 213, 254, 318, 1009, 285, 153, 218, 125, 771, 622, 766, 419, 765, 2, 832, 427, 414, 575, 28, 1015, 411, 698, 460, 725, 963, 54, 343, 671, 227, 468, 290, 896, 169, 715, 62, 366, 648, 177, 846, 561, 971, 729, 373, 399, 511, 7, 12, 340, 588, 921, 287, 364, 191, 518, 755, 919, 739, 747, 6, 250, 999, 520, 638, 197, 171, 566, 1021, 633, 652, 522, 505, 882, 454, 329, 308, 195, 970, 981, 689, 67, 540, 353, 8, 802, 22, 91, 589, 568, 564, 576, 870, 773, 930, 979, 183, 935, 1002, 706, 493, 456, 444, 193, 808, 813, 376, 361, 196, 531, 666, 458, 835, 426, 730, 478, 109, 605, 713, 55, 860, 324, 40, 841, 101, 696, 179, 989, 367, 877, 968, 807, 281, 136, 937, 15, 623, 1007, 734, 50, 78, 544, 1014, 769, 226, 445, 219, 676, 76, 438, 675, 798, 198, 341, 597, 650, 461, 900, 867, 584, 272, 656, 352, 916, 262, 135, 735, 818, 1012, 521, 754, 702, 297, 107, 80, 591, 418, 200, 429, 284, 261, 868, 446, 9, 862, 507, 899, 665, 724, 762, 884, 753, 974, 759, 985, 407, 923, 46, 719, 328, 644, 380, 816, 152, 294, 770, 143, 397, 248, 358, 629, 35, 571, 995, 434, 678, 391, 526, 122, 908, 424, 579, 365, 721, 781, 1019, 265, 647, 286, 346, 496, 127, 234, 905, 289, 181, 891, 488, 158, 342, 939, 387, 501, 73, 592, 138, 925, 601, 599, 751, 19, 4, 632, 437, 929, 466, 240, 27, 945, 510, 482, 224, 13, 608, 362, 497, 975, 52, 253, 961, 479, 1016, 791, 578, 720, 827, 854, 506, 664, 959, 295, 976, 895, 334, 834, 780, 736, 121, 687, 220, 1000, 88, 10, 377, 856, 680, 3, 1022, 187, 815, 145, 471, 994, 33, 853, 140, 271, 837, 277, 933, 172, 139, 500, 931, 485, 560, 991, 525, 495, 369, 820, 124, 499, 746, 216, 117, 94, 577, 26, 360, 872, 36, 147, 349, 175, 333, 797, 541, 775, 312, 243, 701, 470, 298, 947, 383, 283, 126, 546, 354, 823, 869, 580, 455, 663, 768, 161, 463, 938, 998, 16, 368, 65, 888, 626, 973, 63, 538, 750, 850, 955, 707, 43, 621, 871, 950, 651, 157, 417, 567, 789, 325, 602, 1008, 777, 717, 723, 583, 920, 483, 151, 404, 529, 311, 450, 323, 165, 880, 98, 223, 987, 587, 61, 208, 291, 146, 433, 760, 413, 801, 809, 978, 814, 103, 498, 130, 149, 822, 581, 741, 886, 934, 952, 188, 337, 658, 831, 610, 552, 492, 1017, 173, 465, 79, 322, 903, 338, 604, 156, 928, 863, 776, 491, 137, 530, 484, 71, 313, 408, 206, 686, 714, 627, 319, 409, 914, 17, 252, 645, 983, 873, 840, 964, 641, 693, 986, 749, 18, 649, 0, 100, 489, 738, 469, 516, 217, 443, 85, 379, 553, 788, 674, 351, 590, 1018, 758, 164, 70, 624, 915, 767, 803, 954, 30, 170, 21, 984, 876, 246, 406, 266, 204, 842, 524, 897, 716, 503, 799, 806, 256, 378, 486, 742, 852, 550, 1023, 825, 128, 640, 86, 784, 113, 162, 528, 230, 691, 332, 703, 174, 278, 416, 111, 259, 401, 1020, 428, 551, 90, 887, 83, 232, 317, 292, 301, 718, 804, 637, 745, 60, 845, 453, 502, 941, 205, 595, 596, 630, 104, 655, 1005, 761, 932, 396, 670, 980, 851, 662, 72, 381, 448, 430, 155, 879, 672, 279, 685, 966, 603, 192, 705, 695, 669, 726, 74, 988, 258, 532, 565, 727, 756, 189, 440, 864, 494, 1013, 48, 129, 462, 267, 593, 31, 677, 679, 270, 639, 949, 23, 848, 251, 382, 547, 77, 878, 176, 37, 257, 956, 783, 14, 946, 764, 607, 45, 115, 148, 425, 951, 843, 796, 893, 582, 11, 394, 163, 619, 535, 350, 480, 688, 244, 116, 154, 1010, 370, 276, 513, 782, 435, 180]], [[26, 724, 113, 59, 781, 800, 330, 481, 337, 156, 903, 548, 185, 15, 169, 811, 944, 801, 265, 342, 541, 476, 82, 945, 351, 530, 715, 618, 908, 766, 1005, 693, 226, 440, 950, 842, 355, 244, 733, 236, 326, 88, 596, 581, 955, 551, 976, 1001, 181, 769, 519, 978, 374, 832, 139, 384, 499, 297, 668, 748, 859, 0, 924, 610, 157, 221, 333, 513, 311, 406, 898, 716, 106, 972, 694, 841, 142, 104, 721, 905, 515, 241, 171, 222, 971, 977, 778, 302, 39, 64, 315, 216, 349, 592, 319, 646, 729, 130, 631, 619, 539, 836, 47, 195, 828, 449, 589, 498, 144, 329, 435, 912, 632, 56, 900, 962, 571, 402, 314, 198, 203, 645, 93, 809, 450, 746, 609, 506, 823, 942, 358, 636, 212, 964, 784, 273, 558, 886, 10, 366, 580, 909, 740, 745, 981, 475, 65, 783, 793, 338, 701, 926, 361, 545, 204, 767, 853, 206, 376, 249, 92, 529, 500, 614, 600, 759, 207, 790, 754, 109, 574, 5, 474, 2, 1023, 122, 552, 178, 220, 802, 100, 248, 225, 284, 732, 94, 32, 258, 535, 718, 145, 191, 641, 791, 815, 459, 576, 495, 62, 152, 876, 889, 490, 838, 966, 37, 709, 132, 155, 354, 362, 960, 963, 428, 734, 385, 418, 628, 470, 391, 469, 403, 835, 904, 310, 726, 856, 461, 388, 922, 462, 58, 107, 404, 158, 274, 511, 345, 650, 952, 153, 129, 1013, 409, 306, 543, 68, 991, 436, 304, 341, 1016, 591, 451, 379, 821, 60, 264, 1022, 7, 180, 518, 141, 728, 990, 260, 487, 797, 368, 408, 812, 834, 369, 538, 621, 439, 700, 910, 118, 690, 27, 312, 667, 570, 165, 353, 45, 240, 63, 544, 363, 420, 172, 270, 738, 151, 419, 707, 1021, 999, 664, 872, 303, 131, 209, 925, 526, 965, 635, 698, 688, 611, 323, 121, 946, 364, 18, 516, 585, 663, 595, 251, 954, 762, 211, 651, 647, 54, 702, 234, 29, 719, 414, 70, 705, 837, 242, 160, 865, 980, 864, 218, 437, 894, 888, 115, 327, 587, 597, 522, 252, 324, 830, 494, 782, 931, 804, 16, 170, 934, 657, 785, 593, 472, 975, 40, 370, 350, 331, 684, 339, 786, 120, 479, 758, 200, 416, 820, 30, 661, 380, 866, 458, 411, 215, 666, 789, 493, 869, 69, 98, 969, 348, 445, 308, 572, 798, 524, 279, 717, 706, 41, 656, 298, 224, 508, 689, 723, 154, 480, 770, 184, 996, 116, 826, 263, 491, 839, 51, 675, 557, 849, 932, 554, 941, 431, 792, 523, 902, 1019, 831, 239, 1015, 559, 286, 612, 637, 357, 33, 556, 893, 432, 517, 764, 760, 858, 101, 757, 874, 28, 456, 86, 584, 114, 6, 372, 85, 3, 484, 703, 174, 318, 390, 305, 985, 679, 569, 603, 860, 1011, 321, 287, 984, 22, 731, 397, 510, 756, 607, 486, 205, 822, 714, 189, 884, 601, 168, 928, 627, 805, 489, 777, 638, 863, 906, 340, 393, 528, 21, 410, 652, 534, 833, 140, 295, 795, 13, 19, 89, 560, 816, 301, 998, 871, 78, 901, 53, 504, 44, 79, 895, 590, 425, 606, 238, 973, 761, 377, 356, 442, 1008, 166, 779, 396, 658, 594, 136, 434, 915, 634, 625, 527, 177, 583, 501, 672, 123, 399, 995, 568, 624, 825, 455, 313, 533, 8, 775, 71, 111, 881, 936, 670, 162, 99, 84, 188, 712, 20, 230, 74, 887, 855, 448, 259, 655, 344, 725, 1018, 577, 457, 293, 182, 1017, 660, 452, 752, 654, 80, 49, 183, 228, 807, 542, 454, 275, 720, 806, 328, 780, 332, 682, 83, 187, 696, 873, 993, 360, 713, 322, 389, 773, 197, 742, 854, 917, 861, 883, 23, 940, 352, 829, 477, 687, 173, 639, 892, 192, 164, 824, 937, 429, 817, 852, 325, 992, 255, 956, 623, 31, 848, 412, 982, 277, 175, 744, 678, 622, 103, 43, 862, 75, 847, 433, 643, 57, 566, 880, 143, 642, 743, 468, 617, 930, 916, 844, 711, 613, 540, 375, 17, 885, 927, 347, 492, 920, 665, 686, 427, 737, 231, 626, 334, 648, 76, 73, 763, 299, 383, 677, 465, 644, 290, 814, 482, 289, 125, 346, 896, 443, 921, 988, 555, 196, 890, 359, 466, 309, 818, 407, 599, 851, 579, 509, 243, 681, 741, 514, 730, 159, 400, 72, 671, 768, 288, 774, 421, 478, 278, 674, 483, 202, 81, 135, 371, 336, 1000, 813, 604, 405, 1004, 257, 294, 138, 335, 463, 250, 772, 34, 563, 247, 683, 739, 12, 983, 582, 588, 727, 105, 755, 989, 35, 210, 394, 708, 163, 567, 870, 444, 913, 268, 67, 1006, 381, 52, 398, 296, 564, 119, 87, 42, 899, 953, 747, 422, 1003, 387, 24, 1014, 134, 256, 598, 367, 699, 280, 961, 553, 415, 776, 261, 970, 615, 464, 620, 77, 233, 808, 1010, 520, 217, 662, 254, 229, 446, 395, 507, 605, 685, 1012, 602, 438, 918, 967, 4, 974, 272, 117, 575, 237, 750, 497, 149, 1007, 193, 697, 959, 235, 190, 108, 630, 938, 423, 179, 9, 505, 283, 316, 788, 819, 11, 968, 676, 521, 751, 282, 46, 878, 146, 285, 223, 271, 110, 401, 680, 810, 246, 378, 659, 496, 911, 547, 291, 549, 453, 14, 633, 66, 629, 133, 669, 919, 867, 300, 55, 97, 923, 343, 199, 148, 485, 25, 128, 102, 649, 979, 96, 137, 365, 787, 857, 735, 386, 951, 441, 502, 765, 424, 38, 1020, 276, 573, 771, 546, 653, 36, 933, 426, 673, 616, 692, 799, 997, 947, 736, 691, 150, 50, 929, 753, 219, 413, 827, 473, 503, 987, 948, 430, 891, 882, 640, 373, 986, 562, 550, 176, 471, 267, 943, 868, 565, 794, 194, 417, 90, 253, 608, 531, 939, 112, 227, 232, 127, 48, 147, 722, 935, 292, 840, 208, 749, 382, 525, 213, 245, 846, 1009, 875, 91, 95, 586, 214, 317, 126, 447, 803, 949, 958, 796, 843, 704, 897, 266, 695, 994, 578, 957, 307, 161, 167, 61, 488, 879, 1, 561, 281, 512, 124, 907, 269, 186, 532, 262, 467, 320, 877, 537, 536, 392, 1002, 845, 201, 710, 460, 850, 914]], [[236, 1011, 783, 954, 174, 782, 31, 546, 62, 420, 502, 894, 429, 19, 1008, 298, 258, 941, 134, 745, 645, 104, 762, 135, 455, 967, 51, 392, 814, 562, 844, 79, 361, 192, 710, 379, 321, 81, 459, 1017, 257, 673, 24, 556, 853, 88, 513, 156, 472, 927, 125, 848, 650, 295, 272, 583, 701, 377, 1000, 551, 824, 577, 938, 111, 798, 383, 721, 501, 864, 860, 991, 945, 580, 149, 931, 735, 140, 356, 536, 12, 446, 394, 829, 554, 589, 797, 627, 360, 100, 350, 245, 473, 341, 832, 740, 789, 243, 517, 411, 887, 120, 496, 607, 143, 391, 989, 436, 744, 714, 573, 997, 309, 834, 131, 499, 32, 375, 434, 759, 339, 485, 468, 992, 274, 17, 445, 766, 737, 424, 329, 450, 205, 858, 720, 199, 126, 537, 999, 696, 323, 560, 466, 503, 169, 90, 290, 950, 123, 1005, 85, 543, 196, 680, 854, 826, 896, 984, 656, 538, 960, 180, 145, 773, 283, 197, 303, 963, 311, 717, 612, 754, 477, 138, 623, 660, 401, 431, 688, 2, 753, 187, 519, 36, 763, 52, 900, 628, 846, 417, 567, 806, 687, 557, 210, 306, 533, 784, 780, 124, 836, 510, 163, 930, 859, 37, 929, 987, 802, 678, 882, 599, 527, 282, 833, 82, 47, 139, 781, 388, 480, 819, 230, 271, 741, 959, 185, 61, 772, 158, 621, 415, 652, 547, 106, 66, 1006, 389, 672, 427, 665, 63, 87, 921, 640, 711, 403, 928, 275, 231, 222, 38, 831, 820, 530, 46, 917, 93, 515, 342, 457, 803, 769, 899, 449, 279, 869, 315, 872, 54, 493, 972, 250, 419, 217, 1020, 198, 793, 698, 132, 76, 177, 966, 362, 478, 227, 971, 604, 219, 704, 333, 171, 248, 603, 624, 57, 16, 807, 855, 265, 605, 129, 918, 267, 908, 752, 416, 1001, 273, 454, 863, 387, 952, 1010, 585, 619, 314, 758, 456, 840, 212, 511, 559, 973, 133, 439, 828, 926, 867, 889, 563, 638, 743, 55, 876, 68, 637, 378, 697, 299, 590, 726, 1007, 940, 83, 9, 1021, 405, 364, 881, 437, 862, 738, 504, 288, 1016, 332, 731, 957, 294, 661, 0, 203, 354, 975, 467, 788, 286, 706, 521, 982, 885, 371, 460, 905, 73, 904, 920, 160, 852, 463, 426, 235, 676, 494, 491, 347, 4, 634, 596, 642, 865, 278, 816, 677, 756, 287, 368, 693, 200, 475, 684, 715, 8, 810, 497, 335, 300, 253, 130, 579, 936, 568, 667, 598, 95, 978, 685, 157, 72, 369, 581, 27, 548, 380, 727, 218, 346, 509, 29, 25, 337, 742, 365, 119, 305, 812, 561, 939, 804, 141, 709, 647, 152, 42, 167, 137, 526, 850, 699, 270, 128, 320, 113, 483, 564, 366, 847, 648, 664, 175, 874, 432, 372, 702, 367, 244, 800, 998, 943, 109, 523, 263, 440, 609, 349, 479, 632, 792, 914, 92, 324, 18, 651, 304, 184, 495, 572, 327, 50, 7, 827, 406, 121, 703, 412, 679, 292, 276, 962, 404, 1012, 873, 393, 498, 977, 14, 757, 625, 142, 534, 168, 410, 591, 49, 749, 739, 785, 118, 328, 674, 923, 686, 86, 809, 866, 179, 317, 75, 97, 922, 458, 996, 837, 1018, 915, 34, 89, 842, 986, 399, 597, 470, 482, 355, 264, 188, 451, 353, 912, 649, 242, 193, 786, 422, 136, 951, 312, 154, 191, 937, 428, 310, 30, 213, 40, 390, 540, 376, 487, 334, 471, 204, 409, 764, 576, 570, 489, 385, 397, 719, 464, 565, 150, 241, 331, 569, 277, 691, 890, 911, 15, 910, 707, 3, 359, 730, 694, 965, 512, 269, 613, 909, 942, 861, 594, 683, 183, 147, 925, 1002, 488, 444, 115, 995, 813, 338, 96, 525, 59, 532, 976, 849, 508, 5, 114, 675, 103, 461, 45, 644, 505, 592, 944, 44, 421, 490, 101, 228, 110, 818, 326, 313, 190, 615, 165, 402, 949, 182, 566, 301, 617, 166, 835, 1003, 618, 633, 705, 117, 796, 777, 91, 728, 880, 107, 736, 779, 99, 535, 226, 246, 574, 43, 544, 151, 1015, 932, 733, 520, 438, 484, 293, 492, 666, 345, 586, 751, 584, 776, 302, 465, 176, 791, 381, 635, 395, 614, 1022, 297, 336, 11, 374, 98, 41, 857, 712, 308, 969, 821, 441, 486, 201, 112, 400, 280, 181, 524, 550, 974, 616, 1004, 322, 234, 794, 67, 654, 600, 531, 233, 775, 778, 146, 755, 558, 902, 6, 877, 555, 518, 26, 373, 237, 178, 708, 903, 808, 506, 1019, 194, 700, 689, 382, 655, 153, 626, 452, 841, 822, 961, 582, 211, 770, 983, 993, 433, 653, 646, 823, 988, 398, 552, 892, 990, 622, 830, 481, 906, 765, 161, 610, 384, 669, 351, 774, 144, 343, 724, 195, 897, 955, 1014, 799, 408, 588, 500, 761, 39, 239, 207, 35, 65, 71, 713, 357, 229, 811, 425, 851, 620, 259, 94, 795, 220, 215, 247, 985, 878, 216, 443, 251, 801, 682, 21, 462, 948, 787, 80, 78, 318, 639, 919, 260, 541, 888, 396, 261, 924, 363, 805, 593, 447, 934, 734, 127, 587, 843, 916, 901, 370, 671, 206, 768, 291, 453, 746, 913, 729, 825, 435, 553, 319, 529, 549, 748, 760, 162, 53, 254, 659, 542, 629, 815, 172, 285, 33, 256, 284, 186, 266, 886, 108, 202, 884, 893, 148, 907, 442, 641, 69, 732, 74, 658, 358, 528, 252, 296, 344, 1009, 956, 979, 935, 595, 268, 221, 601, 60, 225, 116, 946, 407, 170, 262, 947, 224, 516, 105, 958, 348, 325, 668, 307, 240, 980, 173, 716, 895, 879, 1023, 838, 771, 875, 643, 695, 10, 602, 657, 20, 606, 102, 330, 22, 964, 883, 636, 868, 891, 539, 856, 522, 70, 474, 898, 430, 255, 723, 386, 1, 122, 84, 725, 281, 77, 414, 845, 13, 663, 340, 968, 747, 164, 58, 249, 718, 611, 413, 418, 575, 23, 214, 507, 48, 871, 953, 448, 316, 767, 159, 423, 790, 514, 670, 630, 681, 750, 571, 970, 722, 352, 289, 64, 208, 28, 631, 994, 870, 933, 817, 189, 469, 545, 578, 232, 56, 155, 608, 1013, 476, 692, 839, 662, 981, 223, 238, 209, 690]]]
    #
    # winning_perm_model_sd = []
    # for i in range(5):
    #     winning_perm_model_sd.append(
    #         permute(args.arch, model, winning_permutation[i], sd[i], w[i], nchannels, nclasses, args.width))
    #     # permuted_models.append      (permute(arch, model, self.state[i], sd2[i], w2[i], nchannels, nclasses, nunits))
    # ###### LMC between permuted models
    # pairs = list(itertools.combinations(range(5), 2))
    # pair = 0
    # barrier_test_basin = []
    # for x in pairs:
    #     pair = pair + 1
    #     idx1 = x[0]
    #     idx2 = x[1]
    #     sd1_ = winning_perm_model_sd[idx1]
    #     sd2_ = winning_perm_model_sd[idx2]
    #     dict_after = get_barrier(model, sd1_, sd2_, train_inputs, train_targets, test_inputs, test_targets)
    #
    #     add_element(dict_after, 'winning_permutation', winning_permutation)
    #     add_element(dict_after, 'winning_perm_model_sd', winning_perm_model_sd)
    #
    #     barrier_test = dict_after['barrier_test']
    #     lmc_test = dict_after['test_lmc']
    #
    #     print("barrier_test_SA", barrier_test)
    #     print("lmc_test_SA", lmc_test)
    #     barrier_test_basin.append(barrier_test[0])
    #
    #     source_file_name = f'dict_after_{pair}.pkl'
    #     # destination_blob_name = f'Neurips21/{save_dir}/barrier/SA/auto/{source_file_name}'
    #     destination_blob_name = f'Neurips21/{save_dir}/barrier/SA_InstanceOptimized_v1/oracle/SA/grid/{exp_no}/{source_file_name}'
    #     pickle_out = pickle.dumps(dict_after)
    #     upload_pkl(bucket_name, pickle_out, destination_blob_name)
    # print()
    # print("basin_mean_after", statistics.mean(barrier_test_basin))
    # print("basin_std_after", statistics.stdev(barrier_test_basin))
    #



    # # ########################################## SA original models: model1 and model2
    sd = []
    for j in [7, 8, 9, 15, 16]:
        bucket_name = 'permutation-mlp'
        destination_blob_name = 'model_best.th'
        source_file_name = f'Neurips21/{save_dir}/Train/{j}/{destination_blob_name}'
        download_blob(bucket_name, source_file_name, destination_blob_name)

        checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))

        def key_transformation(old_key):
            if 'module' in old_key:
                return old_key[7:]
            return old_key

        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            new_key = key_transformation(key)
            new_state_dict[new_key] = value
        checkpoint = new_state_dict

        sd.append(checkpoint)

    w = []
    for i in range(5):
        params = []
        for key in sd[i].keys():
            param = sd[i][key]
            params.append(param.cpu().detach().numpy())
        w.append(params)

    conv_arch = False
    for key in sd[0]:
        print(key, sd[0][key].shape)
        if "conv" in key or "running_mean" in key:
            conv_arch = True

    # create permutation list for mlp
    if args.arch == 'mlp':
        len_perm = []
        for i in range(int(len(w[0]) / 2 - 1)):
            len_perm.append(args.width)
    # create permutation list for conv nets
    if conv_arch:
        params = []
        len_perm = []
        for key in sd[0].keys():
            param = model.state_dict()[key]
            if "num_batches_tracked" not in key:
                params.append(param.cpu().detach().numpy())
                if len(param.shape) == 4:
                    len_perm.append(param.shape[0])
                if len(param.shape) == 2:
                    len_perm.append(param.shape[0])

    print("len_perm", len(len_perm))
    print("len_perm", len_perm)


    init_state = []
    for i in range(5):
        random_permuted_index = []
        for z in len_perm:
            lst = [y for y in range(z)]
            random.seed(1)
            rnd = random.sample(lst, z)
            random_permuted_index.append(rnd)
        init_state.append(random_permuted_index)

    exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    winning_permutation = barrier_SA(args.arch, model, sd, w, init_state,
                                     args.tmax, args.tmin, args.steps,
                                     train_inputs, train_targets,
                                     nchannels, nclasses, args.width)
    print("winning_permutation", winning_permutation)
    # winning_permutation = [[[51, 395, 645, 240, 20, 262, 903, 50, 930, 293, 828, 319, 221, 970, 385, 985, 558, 179, 932, 947, 881, 740, 599, 874, 69, 369, 644, 322, 630, 407, 338, 693, 818, 598, 951, 441, 516, 413, 423, 542, 235, 831, 520, 901, 454, 316, 32, 784, 573, 40, 566, 201, 495, 45, 418, 259, 796, 194, 125, 822, 230, 657, 891, 483, 1010, 737, 7, 445, 767, 476, 416, 746, 255, 494, 844, 346, 621, 712, 703, 353, 791, 109, 409, 1007, 887, 959, 436, 639, 339, 198, 629, 8, 183, 782, 112, 946, 399, 12, 246, 1015, 983, 157, 53, 695, 26, 873, 232, 626, 641, 329, 330, 949, 102, 588, 606, 306, 685, 935, 617, 707, 332, 954, 682, 867, 940, 3, 656, 518, 883, 451, 260, 341, 167, 777, 73, 870, 735, 117, 545, 256, 199, 774, 442, 308, 701, 504, 24, 317, 347, 200, 435, 669, 692, 875, 85, 851, 858, 937, 44, 238, 34, 571, 666, 840, 854, 552, 754, 66, 667, 705, 397, 521, 1023, 1019, 188, 360, 897, 866, 957, 261, 439, 124, 661, 860, 908, 716, 182, 401, 525, 815, 801, 379, 447, 913, 460, 812, 914, 964, 575, 433, 215, 832, 83, 169, 489, 233, 530, 206, 759, 819, 321, 146, 381, 22, 468, 415, 60, 211, 855, 431, 537, 837, 921, 508, 234, 943, 218, 772, 506, 294, 562, 714, 1005, 594, 287, 608, 209, 612, 490, 471, 684, 27, 591, 747, 723, 625, 33, 826, 514, 761, 808, 786, 220, 29, 595, 61, 511, 266, 17, 806, 788, 104, 702, 243, 658, 274, 71, 539, 202, 655, 455, 64, 2, 771, 38, 523, 288, 497, 296, 650, 565, 652, 512, 532, 428, 659, 237, 729, 620, 673, 99, 636, 836, 586, 841, 178, 781, 165, 223, 54, 372, 531, 719, 265, 1021, 438, 663, 361, 1017, 88, 480, 474, 242, 925, 880, 488, 94, 884, 615, 279, 59, 528, 743, 817, 197, 6, 865, 103, 677, 250, 604, 992, 960, 839, 440, 152, 607, 756, 596, 36, 895, 541, 139, 670, 335, 805, 425, 299, 533, 337, 35, 568, 929, 792, 613, 450, 343, 509, 579, 89, 877, 963, 654, 522, 196, 14, 790, 285, 16, 680, 147, 479, 477, 462, 1, 734, 364, 192, 270, 484, 349, 574, 141, 660, 1018, 121, 843, 611, 1003, 333, 988, 268, 730, 633, 529, 151, 557, 668, 135, 696, 159, 168, 802, 592, 540, 271, 210, 105, 107, 721, 779, 72, 823, 382, 212, 195, 1000, 166, 384, 609, 43, 5, 155, 632, 678, 219, 519, 700, 400, 833, 581, 408, 548, 309, 70, 295, 189, 628, 113, 706, 948, 745, 144, 340, 63, 538, 448, 576, 998, 554, 527, 153, 838, 138, 857, 378, 93, 1004, 76, 907, 457, 357, 391, 98, 48, 1020, 749, 502, 825, 67, 464, 190, 927, 18, 924, 748, 971, 926, 203, 691, 236, 764, 421, 217, 0, 492, 972, 498, 597, 829, 708, 185, 944, 253, 482, 264, 859, 846, 688, 465, 635, 869, 30, 154, 942, 934, 148, 175, 473, 931, 778, 958, 114, 390, 286, 56, 922, 775, 58, 976, 249, 798, 731, 129, 605, 885, 412, 150, 894, 807, 344, 371, 191, 358, 458, 348, 827, 101, 324, 564, 15, 47, 753, 28, 741, 547, 787, 267, 466, 570, 799, 651, 380, 793, 478, 81, 886, 797, 704, 752, 757, 444, 681, 879, 783, 359, 304, 718, 856, 278, 578, 986, 585, 814, 847, 374, 810, 314, 181, 248, 389, 106, 91, 709, 672, 370, 459, 486, 387, 383, 991, 898, 149, 733, 583, 933, 864, 910, 193, 580, 132, 313, 254, 86, 214, 281, 213, 816, 247, 785, 342, 560, 4, 368, 821, 419, 300, 973, 551, 119, 1012, 429, 809, 283, 780, 919, 563, 732, 982, 161, 862, 133, 375, 999, 952, 405, 751, 559, 95, 619, 417, 961, 363, 830, 848, 163, 115, 968, 590, 569, 403, 122, 676, 298, 1011, 795, 257, 978, 131, 75, 646, 410, 683, 549, 472, 679, 292, 239, 720, 535, 493, 388, 824, 713, 57, 507, 912, 406, 301, 37, 272, 515, 128, 1002, 981, 835, 87, 710, 350, 510, 171, 980, 222, 593, 962, 853, 13, 939, 290, 915, 553, 602, 534, 11, 763, 715, 367, 136, 544, 916, 323, 543, 770, 328, 975, 648, 41, 446, 717, 145, 365, 727, 177, 918, 126, 251, 503, 393, 19, 334, 77, 505, 307, 616, 373, 674, 411, 849, 750, 331, 736, 755, 143, 561, 414, 434, 485, 62, 352, 603, 427, 291, 555, 724, 318, 158, 231, 229, 722, 687, 1001, 500, 773, 868, 690, 536, 872, 280, 989, 513, 969, 23, 90, 108, 803, 116, 899, 904, 310, 426, 996, 945, 863, 643, 276, 769, 228, 966, 640, 351, 226, 834, 974, 726, 184, 386, 634, 888, 463, 377, 302, 461, 21, 275, 876, 871, 258, 437, 241, 766, 882, 452, 845, 917, 244, 889, 46, 517, 1014, 993, 10, 671, 376, 601, 556, 404, 789, 902, 355, 394, 738, 526, 584, 953, 739, 420, 186, 744, 699, 1016, 142, 995, 811, 123, 647, 92, 79, 176, 698, 245, 906, 273, 392, 137, 277, 686, 140, 467, 728, 965, 938, 325, 65, 174, 110, 39, 315, 187, 923, 173, 424, 892, 850, 499, 941, 804, 74, 697, 282, 97, 52, 312, 180, 469, 890, 955, 852, 127, 765, 997, 896, 491, 760, 984, 9, 160, 1008, 637, 311, 204, 987, 622, 1013, 776, 572, 990, 893, 134, 567, 84, 900, 216, 68, 928, 82, 524, 156, 164, 224, 345, 78, 1022, 1006, 861, 55, 336, 909, 694, 305, 289, 263, 269, 25, 430, 920, 130, 354, 813, 624, 653, 356, 638, 979, 162, 642, 366, 208, 205, 327, 800, 1009, 689, 100, 496, 402, 842, 589, 80, 967, 118, 587, 432, 303, 950, 422, 172, 956, 627, 662, 631, 470, 170, 610, 225, 742, 675, 120, 31, 207, 994, 878, 600, 911, 546, 456, 481, 582, 725, 905, 711, 618, 501, 768, 49, 577, 820, 362, 665, 252, 977, 42, 936, 443, 649, 453, 758, 96, 623, 297, 794, 398, 475, 487, 111, 762, 227, 614, 550, 320, 284, 326, 449, 396, 664]], [[980, 939, 405, 139, 387, 13, 656, 546, 969, 931, 1004, 572, 904, 803, 989, 291, 601, 358, 505, 39, 424, 569, 378, 681, 440, 214, 622, 544, 471, 410, 217, 1022, 170, 742, 239, 985, 604, 368, 881, 485, 495, 85, 710, 422, 392, 445, 771, 376, 776, 144, 94, 576, 159, 416, 452, 894, 256, 281, 531, 327, 704, 350, 1016, 521, 693, 759, 299, 840, 185, 964, 791, 748, 114, 799, 918, 486, 160, 595, 517, 914, 425, 72, 884, 646, 589, 675, 986, 443, 533, 17, 502, 10, 464, 639, 773, 732, 833, 631, 588, 484, 86, 625, 57, 655, 462, 490, 211, 179, 215, 279, 56, 207, 265, 889, 975, 666, 176, 846, 71, 231, 873, 75, 95, 926, 988, 556, 698, 902, 643, 102, 101, 198, 828, 537, 623, 747, 872, 278, 103, 635, 874, 634, 616, 233, 686, 450, 116, 374, 89, 680, 527, 762, 491, 178, 540, 47, 1008, 936, 813, 208, 869, 458, 268, 570, 627, 557, 232, 100, 167, 285, 615, 621, 692, 560, 415, 370, 182, 507, 212, 301, 994, 1021, 390, 609, 343, 37, 730, 188, 674, 30, 696, 957, 757, 393, 446, 876, 853, 997, 162, 553, 867, 125, 930, 587, 536, 503, 726, 990, 1018, 722, 220, 709, 583, 414, 240, 61, 397, 115, 890, 109, 355, 247, 995, 868, 218, 255, 628, 302, 135, 558, 1019, 474, 190, 142, 903, 760, 832, 578, 259, 133, 437, 359, 763, 52, 143, 166, 412, 792, 795, 434, 946, 35, 573, 448, 455, 307, 999, 164, 465, 391, 744, 449, 644, 78, 399, 127, 701, 384, 234, 568, 400, 123, 44, 697, 54, 21, 934, 949, 626, 282, 435, 848, 377, 168, 346, 864, 338, 15, 352, 1007, 955, 672, 204, 756, 128, 460, 0, 993, 827, 403, 1001, 787, 727, 594, 5, 866, 51, 2, 252, 577, 981, 529, 64, 304, 630, 702, 597, 266, 784, 482, 1014, 967, 843, 790, 793, 555, 151, 428, 834, 493, 419, 538, 690, 401, 110, 953, 830, 470, 751, 688, 893, 295, 68, 952, 147, 652, 298, 328, 901, 602, 920, 411, 691, 202, 599, 339, 733, 274, 541, 472, 825, 624, 774, 617, 717, 919, 60, 24, 734, 859, 488, 535, 251, 366, 611, 183, 897, 332, 475, 516, 192, 305, 775, 966, 227, 267, 900, 564, 642, 1006, 303, 45, 345, 439, 571, 510, 224, 842, 845, 736, 837, 84, 907, 738, 306, 469, 1013, 479, 447, 620, 381, 324, 525, 645, 348, 360, 812, 849, 1010, 708, 290, 459, 550, 132, 716, 665, 637, 46, 600, 145, 528, 970, 506, 909, 641, 65, 805, 288, 720, 119, 362, 943, 801, 230, 911, 219, 32, 735, 134, 161, 396, 421, 679, 96, 586, 48, 406, 888, 983, 678, 108, 386, 79, 554, 58, 137, 777, 4, 140, 664, 148, 764, 152, 855, 783, 565, 1011, 877, 765, 26, 365, 413, 924, 606, 242, 711, 334, 118, 662, 581, 798, 719, 660, 542, 962, 561, 714, 728, 998, 636, 191, 921, 895, 640, 653, 216, 823, 806, 963, 603, 6, 563, 700, 420, 605, 451, 11, 847, 29, 682, 309, 916, 923, 811, 273, 984, 928, 892, 945, 494, 19, 590, 973, 66, 487, 330, 694, 383, 913, 77, 1, 607, 325, 592, 670, 31, 427, 835, 25, 254, 804, 466, 941, 354, 297, 530, 950, 695, 364, 703, 677, 389, 228, 959, 80, 156, 858, 908, 8, 982, 453, 978, 925, 683, 300, 654, 785, 117, 226, 467, 685, 322, 547, 996, 311, 136, 687, 337, 826, 886, 236, 316, 16, 289, 367, 347, 739, 69, 314, 809, 223, 852, 380, 817, 612, 1012, 28, 87, 741, 431, 885, 146, 122, 53, 778, 107, 264, 948, 574, 82, 875, 824, 810, 344, 292, 870, 712, 3, 9, 50, 272, 906, 715, 22, 74, 130, 619, 815, 937, 927, 173, 598, 34, 201, 1020, 442, 184, 831, 746, 880, 238, 197, 707, 947, 596, 1015, 320, 408, 49, 126, 280, 149, 915, 740, 124, 150, 478, 480, 839, 187, 36, 138, 584, 90, 407, 585, 896, 754, 349, 731, 539, 81, 283, 820, 59, 938, 385, 457, 423, 887, 650, 851, 871, 372, 141, 649, 992, 961, 62, 20, 270, 942, 509, 856, 780, 863, 258, 829, 514, 155, 879, 432, 321, 375, 743, 991, 318, 477, 157, 657, 222, 671, 814, 335, 899, 860, 426, 684, 725, 857, 199, 891, 165, 534, 468, 394, 543, 245, 73, 99, 905, 312, 629, 614, 174, 1017, 120, 632, 882, 854, 489, 647, 663, 786, 749, 418, 838, 315, 782, 689, 373, 794, 504, 713, 271, 808, 409, 97, 175, 515, 40, 968, 850, 960, 113, 193, 971, 356, 382, 163, 878, 706, 402, 456, 250, 974, 429, 800, 308, 758, 522, 524, 91, 284, 483, 575, 865, 169, 7, 241, 844, 562, 441, 417, 932, 331, 767, 944, 210, 189, 822, 633, 910, 753, 591, 551, 933, 755, 745, 253, 194, 249, 501, 816, 438, 463, 1003, 196, 548, 976, 235, 638, 929, 667, 243, 508, 724, 195, 203, 12, 293, 41, 737, 545, 42, 461, 129, 898, 593, 526, 351, 433, 1005, 956, 353, 336, 287, 33, 559, 92, 319, 121, 23, 669, 310, 171, 567, 518, 935, 549, 1009, 269, 750, 705, 779, 772, 979, 263, 532, 807, 388, 781, 912, 209, 200, 497, 661, 14, 221, 673, 275, 492, 342, 513, 104, 206, 294, 260, 262, 444, 404, 987, 496, 1023, 246, 954, 379, 579, 172, 83, 43, 821, 98, 277, 1002, 131, 552, 473, 333, 651, 67, 181, 476, 699, 158, 818, 658, 659, 676, 770, 500, 70, 769, 106, 244, 836, 329, 862, 105, 958, 313, 55, 63, 618, 861, 111, 237, 186, 768, 76, 205, 511, 608, 580, 180, 398, 213, 582, 357, 248, 340, 965, 668, 940, 752, 112, 819, 481, 154, 789, 613, 972, 883, 27, 93, 761, 38, 723, 802, 788, 296, 520, 369, 917, 326, 286, 796, 1000, 566, 363, 257, 276, 88, 371, 512, 718, 341, 610, 841, 153, 436, 766, 317, 18, 498, 430, 225, 922, 229, 523, 361, 797, 395, 454, 499, 323, 177, 519, 261, 729, 721, 951, 977, 648]], [[390, 793, 403, 728, 249, 97, 166, 731, 885, 231, 824, 844, 431, 264, 636, 108, 944, 118, 302, 415, 439, 420, 472, 372, 660, 268, 119, 120, 236, 722, 423, 335, 474, 106, 304, 452, 228, 859, 911, 307, 239, 826, 405, 412, 274, 1, 476, 881, 49, 748, 95, 186, 371, 345, 467, 740, 972, 833, 47, 918, 247, 743, 523, 508, 199, 901, 682, 481, 5, 89, 39, 280, 865, 569, 909, 778, 733, 642, 982, 711, 96, 384, 68, 355, 839, 628, 543, 102, 66, 990, 600, 293, 235, 514, 517, 558, 927, 134, 300, 51, 609, 190, 699, 836, 819, 858, 459, 203, 779, 210, 943, 310, 20, 201, 375, 594, 167, 554, 615, 828, 388, 316, 936, 694, 542, 922, 515, 422, 805, 690, 830, 299, 795, 330, 112, 442, 410, 449, 673, 953, 255, 613, 620, 114, 585, 997, 545, 874, 563, 212, 57, 1003, 132, 309, 875, 811, 708, 81, 348, 269, 534, 948, 646, 774, 105, 211, 866, 772, 786, 519, 757, 392, 1004, 182, 904, 533, 123, 910, 305, 829, 245, 331, 752, 800, 194, 347, 336, 436, 389, 857, 84, 684, 202, 612, 812, 926, 451, 700, 570, 441, 817, 288, 606, 539, 668, 477, 53, 457, 883, 967, 634, 432, 38, 894, 614, 222, 838, 907, 221, 692, 504, 993, 1006, 631, 712, 709, 616, 847, 447, 282, 1001, 787, 59, 25, 704, 75, 890, 238, 942, 99, 892, 667, 93, 400, 574, 209, 681, 229, 785, 861, 992, 263, 69, 965, 58, 395, 327, 306, 464, 618, 737, 326, 296, 659, 559, 82, 710, 233, 275, 790, 536, 912, 225, 855, 359, 421, 110, 393, 557, 44, 185, 385, 41, 363, 969, 661, 555, 178, 957, 562, 572, 617, 386, 56, 260, 402, 573, 643, 64, 906, 654, 344, 763, 87, 732, 924, 611, 958, 237, 29, 697, 144, 133, 598, 889, 168, 849, 321, 960, 940, 902, 241, 357, 527, 475, 509, 913, 242, 142, 398, 374, 683, 150, 92, 962, 356, 1011, 131, 320, 537, 490, 303, 625, 315, 917, 24, 214, 821, 794, 657, 744, 586, 207, 273, 42, 556, 512, 487, 160, 159, 792, 473, 314, 184, 141, 977, 32, 215, 549, 339, 635, 996, 548, 34, 653, 898, 810, 213, 254, 318, 1009, 285, 153, 218, 125, 771, 622, 766, 419, 765, 2, 832, 427, 414, 575, 28, 1015, 411, 698, 460, 725, 963, 54, 343, 671, 227, 468, 290, 896, 169, 715, 62, 366, 648, 177, 846, 561, 971, 729, 373, 399, 511, 7, 12, 340, 588, 921, 287, 364, 191, 518, 755, 919, 739, 747, 6, 250, 999, 520, 638, 197, 171, 566, 1021, 633, 652, 522, 505, 882, 454, 329, 308, 195, 970, 981, 689, 67, 540, 353, 8, 802, 22, 91, 589, 568, 564, 576, 870, 773, 930, 979, 183, 935, 1002, 706, 493, 456, 444, 193, 808, 813, 376, 361, 196, 531, 666, 458, 835, 426, 730, 478, 109, 605, 713, 55, 860, 324, 40, 841, 101, 696, 179, 989, 367, 877, 968, 807, 281, 136, 937, 15, 623, 1007, 734, 50, 78, 544, 1014, 769, 226, 445, 219, 676, 76, 438, 675, 798, 198, 341, 597, 650, 461, 900, 867, 584, 272, 656, 352, 916, 262, 135, 735, 818, 1012, 521, 754, 702, 297, 107, 80, 591, 418, 200, 429, 284, 261, 868, 446, 9, 862, 507, 899, 665, 724, 762, 884, 753, 974, 759, 985, 407, 923, 46, 719, 328, 644, 380, 816, 152, 294, 770, 143, 397, 248, 358, 629, 35, 571, 995, 434, 678, 391, 526, 122, 908, 424, 579, 365, 721, 781, 1019, 265, 647, 286, 346, 496, 127, 234, 905, 289, 181, 891, 488, 158, 342, 939, 387, 501, 73, 592, 138, 925, 601, 599, 751, 19, 4, 632, 437, 929, 466, 240, 27, 945, 510, 482, 224, 13, 608, 362, 497, 975, 52, 253, 961, 479, 1016, 791, 578, 720, 827, 854, 506, 664, 959, 295, 976, 895, 334, 834, 780, 736, 121, 687, 220, 1000, 88, 10, 377, 856, 680, 3, 1022, 187, 815, 145, 471, 994, 33, 853, 140, 271, 837, 277, 933, 172, 139, 500, 931, 485, 560, 991, 525, 495, 369, 820, 124, 499, 746, 216, 117, 94, 577, 26, 360, 872, 36, 147, 349, 175, 333, 797, 541, 775, 312, 243, 701, 470, 298, 947, 383, 283, 126, 546, 354, 823, 869, 580, 455, 663, 768, 161, 463, 938, 998, 16, 368, 65, 888, 626, 973, 63, 538, 750, 850, 955, 707, 43, 621, 871, 950, 651, 157, 417, 567, 789, 325, 602, 1008, 777, 717, 723, 583, 920, 483, 151, 404, 529, 311, 450, 323, 165, 880, 98, 223, 987, 587, 61, 208, 291, 146, 433, 760, 413, 801, 809, 978, 814, 103, 498, 130, 149, 822, 581, 741, 886, 934, 952, 188, 337, 658, 831, 610, 552, 492, 1017, 173, 465, 79, 322, 903, 338, 604, 156, 928, 863, 776, 491, 137, 530, 484, 71, 313, 408, 206, 686, 714, 627, 319, 409, 914, 17, 252, 645, 983, 873, 840, 964, 641, 693, 986, 749, 18, 649, 0, 100, 489, 738, 469, 516, 217, 443, 85, 379, 553, 788, 674, 351, 590, 1018, 758, 164, 70, 624, 915, 767, 803, 954, 30, 170, 21, 984, 876, 246, 406, 266, 204, 842, 524, 897, 716, 503, 799, 806, 256, 378, 486, 742, 852, 550, 1023, 825, 128, 640, 86, 784, 113, 162, 528, 230, 691, 332, 703, 174, 278, 416, 111, 259, 401, 1020, 428, 551, 90, 887, 83, 232, 317, 292, 301, 718, 804, 637, 745, 60, 845, 453, 502, 941, 205, 595, 596, 630, 104, 655, 1005, 761, 932, 396, 670, 980, 851, 662, 72, 381, 448, 430, 155, 879, 672, 279, 685, 966, 603, 192, 705, 695, 669, 726, 74, 988, 258, 532, 565, 727, 756, 189, 440, 864, 494, 1013, 48, 129, 462, 267, 593, 31, 677, 679, 270, 639, 949, 23, 848, 251, 382, 547, 77, 878, 176, 37, 257, 956, 783, 14, 946, 764, 607, 45, 115, 148, 425, 951, 843, 796, 893, 582, 11, 394, 163, 619, 535, 350, 480, 688, 244, 116, 154, 1010, 370, 276, 513, 782, 435, 180]], [[26, 724, 113, 59, 781, 800, 330, 481, 337, 156, 903, 548, 185, 15, 169, 811, 944, 801, 265, 342, 541, 476, 82, 945, 351, 530, 715, 618, 908, 766, 1005, 693, 226, 440, 950, 842, 355, 244, 733, 236, 326, 88, 596, 581, 955, 551, 976, 1001, 181, 769, 519, 978, 374, 832, 139, 384, 499, 297, 668, 748, 859, 0, 924, 610, 157, 221, 333, 513, 311, 406, 898, 716, 106, 972, 694, 841, 142, 104, 721, 905, 515, 241, 171, 222, 971, 977, 778, 302, 39, 64, 315, 216, 349, 592, 319, 646, 729, 130, 631, 619, 539, 836, 47, 195, 828, 449, 589, 498, 144, 329, 435, 912, 632, 56, 900, 962, 571, 402, 314, 198, 203, 645, 93, 809, 450, 746, 609, 506, 823, 942, 358, 636, 212, 964, 784, 273, 558, 886, 10, 366, 580, 909, 740, 745, 981, 475, 65, 783, 793, 338, 701, 926, 361, 545, 204, 767, 853, 206, 376, 249, 92, 529, 500, 614, 600, 759, 207, 790, 754, 109, 574, 5, 474, 2, 1023, 122, 552, 178, 220, 802, 100, 248, 225, 284, 732, 94, 32, 258, 535, 718, 145, 191, 641, 791, 815, 459, 576, 495, 62, 152, 876, 889, 490, 838, 966, 37, 709, 132, 155, 354, 362, 960, 963, 428, 734, 385, 418, 628, 470, 391, 469, 403, 835, 904, 310, 726, 856, 461, 388, 922, 462, 58, 107, 404, 158, 274, 511, 345, 650, 952, 153, 129, 1013, 409, 306, 543, 68, 991, 436, 304, 341, 1016, 591, 451, 379, 821, 60, 264, 1022, 7, 180, 518, 141, 728, 990, 260, 487, 797, 368, 408, 812, 834, 369, 538, 621, 439, 700, 910, 118, 690, 27, 312, 667, 570, 165, 353, 45, 240, 63, 544, 363, 420, 172, 270, 738, 151, 419, 707, 1021, 999, 664, 872, 303, 131, 209, 925, 526, 965, 635, 698, 688, 611, 323, 121, 946, 364, 18, 516, 585, 663, 595, 251, 954, 762, 211, 651, 647, 54, 702, 234, 29, 719, 414, 70, 705, 837, 242, 160, 865, 980, 864, 218, 437, 894, 888, 115, 327, 587, 597, 522, 252, 324, 830, 494, 782, 931, 804, 16, 170, 934, 657, 785, 593, 472, 975, 40, 370, 350, 331, 684, 339, 786, 120, 479, 758, 200, 416, 820, 30, 661, 380, 866, 458, 411, 215, 666, 789, 493, 869, 69, 98, 969, 348, 445, 308, 572, 798, 524, 279, 717, 706, 41, 656, 298, 224, 508, 689, 723, 154, 480, 770, 184, 996, 116, 826, 263, 491, 839, 51, 675, 557, 849, 932, 554, 941, 431, 792, 523, 902, 1019, 831, 239, 1015, 559, 286, 612, 637, 357, 33, 556, 893, 432, 517, 764, 760, 858, 101, 757, 874, 28, 456, 86, 584, 114, 6, 372, 85, 3, 484, 703, 174, 318, 390, 305, 985, 679, 569, 603, 860, 1011, 321, 287, 984, 22, 731, 397, 510, 756, 607, 486, 205, 822, 714, 189, 884, 601, 168, 928, 627, 805, 489, 777, 638, 863, 906, 340, 393, 528, 21, 410, 652, 534, 833, 140, 295, 795, 13, 19, 89, 560, 816, 301, 998, 871, 78, 901, 53, 504, 44, 79, 895, 590, 425, 606, 238, 973, 761, 377, 356, 442, 1008, 166, 779, 396, 658, 594, 136, 434, 915, 634, 625, 527, 177, 583, 501, 672, 123, 399, 995, 568, 624, 825, 455, 313, 533, 8, 775, 71, 111, 881, 936, 670, 162, 99, 84, 188, 712, 20, 230, 74, 887, 855, 448, 259, 655, 344, 725, 1018, 577, 457, 293, 182, 1017, 660, 452, 752, 654, 80, 49, 183, 228, 807, 542, 454, 275, 720, 806, 328, 780, 332, 682, 83, 187, 696, 873, 993, 360, 713, 322, 389, 773, 197, 742, 854, 917, 861, 883, 23, 940, 352, 829, 477, 687, 173, 639, 892, 192, 164, 824, 937, 429, 817, 852, 325, 992, 255, 956, 623, 31, 848, 412, 982, 277, 175, 744, 678, 622, 103, 43, 862, 75, 847, 433, 643, 57, 566, 880, 143, 642, 743, 468, 617, 930, 916, 844, 711, 613, 540, 375, 17, 885, 927, 347, 492, 920, 665, 686, 427, 737, 231, 626, 334, 648, 76, 73, 763, 299, 383, 677, 465, 644, 290, 814, 482, 289, 125, 346, 896, 443, 921, 988, 555, 196, 890, 359, 466, 309, 818, 407, 599, 851, 579, 509, 243, 681, 741, 514, 730, 159, 400, 72, 671, 768, 288, 774, 421, 478, 278, 674, 483, 202, 81, 135, 371, 336, 1000, 813, 604, 405, 1004, 257, 294, 138, 335, 463, 250, 772, 34, 563, 247, 683, 739, 12, 983, 582, 588, 727, 105, 755, 989, 35, 210, 394, 708, 163, 567, 870, 444, 913, 268, 67, 1006, 381, 52, 398, 296, 564, 119, 87, 42, 899, 953, 747, 422, 1003, 387, 24, 1014, 134, 256, 598, 367, 699, 280, 961, 553, 415, 776, 261, 970, 615, 464, 620, 77, 233, 808, 1010, 520, 217, 662, 254, 229, 446, 395, 507, 605, 685, 1012, 602, 438, 918, 967, 4, 974, 272, 117, 575, 237, 750, 497, 149, 1007, 193, 697, 959, 235, 190, 108, 630, 938, 423, 179, 9, 505, 283, 316, 788, 819, 11, 968, 676, 521, 751, 282, 46, 878, 146, 285, 223, 271, 110, 401, 680, 810, 246, 378, 659, 496, 911, 547, 291, 549, 453, 14, 633, 66, 629, 133, 669, 919, 867, 300, 55, 97, 923, 343, 199, 148, 485, 25, 128, 102, 649, 979, 96, 137, 365, 787, 857, 735, 386, 951, 441, 502, 765, 424, 38, 1020, 276, 573, 771, 546, 653, 36, 933, 426, 673, 616, 692, 799, 997, 947, 736, 691, 150, 50, 929, 753, 219, 413, 827, 473, 503, 987, 948, 430, 891, 882, 640, 373, 986, 562, 550, 176, 471, 267, 943, 868, 565, 794, 194, 417, 90, 253, 608, 531, 939, 112, 227, 232, 127, 48, 147, 722, 935, 292, 840, 208, 749, 382, 525, 213, 245, 846, 1009, 875, 91, 95, 586, 214, 317, 126, 447, 803, 949, 958, 796, 843, 704, 897, 266, 695, 994, 578, 957, 307, 161, 167, 61, 488, 879, 1, 561, 281, 512, 124, 907, 269, 186, 532, 262, 467, 320, 877, 537, 536, 392, 1002, 845, 201, 710, 460, 850, 914]], [[236, 1011, 783, 954, 174, 782, 31, 546, 62, 420, 502, 894, 429, 19, 1008, 298, 258, 941, 134, 745, 645, 104, 762, 135, 455, 967, 51, 392, 814, 562, 844, 79, 361, 192, 710, 379, 321, 81, 459, 1017, 257, 673, 24, 556, 853, 88, 513, 156, 472, 927, 125, 848, 650, 295, 272, 583, 701, 377, 1000, 551, 824, 577, 938, 111, 798, 383, 721, 501, 864, 860, 991, 945, 580, 149, 931, 735, 140, 356, 536, 12, 446, 394, 829, 554, 589, 797, 627, 360, 100, 350, 245, 473, 341, 832, 740, 789, 243, 517, 411, 887, 120, 496, 607, 143, 391, 989, 436, 744, 714, 573, 997, 309, 834, 131, 499, 32, 375, 434, 759, 339, 485, 468, 992, 274, 17, 445, 766, 737, 424, 329, 450, 205, 858, 720, 199, 126, 537, 999, 696, 323, 560, 466, 503, 169, 90, 290, 950, 123, 1005, 85, 543, 196, 680, 854, 826, 896, 984, 656, 538, 960, 180, 145, 773, 283, 197, 303, 963, 311, 717, 612, 754, 477, 138, 623, 660, 401, 431, 688, 2, 753, 187, 519, 36, 763, 52, 900, 628, 846, 417, 567, 806, 687, 557, 210, 306, 533, 784, 780, 124, 836, 510, 163, 930, 859, 37, 929, 987, 802, 678, 882, 599, 527, 282, 833, 82, 47, 139, 781, 388, 480, 819, 230, 271, 741, 959, 185, 61, 772, 158, 621, 415, 652, 547, 106, 66, 1006, 389, 672, 427, 665, 63, 87, 921, 640, 711, 403, 928, 275, 231, 222, 38, 831, 820, 530, 46, 917, 93, 515, 342, 457, 803, 769, 899, 449, 279, 869, 315, 872, 54, 493, 972, 250, 419, 217, 1020, 198, 793, 698, 132, 76, 177, 966, 362, 478, 227, 971, 604, 219, 704, 333, 171, 248, 603, 624, 57, 16, 807, 855, 265, 605, 129, 918, 267, 908, 752, 416, 1001, 273, 454, 863, 387, 952, 1010, 585, 619, 314, 758, 456, 840, 212, 511, 559, 973, 133, 439, 828, 926, 867, 889, 563, 638, 743, 55, 876, 68, 637, 378, 697, 299, 590, 726, 1007, 940, 83, 9, 1021, 405, 364, 881, 437, 862, 738, 504, 288, 1016, 332, 731, 957, 294, 661, 0, 203, 354, 975, 467, 788, 286, 706, 521, 982, 885, 371, 460, 905, 73, 904, 920, 160, 852, 463, 426, 235, 676, 494, 491, 347, 4, 634, 596, 642, 865, 278, 816, 677, 756, 287, 368, 693, 200, 475, 684, 715, 8, 810, 497, 335, 300, 253, 130, 579, 936, 568, 667, 598, 95, 978, 685, 157, 72, 369, 581, 27, 548, 380, 727, 218, 346, 509, 29, 25, 337, 742, 365, 119, 305, 812, 561, 939, 804, 141, 709, 647, 152, 42, 167, 137, 526, 850, 699, 270, 128, 320, 113, 483, 564, 366, 847, 648, 664, 175, 874, 432, 372, 702, 367, 244, 800, 998, 943, 109, 523, 263, 440, 609, 349, 479, 632, 792, 914, 92, 324, 18, 651, 304, 184, 495, 572, 327, 50, 7, 827, 406, 121, 703, 412, 679, 292, 276, 962, 404, 1012, 873, 393, 498, 977, 14, 757, 625, 142, 534, 168, 410, 591, 49, 749, 739, 785, 118, 328, 674, 923, 686, 86, 809, 866, 179, 317, 75, 97, 922, 458, 996, 837, 1018, 915, 34, 89, 842, 986, 399, 597, 470, 482, 355, 264, 188, 451, 353, 912, 649, 242, 193, 786, 422, 136, 951, 312, 154, 191, 937, 428, 310, 30, 213, 40, 390, 540, 376, 487, 334, 471, 204, 409, 764, 576, 570, 489, 385, 397, 719, 464, 565, 150, 241, 331, 569, 277, 691, 890, 911, 15, 910, 707, 3, 359, 730, 694, 965, 512, 269, 613, 909, 942, 861, 594, 683, 183, 147, 925, 1002, 488, 444, 115, 995, 813, 338, 96, 525, 59, 532, 976, 849, 508, 5, 114, 675, 103, 461, 45, 644, 505, 592, 944, 44, 421, 490, 101, 228, 110, 818, 326, 313, 190, 615, 165, 402, 949, 182, 566, 301, 617, 166, 835, 1003, 618, 633, 705, 117, 796, 777, 91, 728, 880, 107, 736, 779, 99, 535, 226, 246, 574, 43, 544, 151, 1015, 932, 733, 520, 438, 484, 293, 492, 666, 345, 586, 751, 584, 776, 302, 465, 176, 791, 381, 635, 395, 614, 1022, 297, 336, 11, 374, 98, 41, 857, 712, 308, 969, 821, 441, 486, 201, 112, 400, 280, 181, 524, 550, 974, 616, 1004, 322, 234, 794, 67, 654, 600, 531, 233, 775, 778, 146, 755, 558, 902, 6, 877, 555, 518, 26, 373, 237, 178, 708, 903, 808, 506, 1019, 194, 700, 689, 382, 655, 153, 626, 452, 841, 822, 961, 582, 211, 770, 983, 993, 433, 653, 646, 823, 988, 398, 552, 892, 990, 622, 830, 481, 906, 765, 161, 610, 384, 669, 351, 774, 144, 343, 724, 195, 897, 955, 1014, 799, 408, 588, 500, 761, 39, 239, 207, 35, 65, 71, 713, 357, 229, 811, 425, 851, 620, 259, 94, 795, 220, 215, 247, 985, 878, 216, 443, 251, 801, 682, 21, 462, 948, 787, 80, 78, 318, 639, 919, 260, 541, 888, 396, 261, 924, 363, 805, 593, 447, 934, 734, 127, 587, 843, 916, 901, 370, 671, 206, 768, 291, 453, 746, 913, 729, 825, 435, 553, 319, 529, 549, 748, 760, 162, 53, 254, 659, 542, 629, 815, 172, 285, 33, 256, 284, 186, 266, 886, 108, 202, 884, 893, 148, 907, 442, 641, 69, 732, 74, 658, 358, 528, 252, 296, 344, 1009, 956, 979, 935, 595, 268, 221, 601, 60, 225, 116, 946, 407, 170, 262, 947, 224, 516, 105, 958, 348, 325, 668, 307, 240, 980, 173, 716, 895, 879, 1023, 838, 771, 875, 643, 695, 10, 602, 657, 20, 606, 102, 330, 22, 964, 883, 636, 868, 891, 539, 856, 522, 70, 474, 898, 430, 255, 723, 386, 1, 122, 84, 725, 281, 77, 414, 845, 13, 663, 340, 968, 747, 164, 58, 249, 718, 611, 413, 418, 575, 23, 214, 507, 48, 871, 953, 448, 316, 767, 159, 423, 790, 514, 670, 630, 681, 750, 571, 970, 722, 352, 289, 64, 208, 28, 631, 994, 870, 933, 817, 189, 469, 545, 578, 232, 56, 155, 608, 1013, 476, 692, 839, 662, 981, 223, 238, 209, 690]]]

    winning_perm_model_sd = []
    for i in range(5):
        winning_perm_model_sd.append(permute(args.arch, model, winning_permutation[i], sd[i], w[i], nchannels, nclasses, args.width))
        # permuted_models.append      (permute(arch, model, self.state[i], sd2[i], w2[i], nchannels, nclasses, nunits))
    ###### LMC between permuted models
    pairs = list(itertools.combinations(range(5), 2))
    pair = 0
    barrier_test_basin = []
    for x in pairs:
        pair = pair + 1
        idx1 = x[0]
        idx2 = x[1]
        sd1_ = winning_perm_model_sd[idx1]
        sd2_ = winning_perm_model_sd[idx2]
        dict_after = get_barrier(model, sd1_, sd2_, train_inputs, train_targets, test_inputs, test_targets)

        add_element(dict_after, 'winning_permutation', winning_permutation)
        add_element(dict_after, 'winning_perm_model_sd', winning_perm_model_sd)

        barrier_test = dict_after['barrier_test']
        lmc_test = dict_after['test_lmc']

        print("barrier_test_SA", barrier_test)
        print("lmc_test_SA", lmc_test)
        barrier_test_basin.append(barrier_test[0])

        source_file_name = f'dict_after_{pair}.pkl'
        # destination_blob_name = f'Neurips21/{save_dir}/barrier/SA/auto/{source_file_name}'
        destination_blob_name = f'Neurips21/{save_dir}/barrier/SA_InstanceOptimized_v1/grid/{exp_no}/{source_file_name}'
        pickle_out = pickle.dumps(dict_after)
        upload_pkl(bucket_name, pickle_out, destination_blob_name)
    print()
    print("basin_mean_after", statistics.mean(barrier_test_basin))
    print("basin_std_after", statistics.stdev(barrier_test_basin))



    stop = timeit.default_timer()
    print('Time: ', stop - start)
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_pkl(bucket_name, pickle_out, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(pickle_out)


def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return  pickle.loads(pickle_in)


def load_data(split, dataset_name, datadir, nchannels):
    ## https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    elif dataset_name == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
    elif dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset_name == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def calc_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate_model(model, inputs, targets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for data, target in zip(inputs, targets):
            # input = data.to(device).view(data.size(0), -1)
            input = data.to(device)
            target = target.to(device)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1 = calc_accuracy(output, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))
            # break
        # results = dict(top1=top1.avg, loss=losses.avg, batch_time=batch_time.avg)
        results = dict(top1=top1.avg)

    return {key: float(val) for key, val in results.items()}



def evaluate_model_small(model, inputs, targets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for data, target in zip(inputs, targets):
            # input = data.to(device).view(data.size(0), -1)
            input = data.to(device)
            target = target.to(device)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1 = calc_accuracy(output, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))
            break
        # results = dict(top1=top1.avg, loss=losses.avg, batch_time=batch_time.avg)
        results = dict(top1=top1.avg)

    return {key: float(val) for key, val in results.items()}


def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
            for key in state_dict_1.keys()}



def get_barrier(model, sd1, sd2, train_inputs, train_targets, test_inputs, test_targets):
    dict_barrier = {}
    ####################### get the barrier - before permutation
    ###### LMC
    weights = np.linspace(0, 1, 11)
    result_test = []
    result_train = []
    for i in range(len(weights)):
        model.load_state_dict(interpolate_state_dicts(sd1, sd2, weights[i]))
        result_train.append(evaluate_model(model, train_inputs, train_targets)['top1'])
        result_test.append(evaluate_model(model, test_inputs, test_targets)['top1'])

    model1_eval = result_test[0]
    model2_eval = result_test[-1]
    test_avg_models = (model1_eval + model2_eval) / 2

    model1_eval = result_train[0]
    model2_eval = result_train[-1]
    train_avg_models = (model1_eval + model2_eval) / 2

    add_element(dict_barrier, 'train_avg_models', train_avg_models)
    add_element(dict_barrier, 'test_avg_models', test_avg_models)
    add_element(dict_barrier, 'train_lmc', result_train)
    add_element(dict_barrier, 'test_lmc', result_test)
    add_element(dict_barrier, 'barrier_test', test_avg_models - result_test[5])
    add_element(dict_barrier, 'barrier_train', train_avg_models - result_train[5])

    return dict_barrier


def permute(arch, model, perm_ind, sd, w_2, nchannels, nclasses, nunits):

    # 1layer
    if arch == 'mlp' and len(w_2) == 4:

        ################################################ permutation
        idx = perm_ind[0]
        # print(idx)
        ######################### permute weights of model2, based on idx
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]

        w1_p = w1[idx, :]
        b1_p = b1[idx]
        w2_p = w2[:, idx]
        b2_p = b2

        ##################### save model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP1_layer(n_units=nunits, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.3.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.3.bias"][:] = torch.Tensor(b2_p)
    # 2layers
    elif arch == 'mlp' and len(w_2) == 6:
        ######################### permute weights of model2, based on idx
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]

        idx1 = perm_ind[0]
        w1_p = w1[idx1, :]
        b1_p = b1[idx1]

        idx2 = perm_ind[1]
        w2_p = w2[:, idx1]
        w2_p = w2_p[idx2, :]
        b2_p = b2[idx2]

        idx2 = perm_ind[1]
        w3_p = w3[:, idx2]
        b3_p = b3

        ##################### save model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP2_layer(n_units=nunits, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.3.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.3.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.5.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.5.bias"][:] = torch.Tensor(b3_p)
    # 4layers
    elif arch == 'mlp' and len(w_2) == 10:
        ######################### permute weights of model2, based on idx
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]
        w4 = w_2[6]
        b4 = w_2[7]
        w5 = w_2[8]
        b5 = w_2[9]

        idx1 = perm_ind[0]
        w1_p = w1[idx1, :]
        b1_p = b1[idx1]

        w2_p = w2[:, idx1]
        idx2 = perm_ind[1]
        w2_p = w2_p[idx2, :]
        b2_p = b2[idx2]

        w3_p = w3[:, idx2]
        idx3 = perm_ind[2]
        w3_p = w3_p[idx3, :]
        b3_p = b3[idx3]

        w4_p = w4[:, idx3]
        idx4 = perm_ind[3]
        w4_p = w4_p[idx4, :]
        b4_p = b4[idx4]

        w5_p = w5[:, idx4]
        b5_p = b5
        ##################### save model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### 4layers
        model = MLP4_layer(n_units=nunits, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.3.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.3.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.5.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.5.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["layers.7.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["layers.7.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["layers.9.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["layers.9.bias"][:] = torch.Tensor(b5_p)
    # 8layers
    elif arch == 'mlp' and len(w_2) == 18:
        ######################### permute weights of model2, based on idx
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]
        w4 = w_2[6]
        b4 = w_2[7]
        w5 = w_2[8]
        b5 = w_2[9]
        w6 = w_2[10]
        b6 = w_2[11]
        w7 = w_2[12]
        b7 = w_2[13]
        w8 = w_2[14]
        b8 = w_2[15]
        w9 = w_2[16]
        b9 = w_2[17]

        idx1 = perm_ind[0]
        w1_p = w1[idx1, :]
        b1_p = b1[idx1]

        w2_p = w2[:, idx1]
        idx2 = perm_ind[1]
        w2_p = w2_p[idx2, :]
        b2_p = b2[idx2]

        w3_p = w3[:, idx2]
        idx3 = perm_ind[2]
        w3_p = w3_p[idx3, :]
        b3_p = b3[idx3]

        w4_p = w4[:, idx3]
        idx4 = perm_ind[3]
        w4_p = w4_p[idx4, :]
        b4_p = b4[idx4]

        w5_p = w5[:, idx4]
        idx5 = perm_ind[4]
        w5_p = w5_p[idx5, :]
        b5_p = b5[idx5]

        w6_p = w6[:, idx5]
        idx6 = perm_ind[5]
        w6_p = w6_p[idx6, :]
        b6_p = b6[idx6]

        w7_p = w7[:, idx6]
        idx7 = perm_ind[6]
        w7_p = w7_p[idx7, :]
        b7_p = b7[idx7]

        w8_p = w8[:, idx7]
        idx8 = perm_ind[7]
        w8_p = w8_p[idx8, :]
        b8_p = b8[idx8]

        w9_p = w9[:, idx8]
        b9_p = b9

        ##################### save model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### 8layers
        model = MLP8_layer(n_units=nunits, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.3.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.3.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.5.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.5.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["layers.7.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["layers.7.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["layers.9.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["layers.9.bias"][:] = torch.Tensor(b5_p)
        model.state_dict()["layers.11.weight"][:] = torch.Tensor(w6_p)
        model.state_dict()["layers.11.bias"][:] = torch.Tensor(b6_p)
        model.state_dict()["layers.13.weight"][:] = torch.Tensor(w7_p)
        model.state_dict()["layers.13.bias"][:] = torch.Tensor(b7_p)
        model.state_dict()["layers.15.weight"][:] = torch.Tensor(w8_p)
        model.state_dict()["layers.15.bias"][:] = torch.Tensor(b8_p)
        model.state_dict()["layers.17.weight"][:] = torch.Tensor(w9_p)
        model.state_dict()["layers.17.bias"][:] = torch.Tensor(b9_p)
        # Shallow conv 2 layer
    elif arch == 's_conv' and len(w_2) == 6:
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]

        w1_p = w1[perm_ind[0], :, :, :]
        b1_p = b1[perm_ind[0]]
        ##################################### layer 2 --== conv
        w_p = w2[:, perm_ind[0], :, :]
        w2_p = w_p[perm_ind[1], :, :, :]
        b2_p = b2[perm_ind[1]]

        ##################################### layer 3 ===== linear
        w3_p = w3[:, perm_ind[1]]
        b3_p = b3

        model = s_conv_2layer(nchannels, args.width, nclasses)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.state_dict()["conv1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["conv1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["conv2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["conv2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["classifier.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["classifier.bias"][:] = torch.Tensor(b3_p)
        # Shallow conv 4 layer
    elif arch == 's_conv' and len(w_2) == 10:
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]
        w4 = w_2[6]
        b4 = w_2[7]
        w5 = w_2[8]
        b5 = w_2[9]

        w1_p = w1[perm_ind[0], :, :, :]
        b1_p = b1[perm_ind[0]]
        ##################################### layer 2 --== conv
        w_p = w2[:, perm_ind[0], :, :]
        w2_p = w_p[perm_ind[1], :, :, :]
        b2_p = b2[perm_ind[1]]
        ##################################### layer 3 --== conv
        w_p = w3[:, perm_ind[1], :, :]
        w3_p = w_p[perm_ind[2], :, :, :]
        b3_p = b3[perm_ind[2]]
        ##################################### layer 4 --== conv
        w_p = w4[:, perm_ind[2], :, :]
        w4_p = w_p[perm_ind[3], :, :, :]
        b4_p = b4[perm_ind[3]]
        ##################################### layer 5 ===== linear
        w5_p = w5[:, perm_ind[3]]
        b5_p = b5

        model = s_conv_4layer(nchannels, args.width, nclasses)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.state_dict()["conv1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["conv1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["conv2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["conv2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["conv3.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["conv3.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["conv4.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["conv4.bias"][:] = torch.Tensor(b4_p)

        model.state_dict()["classifier.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["classifier.bias"][:] = torch.Tensor(b5_p)
        # Shallow conv 6 layer
    elif arch == 's_conv' and len(w_2) == 14:
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]
        w4 = w_2[6]
        b4 = w_2[7]
        w5 = w_2[8]
        b5 = w_2[9]
        w6 = w_2[10]
        b6 = w_2[11]
        w7 = w_2[12]
        b7 = w_2[13]

        w1_p = w1[perm_ind[0], :, :, :]
        b1_p = b1[perm_ind[0]]
        ##################################### layer 2 --== conv
        w_p = w2[:, perm_ind[0], :, :]
        w2_p = w_p[perm_ind[1], :, :, :]
        b2_p = b2[perm_ind[1]]
        ##################################### layer 3 --== conv
        w_p = w3[:, perm_ind[1], :, :]
        w3_p = w_p[perm_ind[2], :, :, :]
        b3_p = b3[perm_ind[2]]
        ##################################### layer 4 --== conv
        w_p = w4[:, perm_ind[2], :, :]
        w4_p = w_p[perm_ind[3], :, :, :]
        b4_p = b4[perm_ind[3]]
        ##################################### layer 5--== conv
        w_p = w5[:, perm_ind[3], :, :]
        w5_p = w_p[perm_ind[4], :, :, :]
        b5_p = b5[perm_ind[4]]
        ##################################### layer 6 --== conv
        w_p = w6[:, perm_ind[4], :, :]
        w6_p = w_p[perm_ind[5], :, :, :]
        b6_p = b6[perm_ind[5]]
        ##################################### layer 7 ===== linear
        w7_p = w7[:, perm_ind[5]]
        b7_p = b7

        model = s_conv_6layer(nchannels, args.width, nclasses)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.state_dict()["conv1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["conv1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["conv2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["conv2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["conv3.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["conv3.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["conv4.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["conv4.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["conv5.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["conv5.bias"][:] = torch.Tensor(b5_p)
        model.state_dict()["conv6.weight"][:] = torch.Tensor(w6_p)
        model.state_dict()["conv6.bias"][:] = torch.Tensor(b6_p)

        model.state_dict()["classifier.weight"][:] = torch.Tensor(w7_p)
        model.state_dict()["classifier.bias"][:] = torch.Tensor(b7_p)
        # Shallow conv 8 layer
    elif arch == 's_conv' and len(w_2) == 18:
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]
        w4 = w_2[6]
        b4 = w_2[7]
        w5 = w_2[8]
        b5 = w_2[9]
        w6 = w_2[10]
        b6 = w_2[11]
        w7 = w_2[12]
        b7 = w_2[13]
        w8 = w_2[14]
        b8 = w_2[15]
        w9 = w_2[16]
        b9 = w_2[17]

        w1_p = w1[perm_ind[0], :, :, :]
        b1_p = b1[perm_ind[0]]
        ##################################### layer 2 --== conv
        w_p = w2[:, perm_ind[0], :, :]
        w2_p = w_p[perm_ind[1], :, :, :]
        b2_p = b2[perm_ind[1]]
        ##################################### layer 3 --== conv
        w_p = w3[:, perm_ind[1], :, :]
        w3_p = w_p[perm_ind[2], :, :, :]
        b3_p = b3[perm_ind[2]]
        ##################################### layer 4 --== conv
        w_p = w4[:, perm_ind[2], :, :]
        w4_p = w_p[perm_ind[3], :, :, :]
        b4_p = b4[perm_ind[3]]
        ##################################### layer 5--== conv
        w_p = w5[:, perm_ind[3], :, :]
        w5_p = w_p[perm_ind[4], :, :, :]
        b5_p = b5[perm_ind[4]]
        ##################################### layer 6 --== conv
        w_p = w6[:, perm_ind[4], :, :]
        w6_p = w_p[perm_ind[5], :, :, :]
        b6_p = b6[perm_ind[5]]
        ##################################### layer 7 --== conv
        w_p = w7[:, perm_ind[5], :, :]
        w7_p = w_p[perm_ind[6], :, :, :]
        b7_p = b7[perm_ind[6]]
        ##################################### layer 8 --== conv
        w_p = w8[:, perm_ind[6], :, :]
        w8_p = w_p[perm_ind[7], :, :, :]
        b8_p = b8[perm_ind[7]]
        ##################################### layer 9 ===== linear
        w9_p = w9[:, perm_ind[7]]
        b9_p = b9

        model = s_conv_8layer(nchannels, args.width, nclasses)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.state_dict()["conv1.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["conv1.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["conv2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["conv2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["conv3.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["conv3.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["conv4.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["conv4.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["conv5.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["conv5.bias"][:] = torch.Tensor(b5_p)
        model.state_dict()["conv6.weight"][:] = torch.Tensor(w6_p)
        model.state_dict()["conv6.bias"][:] = torch.Tensor(b6_p)
        model.state_dict()["conv7.weight"][:] = torch.Tensor(w7_p)
        model.state_dict()["conv7.bias"][:] = torch.Tensor(b7_p)
        model.state_dict()["conv8.weight"][:] = torch.Tensor(w8_p)
        model.state_dict()["conv8.bias"][:] = torch.Tensor(b8_p)

        model.state_dict()["classifier.weight"][:] = torch.Tensor(w9_p)
        model.state_dict()["classifier.bias"][:] = torch.Tensor(b9_p)

    # VGG
    elif arch == 'vgg16_bn':

        idx1 = perm_ind[0]
        key = 'features.0.weight'
        param = sd[key]

        # print("before", model.state_dict()[key][0:2])
        # print("before", sd[key][0:2])

        w_p = param[idx1, :, :, :]
        sd[key][:] = w_p
        # print("after", sd[key][:2])

        key = 'features.0.bias'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'features.1.weight'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'features.1.bias'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'features.1.running_mean'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'features.1.running_var'
        param = sd[key]


        w_p = param[idx1]
        sd[key][:] = w_p
        # ##################################### layer 2
        idx2 = perm_ind[1]
        key = 'features.3.weight'
        param = sd[key]


        w_p = param[:, idx1, :, :]
        w_p = w_p[idx2, :, :, :]
        sd[key][:] = w_p

        key = 'features.3.bias'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.4.weight'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.4.bias'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.4.running_mean'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.4.running_var'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p
        ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'features.7.weight'
        param = sd[key]


        w_p = param[:, idx2, :, :]
        w_p = w_p[idx3, :, :, :]
        sd[key][:] = w_p

        key = 'features.7.bias'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.8.weight'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.8.bias'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.8.running_mean'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.8.running_var'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p
        ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'features.10.weight'
        param = sd[key]


        w_p = param[:, idx3, :, :]
        w_p = w_p[idx4, :, :, :]
        sd[key][:] = w_p

        key = 'features.10.bias'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.11.weight'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.11.bias'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.11.running_mean'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.11.running_var'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p
        ##################################### layer 5
        idx5 = perm_ind[4]
        key = 'features.14.weight'
        param = sd[key]


        w_p = param[:, idx4, :, :]
        w_p = w_p[idx5, :, :, :]
        sd[key][:] = w_p

        key = 'features.14.bias'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.15.weight'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.15.bias'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.15.running_mean'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.15.running_var'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p
        ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'features.17.weight'
        param = sd[key]


        w_p = param[:, idx5, :, :]
        w_p = w_p[idx6, :, :, :]
        sd[key][:] = w_p

        key = 'features.17.bias'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.18.weight'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.18.bias'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.18.running_mean'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.18.running_var'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p
        ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'features.20.weight'
        param = sd[key]


        w_p = param[:, idx6, :, :]
        w_p = w_p[idx7, :, :, :]
        sd[key][:] = w_p

        key = 'features.20.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.21.weight'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.21.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.21.running_mean'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.21.running_var'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p
        ##################################### layer 8
        idx8 = perm_ind[7]
        key = 'features.24.weight'
        param = sd[key]


        w_p = param[:, idx7, :, :]
        w_p = w_p[idx8, :, :, :]
        sd[key][:] = w_p

        key = 'features.24.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.25.weight'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.25.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.25.running_mean'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.25.running_var'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p
        ##################################### layer 9
        idx9 = perm_ind[8]
        key = 'features.27.weight'
        param = sd[key]


        w_p = param[:, idx8, :, :]
        w_p = w_p[idx9, :, :, :]
        sd[key][:] = w_p

        key = 'features.27.bias'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.28.weight'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.28.bias'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.28.running_mean'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.28.running_var'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p
        ##################################### layer 10
        idx10 = perm_ind[9]
        key = 'features.30.weight'
        param = sd[key]


        w_p = param[:, idx9, :, :]
        w_p = w_p[idx10, :, :, :]
        sd[key][:] = w_p

        key = 'features.30.bias'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.31.weight'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.31.bias'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.31.running_mean'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.31.running_var'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p
        ##################################### layer 11
        idx11 = perm_ind[10]
        key = 'features.34.weight'
        param = sd[key]


        w_p = param[:, idx10, :, :]
        w_p = w_p[idx11, :, :, :]
        sd[key][:] = w_p

        key = 'features.34.bias'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.35.weight'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.35.bias'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.35.running_mean'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.35.running_var'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p
        ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'features.37.weight'
        param = sd[key]


        w_p = param[:, idx11, :, :]
        w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        key = 'features.37.bias'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.38.weight'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.38.bias'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.38.running_mean'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.38.running_var'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p
        ##################################### layer 13
        idx13 = perm_ind[12]
        key = 'features.40.weight'
        param = sd[key]


        w_p = param[:, idx12, :, :]
        w_p = w_p[idx13, :, :, :]
        sd[key][:] = w_p

        key = 'features.40.bias'
        param = sd[key]


        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'features.41.weight'
        param = sd[key]


        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'features.41.bias'
        param = sd[key]


        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'features.41.running_mean'
        param = sd[key]


        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'features.41.running_var'
        param = sd[key]


        w_p = param[idx13]
        sd[key][:] = w_p
        ##################################### layer 14 ===== linear
        idx14 = perm_ind[13]
        key = 'classifier.1.weight'
        param = sd[key]


        w_p = param[:, idx13]
        w_p = w_p[idx14, :]
        sd[key][:] = w_p

        key = 'classifier.1.bias'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p
        ##################################### layer 15 ===== linear
        idx15 = perm_ind[14]
        key = 'classifier.4.weight'
        param = sd[key]


        w_p = param[:, idx14]
        w_p = w_p[idx15, :]
        sd[key][:] = w_p

        key = 'classifier.4.bias'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p
        # ##################################### layer 16 ===== linear
        key = 'classifier.6.weight'
        param = sd[key]


        w_p = param[:, idx15]
        # w_p = w_p[idx16, :]
        sd[key][:] = w_p

        key = 'classifier.6.bias'
        param = sd[key]


        # w_p = param   ############################## no change
        # sd[key][:] = w_p

        model = vgg.__dict__[args.arch](nclasses)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    elif arch == 'resnet18':
        # ##################################### layer 1
        idx1 = perm_ind[0]
        key = 'conv1.weight'  # [64, 64, 3, 3]
        param = sd[key]

        w_p = param[idx1, :, :, :]
        sd[key][:] = w_p

        key = 'bn1.weight'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'bn1.bias'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'bn1.running_mean'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p

        key = 'bn1.running_var'
        param = sd[key]

        w_p = param[idx1]
        sd[key][:] = w_p
        # ##################################### layer 2
        idx2 = perm_ind[1]
        key = 'layer1.0.conv1.weight'
        param = sd[key]


        w_p = param[:, idx1, :, :]
        w_p = w_p[idx2, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.0.bn1.weight'
        param = sd[key]

        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'layer1.0.bn1.bias'
        param = sd[key]

        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'layer1.0.bn1.running_mean'
        param = sd[key]

        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'layer1.0.bn1.running_var'
        param = sd[key]

        w_p = param[idx2]
        sd[key][:] = w_p
        ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'layer1.0.conv2.weight'
        param = sd[key]


        w_p = param[:, idx2, :, :]
        w_p = w_p[idx3, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.0.bn2.weight'
        param = sd[key]

        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'layer1.0.bn2.bias'
        param = sd[key]

        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'layer1.0.bn2.running_mean'
        param = sd[key]

        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'layer1.0.bn2.running_var'
        param = sd[key]

        w_p = param[idx3]
        sd[key][:] = w_p
        ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'layer1.1.conv1.weight'
        param = sd[key]

        w_p = param[:, idx3, :, :]
        w_p = w_p[idx4, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.weight'
        param = sd[key]

        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.bias'
        param = sd[key]

        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx4]
        sd[key][:] = w_p
        ##################################### layer 5
        idx5 = perm_ind[4]
        key = 'layer1.1.conv2.weight'
        param = sd[key]

        w_p = param[:, idx4, :, :]
        w_p = w_p[idx5, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.weight'
        param = sd[key]

        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.bias'
        param = sd[key]

        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.running_mean'
        param = sd[key]

        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.running_var'
        param = sd[key]

        w_p = param[idx5]
        sd[key][:] = w_p
        ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'layer2.0.conv1.weight'
        param = sd[key]

        w_p = param[:, idx5, :, :]
        w_p = w_p[idx6, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.weight'
        param = sd[key]

        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.bias'
        param = sd[key]

        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_mean'
        param = sd[key]

        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_var'
        param = sd[key]

        w_p = param[idx6]
        sd[key][:] = w_p
        ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'layer2.0.conv2.weight'
        param = sd[key]

        w_p = param[:, idx6, :, :]
        w_p = w_p[idx7, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.weight'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.bias'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.running_mean'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.running_var'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p
        ##################################### layer 8
        idx8 = perm_ind[7]
        key = 'layer2.0.shortcut.0.weight'
        param = sd[key]

        # print(param.shape) ## (128, 64, 1, 1)
        # w_p = param[:, idx7, :, :] ###
        # print(len(idx5), len(idx6)) ## 64,128
        w_p = param[:, idx5, :, :]  ## layer2.0.conv1.weight
        w_p = w_p[idx7, :, :, :]  #### layer2.0.conv2.weight
        sd[key][:] = w_p

        key = 'layer2.0.shortcut.1.weight'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer2.0.shortcut.1.bias'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer2.0.shortcut.1.running_mean'
        param = sd[key]

        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer2.0.shortcut.1.running_var'
        param = sd[key]

        w_p = param[idx7]
        sd[key][:] = w_p
        ##################################### layer 9
        idx9 = perm_ind[8]
        key = 'layer2.1.conv1.weight'
        param = sd[key]

        w_p = param[:, idx7, :, :]
        w_p = w_p[idx9, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.weight'
        param = sd[key]

        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.bias'
        param = sd[key]

        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx9]
        sd[key][:] = w_p
        ##################################### layer 10
        idx10 = perm_ind[9]
        key = 'layer2.1.conv2.weight'
        param = sd[key]

        w_p = param[:, idx9, :, :]
        w_p = w_p[idx10, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.weight'
        param = sd[key]

        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.bias'
        param = sd[key]

        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.running_mean'
        param = sd[key]

        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.running_var'
        param = sd[key]

        w_p = param[idx10]
        sd[key][:] = w_p
        ##################################### layer 11
        idx11 = perm_ind[10]
        key = 'layer3.0.conv1.weight'
        param = sd[key]

        w_p = param[:, idx10, :, :]
        w_p = w_p[idx11, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.weight'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.bias'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_mean'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_var'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p
        ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'layer3.0.conv2.weight'
        param = sd[key]

        w_p = param[:, idx11, :, :]
        w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.weight'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.bias'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.running_mean'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.running_var'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p
        ##################################### layer 13 ===================== shortcut
        idx13 = perm_ind[12]
        key = 'layer3.0.shortcut.0.weight'
        param = sd[key]

        # w_p = param[:, idx12, :, :]
        w_p = param[:, idx10, :, :]  ## layer3.0.conv1.weight
        w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.shortcut.1.weight'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.shortcut.1.bias'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.shortcut.1.running_mean'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer3.0.shortcut.1.running_var'
        param = sd[key]

        w_p = param[idx12]
        sd[key][:] = w_p
        ##################################### layer 14
        idx14 = perm_ind[13]
        key = 'layer3.1.conv1.weight'
        param = sd[key]

        w_p = param[:, idx12, :, :]
        w_p = w_p[idx14, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.weight'
        param = sd[key]

        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.bias'
        param = sd[key]

        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx14]
        sd[key][:] = w_p
        ##################################### layer 15
        idx15 = perm_ind[14]
        key = 'layer3.1.conv2.weight'
        param = sd[key]

        w_p = param[:, idx14, :, :]
        w_p = w_p[idx15, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.weight'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.bias'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.running_mean'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.running_var'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p
        ##################################### layer 16
        idx16 = perm_ind[15]
        key = 'layer4.0.conv1.weight'
        param = sd[key]

        w_p = param[:, idx15, :, :]
        w_p = w_p[idx16, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.weight'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.bias'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_mean'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_var'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p
        ##################################### layer 17
        idx17 = perm_ind[16]
        key = 'layer4.0.conv2.weight'
        param = sd[key]

        w_p = param[:, idx16, :, :]
        w_p = w_p[idx17, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.weight'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.bias'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.running_mean'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.running_var'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p
        ##################################### layer 18
        idx18 = perm_ind[17]
        key = 'layer4.0.shortcut.0.weight'
        param = sd[key]

        # w_p = param[:, idx17, :, :]
        w_p = param[:, idx15, :, :]  ### layer4.0.conv1.weight
        w_p = w_p[idx17, :, :, :]  ### layer4.0.conv2.weight
        sd[key][:] = w_p

        key = 'layer4.0.shortcut.1.weight'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.shortcut.1.bias'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.shortcut.1.running_mean'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer4.0.shortcut.1.running_var'
        param = sd[key]

        w_p = param[idx17]
        sd[key][:] = w_p
        ##################################### layer 19
        idx19 = perm_ind[18]
        key = 'layer4.1.conv1.weight'
        param = sd[key]

        w_p = param[:, idx17, :, :]
        w_p = w_p[idx19, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.weight'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.bias'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p
        ##################################### layer 20
        idx20 = perm_ind[19]
        key = 'layer4.1.conv2.weight'
        param = sd[key]

        w_p = param[:, idx19, :, :]
        w_p = w_p[idx20, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.weight'
        param = sd[key]

        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.bias'
        param = sd[key]

        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.running_mean'
        param = sd[key]

        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.running_var'
        param = sd[key]

        w_p = param[idx20]
        sd[key][:] = w_p
        # ##################################### layer 21 ===== linear
        key = 'linear.weight'
        param = sd[key]

        w_p = param[:, idx20]
        # w_p = w_p[idx16, :]
        sd[key][:] = w_p

        key = 'linear.bias'
        param = sd[key]

        # w_p = param   ############################## no change
        # sd[key][:] = w_p

    return model.state_dict()






def barrier_SA(arch, model, sd2, w2, init_state, tmax, tmin, steps, train_inputs, train_targets, nchannels, nclasses, nunits):
    ################################################### Simulated Annealing
    class BarrierCalculationProblem(Annealer):
        """Test annealer with a travelling salesman problem.
        """

        # pass extra data (the distance matrix) into the constructor
        def __init__(self, state):
            super(BarrierCalculationProblem, self).__init__(state)  # important!

        def move(self):
            """Swaps two cities in the route."""
            # no efficiency gain, just proof of concept
            # demonstrates returning the delta energy (optional)
            initial_energy = self.energy()
            for j in range(5):
                for i in range(len(self.state[j])):
                    x = self.state[j][i]
                    a = random.randint(0, len(x) - 1)
                    b = random.randint(0, len(x) - 1)
                    self.state[j][i][a], self.state[j][i][b] = self.state[j][i][b], self.state[j][i][a]
            return self.energy() - initial_energy

        def energy(self):
            """Calculates the length of the route."""
            permuted_models_sd = []
            for i in range(5):
                permuted_models_sd.append(permute(arch, model, self.state[i], sd2[i], w2[i], nchannels, nclasses, nunits))
            ###### LMC between permuted models
            pairs = list(itertools.combinations(range(5), 2))
            barriers = []
            for x in pairs:
                idx1 = x[0]
                idx2 = x[1]
                sd1_ = permuted_models_sd[idx1]
                sd2_ = permuted_models_sd[idx2]
                weights = np.linspace(0, 1, 11)
                results = []
                for i in range(len(weights)):
                    model.load_state_dict(interpolate_state_dicts(sd1_, sd2_, weights[i]))
                    results.append(evaluate_model(model, train_inputs, train_targets)['top1'])
                p1_eval = results[0]
                p2_eval = results[-1]
                result_avg_models = (p1_eval + p2_eval) / 2
                barriers.append(result_avg_models - results[5])
                # print(p1_eval, p2_eval, result_avg_models - results[4])
            #######
            cost = statistics.mean(barriers)  ### mean over 20*19/2 or 5*4/2 barriers
            return cost
    ############################# start the process
    bcp = BarrierCalculationProblem(init_state)
    # bcp.set_schedule(bcp.auto(minutes=0.2))


    bcp_auto = {'tmax': tmax, 'tmin': tmin, 'steps': steps, 'updates': 100}
    bcp.set_schedule(bcp_auto)
    # print('tmax', 'tmin', 'steps', 'updates', bcp_auto['tmax'], bcp_auto['tmin'], bcp_auto['steps'], bcp_auto['updates'])

    winning_state, e = bcp.anneal()
    print()
    print("best", winning_state)
    print("%0.4f least barrier:" % e)
    print("--- %s seconds ---" % (time.time() - start_time))

    return winning_state

if __name__ == '__main__':
    main()