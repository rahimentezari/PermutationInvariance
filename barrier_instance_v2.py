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
    # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/original/{args.seed}/{source_file_name}'
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
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Train_target_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_input_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_target_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
        else:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST3d_Train_input_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Train_target_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_input_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_target_org.pkl'
            destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
    elif (args.dataset == 'CIFAR10'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'SVHN'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'CIFAR100'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_input_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_target_org.pkl'
        destination_blob_name = f'Neurips21_Arxiv/dataset_pkl/{source_file_name}'
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
    # ##################################################### LMC original models
    # ############################# load selected models
    # sd = []
    # # for j in [7,8,9,15,16]:
    # for j in [20,11,12,13,19]:
    #     bucket_name = 'permutation-mlp'
    #     destination_blob_name = 'model_best.th'
    #     source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{j}/{destination_blob_name}'
    #     download_blob(bucket_name, source_file_name, destination_blob_name)
    # 
    #     checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # 
    #     def key_transformation(old_key):
    #         if 'module' in old_key:
    #             return old_key[7:]
    #         return old_key
    # 
    #     new_state_dict = OrderedDict()
    #     for key, value in checkpoint.items():
    #         new_key = key_transformation(key)
    #         new_state_dict[new_key] = value
    #     checkpoint = new_state_dict
    # 
    #     sd.append(checkpoint)
    # 
    # 
    # 
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
    # pairs = list(itertools.combinations(range(5), 2))
    # pair = 0
    # barrier_test_basin_before = []
    # for x in pairs:
    #     pair = pair + 1
    #     idx1 = x[0]
    #     idx2 = x[1]
    #     sd1_ = sd[idx1]
    #     sd2_ = sd[idx2]
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
    #     # source_file_name = f'dict_before_{pair}.pkl'
    #     source_file_name = f'dict_before_{pair}.pkl'
    #     # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/auto/{source_file_name}'
    #     destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA_InstanceOptimized_v2/original/{source_file_name}'
    #     pickle_out = pickle.dumps(dict_after)
    #     upload_pkl(bucket_name, pickle_out, destination_blob_name)
    # print()
    # print("basin_mean_before", statistics.mean(barrier_test_basin_before))
    # print("basin_std_before", statistics.stdev(barrier_test_basin_before))
    # 
    # 
    # 



    ########################################## oracle barrier
    bucket_name = 'permutation-mlp'
    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{21}/{destination_blob_name}'
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
    sd1 = checkpoint

    w1 = []
    for key in sd1.keys():
        param = sd1[key]
        w1.append(param.cpu().detach().numpy())
    # create permutation list for mlp
    if args.arch == 'mlp':
        len_perm = []
        for i in range(int(len(w1) / 2 - 1)):
            len_perm.append(args.width)
    # create permutation list for conv nets
    conv_arch = False
    for key in sd1:
        print(key, sd1[key].shape)
        if "conv" in key or "running_mean" in key:
            conv_arch = True

    if conv_arch:
        params = []
        len_perm = []
        for key in sd1.keys():
            param = model.state_dict()[key]
            if "num_batches_tracked" not in key:
                params.append(param.cpu().detach().numpy())
                if len(param.shape) == 4:
                    len_perm.append(param.shape[0])
                if len(param.shape) == 2:
                    len_perm.append(param.shape[0])

    print("len_perm", len(len_perm))
    print("len_perm", len_perm)


    init_states = []
    for i in range(1,6):
        random_permuted_index = []
        for z in len_perm:
            lst = [y for y in range(z)]
            random.seed(i)
            rnd = random.sample(lst, z)
            random_permuted_index.append(rnd)
        init_states.append(random_permuted_index)


    # print(sd1["features.0.weight"][0:2])
    permuted_oracle_sds = []
    for i in range(5):
        permuted_oracle_sds.append(permute(args.arch, model, init_states[i], sd1, w1, nchannels, nclasses, args.width))
    # print(permuted_oracle_sd1["features.0.weight"][0:2])


    # # #### sanity check if permutation is done properly: L2 Gaussian Noise
    # # ##################################################
    # # # hooks = {}
    # # # for name, module in model.named_modules():
    # # #     hooks[name] = module.register_forward_hook(self, hook_fn)
    # #
    # #
    # # activation = {}
    # # def get_activation(name):
    # #     def hook(model, input, output):
    # #         activation[name] = output.detach()
    # #
    # #     return hook
    # #
    # # device = torch.device('cuda')
    # # torch.manual_seed(1)
    # # input_g = torch.randn(256, 1, 32, 32)
    # # input_g = input_g.to(device)
    # # # input_g = input_g.to(device).view(input_g.size(0), -1)
    # # ######################### to model1
    # # model.load_state_dict(sd1)
    # # model.register_forward_hook(get_activation('layer4.1.bn2'))
    # # output = model(input_g)
    # # print(activation['layer4.1.bn2'].shape)
    # # print(torch.transpose(sd1['linear.weight'], 0, 1).shape)
    # # gaussian_out1 = torch.matmul(activation['layer4.1.bn2'], torch.transpose(sd1['linear.weight'], 0, 1))
    # # ######################### to model2
    # # model.load_state_dict(permuted_oracle_sd)
    # # model.register_forward_hook(get_activation(['layer4.1.bn2']))
    # # output = model(input_g)
    # # gaussian_out2 = torch.matmul(activation['layer4.1.bn2'], torch.transpose(sd1['linear.weight'], 0, 1))
    # #
    # # dist = np.linalg.norm(gaussian_out1.cpu() - gaussian_out2.cpu())
    # # print(f"L2 noise:", dist)
    # # print('{0:4f}'.format(dist))


    ##################################################
    bucket_name = 'permutation-mlp'
    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{21}/{destination_blob_name}'
    download_blob(bucket_name, source_file_name, destination_blob_name)

    checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))

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
    # for i in range(5):
    #
    #     dict_oracle = get_barrier(model, sd1, permuted_oracle_sds[i], train_inputs, train_targets, test_inputs, test_targets)
    #     barrier_test = dict_oracle['barrier_test']
    #     lmc_test = dict_oracle['test_lmc']
    #
    #     print("barrier_test_oracle", barrier_test)
    #     print("lmc_test_oracle", lmc_test)
    #
    #     source_file_name = f'dict_oracle_{i}.pkl'
    #     destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA_InstanceOptimized_v2/oracle/before/{source_file_name}'
    #     pickle_out = pickle.dumps(dict_oracle)
    #     upload_pkl(bucket_name, pickle_out, destination_blob_name)


    pairs = list(itertools.combinations(range(5), 2))
    pair = 0
    barrier_test_basin_before = []
    for x in pairs:
        pair = pair + 1
        idx1 = x[0]
        idx2 = x[1]
        sd1_ = permuted_oracle_sds[idx1]
        sd2_ = permuted_oracle_sds[idx2]
        dict_after = get_barrier(model, sd1_, sd2_, train_inputs, train_targets, test_inputs, test_targets)


        barrier_test = dict_after['barrier_test']
        lmc_test = dict_after['test_lmc']

        print("barrier_test_pairwise_original", barrier_test)
        print("lmc_test_pairwise_original", lmc_test)
        barrier_test_basin_before.append(barrier_test[0])

        source_file_name = f'dict_before_{pair}.pkl'
        # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/auto/{source_file_name}'
        destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA_InstanceOptimized_v2/oracle/before/{source_file_name}'
        pickle_out = pickle.dumps(dict_after)
        upload_pkl(bucket_name, pickle_out, destination_blob_name)
    print()
    print("basin_mean_after", statistics.mean(barrier_test_basin_before))
    print("basin_std_after", statistics.stdev(barrier_test_basin_before))














    # ########################################## SA oracle: model1 and permuted model1
    sd = permuted_oracle_sds

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
            random.seed(i)
            rnd = random.sample(lst, z)
            random_permuted_index.append(rnd)
        init_state.append(random_permuted_index)

    exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    # winning_permutation = barrier_SA(args.arch, model, sd, w, init_state,
    #                                  args.tmax, args.tmin, args.steps,
    #                                  train_inputs, train_targets,
    #                                  nchannels, nclasses, args.width)
    # print("winning_permutation", winning_permutation)
    winning_permutation = [[[790, 6, 888, 690, 902, 483, 521, 29, 355, 669, 539, 250, 149, 153, 554, 134, 694, 127, 782, 704, 621, 575, 589, 972, 915, 665, 754, 326, 569, 597, 793, 21, 339, 363, 251, 30, 267, 845, 66, 172, 512, 237, 609, 615, 294, 771, 764, 139, 101, 852, 283, 979, 86, 869, 74, 568, 994, 10, 837, 730, 416, 733, 971, 616, 707, 42, 602, 362, 928, 11, 706, 999, 797, 124, 598, 322, 4, 641, 276, 601, 423, 438, 25, 970, 176, 824, 504, 749, 715, 908, 892, 435, 161, 321, 43, 611, 533, 921, 604, 862, 7, 841, 590, 226, 713, 457, 130, 335, 98, 798, 712, 767, 345, 714, 523, 475, 22, 657, 717, 631, 431, 788, 214, 651, 432, 379, 955, 808, 390, 1012, 668, 666, 65, 803, 670, 224, 998, 417, 770, 957, 917, 622, 952, 392, 259, 813, 948, 120, 610, 996, 147, 873, 667, 593, 618, 975, 395, 78, 372, 463, 371, 925, 351, 442, 831, 274, 495, 158, 909, 865, 227, 655, 686, 1007, 580, 524, 415, 1022, 927, 856, 382, 962, 264, 638, 292, 95, 838, 702, 900, 626, 849, 827, 558, 50, 473, 783, 213, 924, 935, 779, 253, 988, 357, 394, 503, 288, 28, 653, 393, 211, 891, 419, 502, 736, 907, 954, 123, 801, 561, 640, 981, 572, 156, 248, 44, 77, 969, 32, 140, 412, 679, 584, 258, 406, 851, 537, 496, 374, 441, 143, 404, 807, 150, 886, 929, 206, 275, 333, 961, 995, 207, 48, 38, 500, 1018, 155, 821, 466, 967, 795, 529, 299, 94, 600, 232, 692, 805, 103, 18, 619, 847, 265, 931, 202, 446, 112, 108, 421, 60, 332, 560, 828, 280, 540, 376, 650, 701, 675, 315, 35, 309, 732, 526, 450, 871, 51, 1001, 899, 642, 298, 257, 9, 272, 238, 693, 990, 1023, 47, 448, 434, 225, 507, 517, 956, 425, 380, 69, 373, 254, 145, 91, 284, 12, 894, 898, 963, 198, 26, 131, 159, 99, 452, 162, 586, 624, 56, 314, 950, 76, 472, 312, 858, 784, 68, 460, 993, 368, 881, 740, 906, 633, 245, 860, 816, 24, 748, 747, 829, 645, 183, 658, 358, 13, 281, 810, 868, 984, 493, 991, 367, 334, 468, 116, 938, 599, 109, 607, 490, 205, 73, 413, 509, 486, 166, 215, 617, 286, 128, 105, 273, 458, 976, 741, 820, 49, 1002, 933, 557, 709, 306, 461, 661, 46, 781, 738, 411, 719, 884, 173, 551, 647, 522, 5, 235, 308, 687, 875, 939, 23, 684, 762, 834, 87, 365, 348, 37, 239, 381, 3, 777, 877, 277, 45, 542, 467, 854, 64, 765, 488, 773, 252, 489, 566, 832, 71, 316, 644, 241, 664, 200, 97, 587, 188, 855, 146, 157, 895, 632, 613, 880, 531, 233, 88, 913, 323, 974, 893, 451, 151, 219, 407, 174, 244, 289, 167, 453, 859, 536, 1014, 170, 165, 266, 424, 223, 437, 58, 758, 113, 2, 698, 595, 347, 182, 427, 819, 282, 746, 800, 397, 739, 296, 627, 591, 178, 84, 978, 556, 983, 946, 549, 356, 637, 132, 387, 385, 905, 209, 449, 612, 342, 96, 689, 117, 648, 722, 745, 154, 320, 691, 582, 61, 897, 734, 179, 391, 15, 840, 922, 72, 344, 923, 420, 8, 400, 455, 562, 125, 699, 960, 304, 389, 67, 506, 54, 588, 216, 761, 676, 465, 279, 656, 189, 680, 763, 115, 550, 89, 93, 197, 685, 484, 809, 148, 485, 408, 270, 936, 383, 164, 623, 853, 194, 317, 439, 787, 980, 755, 110, 135, 469, 677, 137, 843, 171, 520, 769, 222, 941, 751, 581, 634, 311, 511, 785, 230, 319, 879, 1016, 791, 538, 175, 129, 772, 789, 260, 643, 471, 863, 930, 654, 398, 85, 278, 36, 592, 40, 565, 786, 725, 681, 727, 545, 19, 519, 303, 34, 80, 628, 295, 0, 59, 530, 100, 388, 760, 901, 459, 535, 331, 514, 133, 603, 564, 910, 247, 480, 92, 639, 318, 399, 986, 199, 608, 731, 1010, 516, 350, 329, 726, 912, 1017, 552, 953, 106, 659, 1006, 287, 218, 943, 660, 192, 703, 195, 193, 625, 291, 1015, 328, 579, 876, 81, 585, 1019, 263, 547, 559, 866, 422, 835, 377, 649, 119, 191, 190, 721, 285, 505, 532, 228, 17, 776, 338, 883, 142, 336, 187, 989, 217, 774, 711, 1, 844, 878, 728, 945, 775, 114, 262, 138, 208, 477, 428, 861, 752, 527, 186, 708, 482, 887, 341, 964, 555, 958, 544, 848, 919, 850, 79, 63, 102, 973, 474, 297, 890, 401, 204, 799, 872, 630, 491, 944, 41, 916, 857, 361, 301, 126, 57, 812, 1020, 418, 107, 804, 210, 982, 543, 447, 62, 696, 144, 325, 903, 152, 987, 992, 443, 750, 515, 456, 744, 977, 243, 574, 570, 169, 55, 271, 683, 52, 501, 911, 662, 735, 1004, 177, 926, 942, 352, 269, 780, 759, 563, 386, 737, 635, 364, 305, 180, 83, 498, 814, 815, 478, 578, 846, 937, 366, 663, 369, 307, 343, 968, 513, 360, 966, 470, 614, 220, 573, 444, 768, 27, 221, 606, 430, 528, 111, 596, 716, 242, 766, 914, 743, 384, 31, 710, 723, 346, 359, 302, 290, 185, 818, 822, 327, 268, 918, 14, 1005, 594, 508, 234, 1011, 479, 396, 433, 1000, 122, 636, 874, 778, 403, 673, 1008, 476, 965, 947, 525, 497, 414, 718, 753, 959, 870, 261, 293, 700, 53, 33, 688, 429, 567, 932, 246, 817, 1009, 510, 576, 212, 402, 724, 830, 823, 1003, 184, 481, 82, 920, 349, 646, 695, 136, 896, 405, 445, 141, 577, 240, 904, 370, 794, 518, 1013, 825, 160, 885, 729, 163, 756, 534, 934, 440, 826, 951, 340, 426, 462, 20, 806, 674, 249, 499, 454, 742, 867, 196, 492, 310, 464, 652, 410, 546, 682, 231, 864, 678, 353, 330, 324, 792, 118, 836, 940, 181, 436, 409, 16, 842, 121, 889, 997, 605, 583, 236, 697, 671, 839, 705, 833, 378, 796, 375, 313, 1021, 337, 203, 39, 201, 620, 90, 949, 811, 757, 720, 70, 548, 629, 541, 487, 882, 985, 104, 75, 255, 672, 300, 256, 571, 168, 229, 553, 354, 802, 494], [474, 4, 742, 922, 957, 610, 951, 560, 879, 416, 650, 686, 979, 786, 493, 183, 612, 377, 450, 267, 991, 649, 1003, 569, 860, 69, 614, 448, 868, 561, 691, 278, 73, 317, 892, 1020, 443, 295, 481, 1014, 68, 1011, 915, 710, 132, 49, 804, 1007, 726, 425, 434, 249, 659, 522, 121, 665, 718, 643, 1005, 95, 77, 335, 604, 86, 405, 609, 520, 200, 241, 824, 689, 162, 586, 177, 507, 954, 343, 767, 591, 123, 702, 647, 259, 876, 576, 822, 339, 999, 457, 540, 1016, 635, 1, 1004, 973, 216, 47, 403, 859, 509, 120, 411, 626, 158, 355, 722, 134, 766, 34, 547, 88, 408, 867, 143, 150, 308, 269, 980, 89, 599, 163, 470, 592, 685, 730, 344, 187, 182, 301, 486, 548, 618, 653, 106, 524, 658, 39, 808, 454, 213, 942, 499, 775, 1009, 794, 749, 190, 630, 770, 651, 356, 673, 174, 85, 209, 198, 629, 440, 525, 468, 58, 833, 188, 895, 974, 995, 307, 842, 20, 529, 858, 953, 370, 840, 738, 145, 251, 809, 116, 491, 255, 312, 300, 352, 414, 224, 988, 153, 197, 672, 309, 466, 325, 337, 331, 697, 834, 400, 780, 950, 96, 907, 387, 943, 558, 394, 674, 914, 961, 930, 701, 54, 489, 154, 384, 577, 76, 602, 9, 754, 664, 578, 36, 2, 412, 11, 0, 148, 532, 917, 921, 755, 211, 340, 193, 579, 777, 952, 550, 937, 1018, 600, 296, 30, 556, 1021, 945, 166, 631, 361, 483, 891, 706, 657, 451, 641, 581, 265, 272, 258, 681, 83, 81, 546, 221, 713, 965, 603, 27, 880, 165, 500, 985, 984, 987, 611, 795, 91, 23, 981, 906, 725, 584, 607, 38, 349, 633, 553, 852, 185, 329, 179, 62, 946, 862, 903, 872, 189, 52, 293, 1022, 149, 479, 105, 232, 225, 80, 439, 941, 531, 472, 627, 772, 444, 292, 392, 568, 71, 141, 281, 498, 934, 427, 623, 17, 866, 536, 330, 835, 983, 715, 462, 908, 541, 637, 254, 243, 818, 617, 368, 275, 311, 391, 341, 459, 72, 367, 10, 274, 992, 205, 53, 670, 397, 407, 662, 756, 502, 883, 594, 3, 364, 410, 831, 962, 839, 705, 423, 327, 246, 220, 497, 45, 884, 460, 107, 778, 404, 428, 750, 469, 1008, 639, 523, 40, 535, 122, 476, 977, 25, 573, 1001, 1023, 625, 933, 114, 369, 138, 849, 897, 160, 385, 1000, 762, 904, 378, 736, 208, 294, 266, 305, 837, 857, 829, 928, 242, 302, 588, 877, 248, 825, 291, 765, 856, 743, 792, 64, 870, 264, 37, 136, 698, 675, 57, 127, 157, 565, 821, 108, 696, 175, 585, 621, 478, 964, 959, 195, 196, 285, 131, 669, 429, 782, 893, 12, 732, 99, 488, 393, 855, 739, 374, 406, 812, 613, 733, 383, 516, 113, 333, 496, 363, 118, 793, 634, 16, 66, 43, 288, 760, 453, 386, 787, 61, 783, 721, 744, 716, 545, 152, 181, 22, 388, 769, 70, 100, 50, 276, 218, 146, 913, 797, 949, 202, 32, 680, 869, 287, 709, 279, 55, 784, 350, 888, 436, 320, 807, 465, 402, 133, 768, 567, 1010, 487, 566, 284, 865, 441, 998, 757, 800, 597, 303, 533, 601, 542, 357, 511, 828, 60, 555, 564, 59, 5, 640, 372, 605, 728, 986, 101, 863, 644, 692, 381, 129, 432, 632, 461, 887, 939, 191, 583, 233, 467, 947, 989, 845, 968, 184, 924, 562, 655, 455, 918, 125, 747, 419, 620, 660, 112, 997, 297, 324, 44, 283, 596, 969, 848, 574, 135, 789, 729, 261, 544, 103, 65, 619, 395, 186, 401, 477, 527, 761, 735, 996, 836, 119, 970, 746, 433, 446, 290, 323, 240, 615, 803, 687, 92, 684, 752, 806, 194, 473, 14, 26, 538, 751, 853, 990, 885, 593, 495, 365, 318, 919, 526, 115, 201, 447, 652, 435, 724, 850, 796, 46, 923, 56, 622, 805, 819, 173, 140, 636, 228, 426, 815, 688, 322, 727, 326, 656, 711, 63, 1019, 171, 379, 830, 270, 971, 963, 334, 898, 901, 671, 8, 648, 437, 147, 519, 759, 587, 247, 442, 799, 286, 310, 1006, 314, 79, 518, 48, 925, 741, 543, 373, 642, 814, 832, 844, 753, 932, 972, 700, 886, 230, 851, 551, 389, 505, 142, 948, 758, 810, 846, 723, 458, 226, 1002, 494, 1012, 512, 905, 298, 42, 598, 328, 342, 816, 589, 271, 894, 861, 788, 506, 353, 779, 927, 785, 628, 572, 245, 415, 346, 552, 490, 167, 409, 528, 417, 912, 661, 422, 206, 87, 707, 624, 940, 534, 210, 463, 144, 33, 975, 227, 413, 299, 176, 74, 172, 737, 559, 354, 654, 776, 978, 424, 683, 421, 580, 720, 889, 438, 238, 75, 203, 390, 304, 110, 382, 902, 449, 873, 398, 332, 156, 575, 645, 236, 359, 590, 178, 811, 896, 616, 521, 464, 380, 929, 41, 168, 539, 215, 843, 319, 900, 139, 679, 321, 646, 235, 306, 916, 336, 396, 97, 124, 345, 695, 375, 420, 231, 748, 781, 338, 704, 456, 250, 104, 936, 926, 98, 982, 315, 93, 899, 28, 826, 820, 180, 484, 475, 280, 530, 608, 376, 878, 67, 909, 222, 130, 256, 663, 316, 262, 126, 239, 958, 508, 31, 676, 430, 606, 78, 513, 471, 159, 485, 554, 1015, 234, 790, 277, 1013, 514, 935, 15, 841, 802, 773, 557, 169, 993, 976, 910, 7, 931, 956, 273, 967, 109, 703, 452, 257, 920, 771, 348, 29, 854, 731, 192, 102, 51, 237, 517, 693, 798, 1017, 503, 515, 431, 774, 571, 199, 161, 570, 244, 214, 563, 128, 690, 504, 399, 871, 955, 501, 510, 764, 699, 351, 18, 362, 151, 6, 289, 418, 717, 358, 217, 864, 827, 219, 268, 82, 137, 813, 847, 668, 35, 881, 549, 347, 890, 252, 19, 155, 734, 582, 838, 282, 482, 694, 260, 492, 90, 712, 682, 740, 966, 263, 960, 875, 117, 111, 944, 638, 678, 207, 801, 24, 667, 13, 204, 84, 164, 366, 253, 763, 666, 212, 994, 371, 745, 708, 817, 791, 714, 170, 911, 480, 719, 882, 677, 21, 874, 595, 445, 313, 229, 360, 823, 537, 938, 223, 94], [179, 945, 267, 619, 37, 810, 374, 226, 897, 846, 293, 577, 372, 291, 150, 906, 45, 822, 969, 210, 101, 295, 29, 237, 919, 539, 653, 617, 741, 345, 695, 465, 455, 672, 139, 284, 411, 162, 717, 33, 448, 30, 312, 607, 951, 892, 606, 76, 85, 397, 523, 165, 382, 950, 154, 658, 826, 290, 198, 25, 890, 837, 599, 968, 96, 736, 967, 721, 215, 337, 142, 575, 509, 87, 733, 219, 413, 669, 92, 845, 702, 932, 765, 705, 681, 777, 492, 489, 864, 916, 458, 72, 35, 569, 280, 7, 558, 662, 522, 464, 483, 454, 925, 719, 297, 98, 552, 708, 396, 359, 188, 891, 21, 712, 775, 566, 963, 395, 501, 68, 526, 646, 146, 598, 61, 909, 644, 306, 718, 233, 858, 59, 103, 544, 14, 89, 854, 870, 581, 197, 187, 459, 495, 772, 32, 875, 27, 40, 830, 661, 304, 155, 911, 159, 677, 530, 1001, 869, 881, 942, 235, 682, 929, 934, 439, 637, 490, 986, 673, 641, 19, 310, 692, 475, 473, 910, 1021, 230, 425, 807, 632, 973, 184, 592, 255, 1014, 965, 441, 636, 589, 74, 54, 130, 1015, 462, 593, 621, 446, 745, 402, 958, 551, 169, 572, 900, 43, 823, 47, 713, 664, 519, 42, 46, 141, 770, 725, 901, 734, 17, 82, 895, 278, 629, 117, 378, 356, 269, 1010, 276, 58, 245, 587, 239, 921, 894, 962, 931, 419, 12, 597, 90, 67, 996, 300, 559, 427, 867, 151, 724, 13, 978, 153, 1004, 886, 803, 750, 768, 535, 316, 405, 802, 513, 547, 1022, 841, 585, 299, 15, 121, 588, 234, 48, 866, 361, 756, 420, 496, 568, 423, 108, 429, 541, 91, 421, 808, 288, 315, 332, 426, 296, 114, 199, 453, 60, 977, 898, 583, 350, 527, 302, 508, 652, 251, 722, 52, 348, 805, 381, 186, 956, 615, 422, 876, 1020, 231, 670, 272, 966, 1018, 570, 377, 93, 20, 1013, 38, 862, 503, 206, 479, 518, 457, 796, 935, 440, 346, 243, 647, 561, 408, 343, 371, 388, 524, 127, 532, 333, 185, 714, 335, 8, 415, 668, 2, 308, 3, 655, 53, 739, 913, 773, 286, 200, 946, 476, 748, 816, 400, 554, 339, 229, 275, 113, 358, 964, 69, 342, 491, 0, 794, 806, 831, 375, 327, 1000, 10, 266, 157, 44, 451, 487, 654, 852, 771, 262, 840, 232, 176, 178, 933, 362, 241, 129, 584, 528, 612, 896, 227, 480, 626, 603, 753, 223, 110, 379, 244, 787, 908, 842, 445, 999, 790, 809, 789, 758, 674, 981, 804, 614, 192, 991, 472, 689, 591, 285, 939, 203, 627, 172, 116, 207, 224, 22, 884, 643, 309, 960, 1006, 834, 507, 927, 949, 183, 514, 238, 292, 699, 399, 634, 432, 216, 690, 481, 318, 684, 995, 168, 209, 731, 820, 1009, 835, 257, 610, 762, 64, 573, 140, 352, 940, 565, 246, 314, 848, 751, 469, 16, 711, 847, 409, 102, 818, 631, 1011, 320, 605, 471, 785, 687, 774, 493, 889, 450, 319, 466, 78, 590, 779, 706, 218, 104, 922, 201, 149, 488, 649, 836, 947, 595, 301, 1016, 814, 918, 728, 838, 815, 34, 738, 62, 354, 788, 414, 261, 324, 11, 856, 452, 485, 428, 786, 825, 387, 952, 410, 264, 265, 1019, 520, 609, 594, 557, 386, 766, 406, 883, 497, 305, 899, 994, 580, 328, 882, 505, 974, 549, 914, 548, 9, 633, 208, 715, 676, 253, 77, 126, 743, 502, 984, 211, 498, 576, 6, 443, 294, 274, 120, 398, 182, 697, 844, 795, 696, 761, 365, 279, 437, 1005, 735, 303, 190, 136, 273, 678, 645, 961, 764, 613, 456, 132, 510, 628, 353, 701, 461, 757, 563, 937, 546, 115, 780, 685, 404, 813, 119, 686, 920, 271, 369, 957, 433, 602, 618, 912, 189, 755, 341, 213, 868, 222, 885, 817, 161, 928, 254, 574, 147, 81, 727, 851, 385, 740, 992, 95, 744, 648, 202, 538, 811, 390, 418, 23, 624, 283, 173, 990, 887, 170, 860, 1012, 133, 383, 800, 819, 460, 871, 888, 917, 36, 148, 368, 716, 366, 824, 749, 79, 997, 252, 924, 217, 878, 504, 533, 512, 28, 729, 486, 542, 167, 625, 746, 742, 732, 797, 853, 793, 468, 536, 49, 166, 665, 839, 782, 349, 855, 720, 467, 194, 656, 828, 131, 769, 723, 776, 100, 517, 903, 515, 511, 71, 164, 760, 259, 86, 560, 879, 630, 671, 707, 135, 474, 726, 376, 578, 938, 936, 340, 923, 611, 156, 373, 177, 833, 930, 307, 281, 1, 63, 567, 80, 50, 877, 537, 138, 258, 355, 550, 228, 122, 642, 993, 500, 351, 431, 152, 679, 124, 180, 75, 99, 357, 128, 905, 249, 683, 84, 384, 1008, 971, 663, 123, 784, 73, 680, 556, 792, 18, 657, 214, 976, 975, 499, 763, 874, 298, 134, 666, 97, 403, 959, 424, 330, 980, 955, 941, 436, 470, 534, 311, 635, 596, 604, 752, 51, 407, 700, 893, 693, 41, 688, 651, 442, 812, 313, 904, 525, 70, 416, 221, 582, 521, 709, 754, 915, 370, 478, 326, 983, 781, 256, 827, 391, 171, 247, 174, 24, 710, 317, 112, 277, 482, 331, 620, 242, 394, 321, 137, 516, 66, 989, 435, 106, 220, 545, 698, 393, 799, 447, 873, 985, 270, 659, 767, 1003, 863, 158, 347, 703, 118, 380, 821, 338, 322, 944, 571, 555, 205, 622, 363, 88, 449, 660, 954, 282, 125, 204, 287, 196, 26, 948, 430, 163, 389, 783, 94, 289, 31, 640, 791, 344, 759, 553, 704, 1017, 747, 360, 953, 191, 1002, 850, 484, 65, 737, 181, 57, 367, 982, 392, 109, 564, 586, 694, 160, 623, 111, 143, 650, 334, 250, 988, 675, 506, 248, 236, 872, 323, 778, 39, 5, 540, 801, 212, 865, 240, 260, 434, 926, 175, 798, 639, 444, 193, 195, 998, 832, 325, 105, 1023, 579, 494, 463, 859, 979, 691, 616, 401, 907, 364, 1007, 638, 336, 263, 843, 4, 829, 225, 531, 857, 730, 529, 543, 972, 667, 880, 562, 145, 601, 83, 970, 600, 438, 608, 902, 987, 861, 477, 144, 417, 55, 849, 268, 107, 412, 56, 943, 329], [997, 616, 285, 706, 443, 601, 327, 253, 1004, 124, 366, 63, 212, 844, 261, 989, 78, 181, 92, 984, 988, 475, 758, 231, 200, 127, 770, 877, 829, 288, 478, 842, 221, 636, 719, 999, 946, 733, 756, 11, 551, 16, 452, 147, 725, 232, 304, 139, 42, 113, 335, 957, 312, 154, 907, 401, 497, 134, 100, 501, 855, 663, 52, 125, 865, 90, 962, 204, 246, 613, 859, 701, 562, 607, 203, 932, 486, 918, 882, 30, 791, 782, 303, 909, 925, 941, 438, 858, 271, 530, 184, 919, 629, 774, 222, 247, 965, 167, 268, 678, 611, 695, 202, 708, 594, 225, 639, 468, 357, 1020, 102, 38, 1014, 432, 450, 503, 807, 906, 648, 980, 187, 674, 276, 39, 631, 118, 500, 162, 747, 664, 843, 24, 667, 744, 146, 376, 698, 337, 89, 937, 9, 517, 0, 745, 151, 532, 1021, 769, 481, 852, 270, 110, 687, 1006, 776, 265, 588, 761, 933, 69, 913, 596, 267, 750, 359, 956, 940, 310, 10, 788, 424, 72, 488, 423, 473, 741, 218, 928, 570, 320, 949, 396, 371, 6, 447, 21, 645, 34, 893, 155, 713, 318, 194, 116, 628, 752, 157, 263, 96, 571, 679, 942, 890, 765, 914, 101, 400, 363, 813, 857, 211, 771, 216, 403, 377, 322, 257, 947, 567, 299, 35, 451, 585, 353, 239, 180, 13, 614, 94, 61, 802, 339, 669, 453, 943, 700, 513, 182, 407, 193, 289, 847, 4, 234, 161, 144, 680, 82, 388, 37, 343, 248, 91, 65, 86, 120, 60, 521, 512, 967, 492, 41, 50, 133, 587, 945, 415, 243, 430, 510, 557, 254, 190, 830, 763, 536, 764, 618, 939, 56, 54, 952, 296, 589, 900, 995, 798, 274, 3, 579, 768, 291, 904, 171, 827, 166, 213, 419, 374, 97, 384, 390, 284, 642, 533, 328, 282, 854, 281, 839, 22, 1000, 730, 106, 502, 1009, 386, 602, 165, 685, 191, 757, 749, 818, 801, 863, 1016, 920, 650, 568, 467, 739, 480, 73, 740, 59, 974, 622, 975, 83, 188, 53, 523, 479, 888, 256, 646, 176, 889, 365, 519, 183, 871, 439, 710, 875, 5, 466, 220, 55, 140, 57, 958, 108, 8, 461, 954, 308, 541, 709, 394, 383, 141, 545, 88, 566, 85, 495, 368, 505, 576, 43, 329, 199, 524, 1022, 169, 484, 278, 273, 192, 626, 283, 298, 838, 143, 808, 959, 294, 835, 597, 986, 445, 881, 850, 405, 67, 93, 20, 28, 464, 675, 902, 387, 506, 785, 555, 51, 347, 684, 404, 68, 917, 655, 901, 399, 36, 172, 781, 44, 563, 849, 599, 485, 994, 672, 414, 1019, 821, 237, 892, 960, 228, 168, 908, 344, 156, 659, 259, 879, 245, 150, 153, 2, 705, 224, 302, 115, 380, 474, 145, 409, 751, 74, 277, 779, 856, 886, 25, 697, 1005, 186, 732, 620, 112, 718, 683, 170, 565, 434, 236, 137, 31, 819, 32, 449, 255, 766, 624, 931, 511, 436, 727, 591, 598, 982, 748, 130, 811, 586, 822, 331, 755, 600, 569, 573, 734, 398, 572, 661, 493, 707, 963, 910, 422, 841, 688, 107, 550, 427, 7, 696, 307, 109, 70, 412, 950, 582, 640, 392, 280, 81, 972, 762, 787, 33, 163, 98, 704, 469, 693, 402, 979, 559, 736, 936, 544, 71, 418, 472, 584, 846, 527, 866, 361, 927, 324, 671, 735, 105, 647, 682, 103, 19, 746, 627, 465, 926, 556, 731, 75, 912, 305, 944, 615, 773, 714, 826, 794, 252, 340, 279, 903, 883, 207, 896, 95, 46, 477, 606, 870, 722, 716, 214, 681, 269, 429, 251, 558, 658, 471, 364, 824, 538, 323, 961, 552, 48, 833, 692, 40, 637, 111, 867, 515, 581, 795, 185, 463, 381, 275, 12, 993, 662, 934, 330, 528, 694, 998, 809, 969, 230, 799, 728, 621, 970, 820, 1023, 848, 723, 1012, 657, 375, 924, 772, 433, 411, 131, 665, 964, 117, 869, 522, 670, 592, 514, 425, 775, 482, 421, 173, 691, 196, 816, 978, 286, 321, 1011, 309, 790, 712, 178, 689, 410, 138, 1015, 306, 29, 126, 531, 1007, 122, 634, 789, 351, 420, 676, 577, 1013, 17, 440, 780, 633, 80, 1008, 635, 1, 915, 121, 64, 27, 560, 435, 690, 373, 929, 612, 197, 630, 897, 792, 976, 643, 1001, 673, 152, 834, 350, 542, 77, 356, 47, 489, 992, 1002, 575, 837, 264, 272, 266, 891, 580, 487, 408, 201, 499, 459, 977, 313, 930, 605, 385, 610, 535, 460, 796, 759, 868, 195, 554, 711, 326, 382, 743, 668, 348, 561, 987, 815, 240, 717, 777, 367, 720, 79, 442, 887, 593, 836, 397, 729, 287, 715, 132, 800, 104, 319, 226, 632, 260, 114, 66, 23, 370, 345, 651, 229, 760, 814, 290, 583, 249, 652, 983, 783, 831, 895, 49, 539, 619, 26, 861, 617, 1003, 498, 921, 62, 653, 258, 873, 504, 352, 297, 590, 338, 354, 444, 529, 295, 206, 333, 417, 860, 547, 128, 483, 241, 754, 526, 238, 393, 968, 458, 537, 336, 15, 778, 164, 148, 119, 129, 332, 686, 142, 742, 198, 311, 991, 948, 316, 546, 540, 490, 862, 564, 360, 793, 724, 955, 448, 87, 317, 549, 786, 395, 174, 644, 638, 810, 737, 496, 437, 292, 1010, 205, 346, 158, 509, 136, 981, 548, 159, 175, 702, 878, 379, 520, 953, 738, 355, 250, 160, 14, 966, 721, 242, 369, 851, 518, 454, 179, 135, 406, 494, 457, 595, 1017, 508, 625, 703, 828, 553, 923, 534, 812, 215, 491, 456, 990, 470, 227, 362, 623, 840, 431, 641, 123, 208, 358, 372, 825, 604, 293, 99, 899, 446, 349, 301, 18, 677, 219, 76, 951, 894, 235, 784, 233, 45, 413, 884, 416, 428, 880, 898, 574, 391, 262, 938, 426, 209, 217, 660, 58, 804, 832, 666, 608, 654, 609, 189, 341, 149, 476, 342, 244, 603, 911, 971, 973, 462, 516, 853, 876, 905, 872, 935, 378, 84, 543, 805, 823, 874, 699, 985, 455, 656, 726, 864, 817, 177, 916, 885, 767, 922, 507, 210, 797, 649, 806, 334, 1018, 223, 845, 314, 803, 300, 578, 325, 525, 441, 753, 315, 996, 389]], [[265, 109, 7, 664, 662, 120, 213, 918, 472, 476, 389, 638, 193, 795, 600, 689, 985, 421, 49, 945, 992, 184, 463, 920, 794, 414, 70, 415, 904, 677, 906, 813, 146, 264, 321, 948, 150, 515, 727, 520, 38, 539, 111, 176, 942, 298, 953, 9, 777, 173, 494, 892, 491, 133, 20, 156, 138, 629, 351, 358, 868, 866, 238, 749, 680, 2, 69, 673, 77, 302, 40, 245, 142, 678, 151, 411, 932, 840, 559, 178, 699, 478, 921, 817, 101, 160, 335, 148, 688, 144, 260, 1010, 1006, 835, 73, 962, 882, 941, 527, 326, 14, 593, 359, 723, 598, 1002, 149, 297, 547, 743, 977, 74, 732, 79, 885, 199, 1003, 4, 286, 290, 710, 289, 464, 401, 492, 499, 433, 955, 189, 592, 182, 579, 686, 958, 12, 445, 470, 990, 152, 261, 262, 517, 317, 591, 347, 635, 997, 194, 674, 1, 676, 1011, 700, 653, 800, 789, 605, 257, 711, 479, 588, 64, 756, 393, 879, 498, 991, 274, 57, 30, 960, 348, 400, 995, 974, 132, 116, 632, 950, 546, 37, 469, 299, 973, 338, 439, 51, 24, 139, 334, 944, 481, 510, 61, 1020, 709, 258, 131, 667, 490, 228, 250, 535, 1009, 390, 758, 254, 190, 354, 858, 223, 965, 728, 405, 200, 630, 284, 236, 461, 427, 155, 90, 328, 730, 838, 669, 507, 911, 922, 186, 256, 493, 484, 518, 161, 382, 987, 847, 574, 759, 72, 548, 513, 86, 195, 429, 19, 506, 477, 1001, 447, 36, 696, 203, 826, 407, 118, 905, 319, 721, 52, 13, 555, 769, 929, 685, 612, 818, 460, 735, 737, 571, 305, 1000, 126, 634, 618, 26, 776, 704, 363, 633, 734, 10, 206, 423, 56, 448, 454, 928, 665, 691, 914, 836, 617, 738, 399, 956, 221, 211, 557, 431, 596, 886, 582, 952, 244, 889, 364, 720, 311, 96, 619, 219, 432, 235, 594, 296, 656, 444, 781, 330, 344, 295, 851, 316, 62, 538, 841, 315, 580, 1012, 624, 784, 252, 419, 560, 475, 314, 446, 829, 1005, 442, 874, 930, 585, 558, 595, 525, 505, 217, 357, 99, 508, 462, 487, 523, 862, 796, 372, 422, 846, 741, 6, 185, 845, 979, 119, 590, 277, 566, 640, 563, 788, 567, 80, 861, 562, 366, 516, 972, 912, 745, 628, 984, 457, 802, 259, 1023, 436, 125, 197, 536, 951, 702, 293, 875, 561, 135, 876, 934, 856, 626, 402, 891, 602, 489, 778, 553, 502, 242, 715, 603, 687, 169, 639, 360, 331, 908, 791, 27, 300, 468, 690, 705, 663, 855, 230, 627, 551, 215, 853, 575, 140, 746, 214, 60, 381, 22, 659, 819, 807, 939, 976, 94, 241, 43, 441, 910, 666, 524, 893, 842, 860, 645, 733, 497, 884, 961, 647, 589, 933, 486, 209, 572, 306, 269, 712, 373, 786, 488, 167, 180, 999, 23, 610, 418, 188, 435, 337, 33, 825, 343, 201, 981, 88, 263, 325, 899, 988, 655, 724, 322, 121, 166, 312, 514, 55, 246, 913, 129, 606, 100, 63, 982, 768, 916, 110, 1022, 926, 919, 890, 181, 1004, 801, 545, 751, 898, 642, 681, 480, 772, 983, 162, 220, 17, 978, 844, 543, 501, 816, 234, 179, 947, 353, 380, 384, 417, 625, 649, 609, 587, 170, 931, 304, 857, 613, 552, 154, 255, 232, 810, 466, 75, 127, 456, 270, 809, 425, 550, 483, 465, 998, 45, 183, 767, 143, 362, 949, 648, 225, 104, 482, 577, 532, 719, 58, 927, 651, 50, 568, 570, 134, 451, 332, 725, 731, 637, 726, 668, 909, 717, 459, 943, 240, 53, 863, 426, 35, 287, 775, 145, 864, 703, 692, 811, 229, 670, 339, 646, 765, 597, 237, 815, 614, 1008, 268, 455, 440, 141, 65, 871, 556, 793, 878, 970, 374, 113, 694, 98, 804, 578, 66, 695, 843, 123, 877, 168, 701, 683, 392, 822, 266, 599, 837, 790, 350, 760, 867, 569, 177, 231, 385, 584, 530, 303, 573, 371, 136, 986, 175, 509, 163, 227, 341, 654, 996, 204, 191, 76, 153, 636, 849, 434, 29, 771, 0, 675, 722, 115, 207, 1013, 32, 81, 272, 205, 68, 397, 528, 313, 416, 192, 968, 198, 345, 210, 1018, 318, 742, 47, 42, 226, 253, 105, 827, 954, 706, 650, 282, 25, 522, 128, 496, 500, 753, 92, 975, 273, 420, 631, 957, 108, 754, 963, 216, 394, 91, 471, 658, 386, 792, 106, 18, 830, 937, 54, 449, 249, 474, 437, 896, 413, 398, 980, 83, 1015, 458, 292, 248, 697, 888, 114, 989, 823, 542, 365, 89, 764, 839, 850, 824, 511, 752, 887, 622, 1007, 1017, 324, 969, 854, 554, 485, 744, 327, 870, 865, 78, 1021, 387, 329, 187, 698, 615, 902, 85, 355, 644, 349, 430, 540, 196, 529, 541, 718, 122, 808, 512, 130, 873, 267, 396, 164, 787, 208, 412, 821, 779, 383, 450, 924, 938, 410, 31, 544, 693, 453, 46, 112, 395, 5, 107, 925, 443, 621, 1016, 218, 917, 679, 503, 806, 900, 291, 608, 936, 137, 833, 994, 294, 356, 307, 831, 239, 224, 124, 966, 379, 643, 799, 533, 408, 607, 946, 285, 959, 428, 881, 803, 21, 3, 93, 923, 805, 283, 755, 278, 59, 534, 814, 84, 377, 526, 247, 342, 34, 616, 103, 233, 243, 747, 159, 288, 280, 707, 716, 301, 95, 251, 872, 367, 352, 452, 281, 172, 369, 797, 971, 832, 736, 388, 869, 346, 409, 682, 798, 404, 601, 852, 97, 28, 279, 16, 785, 713, 309, 378, 521, 774, 583, 1014, 623, 157, 883, 763, 660, 684, 403, 174, 766, 473, 782, 714, 537, 336, 820, 641, 967, 661, 762, 531, 708, 895, 202, 8, 504, 495, 915, 964, 102, 48, 581, 519, 87, 147, 11, 276, 848, 770, 549, 310, 333, 41, 165, 212, 773, 993, 859, 117, 748, 370, 1019, 897, 834, 894, 361, 406, 171, 222, 565, 467, 564, 880, 376, 15, 604, 901, 729, 323, 740, 71, 438, 739, 275, 340, 620, 611, 586, 750, 82, 940, 652, 657, 67, 375, 907, 320, 576, 828, 935, 783, 44, 672, 39, 271, 757, 761, 368, 903, 308, 780, 424, 671, 391, 812, 158], [1023, 891, 720, 75, 624, 219, 805, 413, 426, 873, 785, 824, 779, 225, 609, 634, 904, 899, 114, 434, 874, 269, 50, 451, 412, 747, 717, 1007, 390, 778, 776, 948, 494, 716, 374, 138, 117, 168, 692, 957, 178, 575, 213, 311, 970, 266, 173, 954, 329, 713, 783, 757, 234, 151, 847, 921, 802, 649, 865, 181, 340, 887, 358, 113, 408, 96, 686, 314, 104, 782, 460, 405, 450, 15, 728, 584, 725, 395, 1003, 809, 470, 249, 619, 132, 44, 815, 1019, 254, 273, 94, 142, 601, 430, 832, 928, 868, 166, 191, 736, 320, 204, 751, 561, 301, 345, 258, 528, 648, 183, 148, 702, 243, 56, 980, 6, 14, 533, 482, 344, 527, 926, 476, 373, 21, 858, 167, 571, 206, 659, 248, 603, 708, 626, 959, 896, 217, 105, 30, 447, 495, 36, 501, 1, 106, 743, 857, 222, 975, 491, 403, 931, 135, 812, 93, 666, 825, 848, 637, 226, 705, 325, 361, 221, 195, 591, 477, 798, 149, 188, 1012, 431, 576, 176, 492, 364, 55, 235, 647, 107, 536, 590, 806, 153, 859, 769, 960, 493, 321, 459, 1018, 745, 838, 214, 150, 883, 966, 305, 509, 346, 61, 420, 552, 569, 617, 338, 920, 157, 417, 670, 179, 279, 558, 587, 317, 34, 190, 1010, 520, 388, 772, 25, 526, 665, 677, 907, 605, 532, 816, 283, 145, 923, 893, 631, 735, 748, 897, 918, 843, 362, 521, 400, 63, 88, 49, 245, 369, 688, 955, 127, 120, 540, 661, 978, 28, 462, 953, 797, 102, 80, 457, 719, 508, 144, 62, 737, 251, 211, 302, 1004, 1005, 581, 801, 908, 936, 612, 939, 415, 901, 834, 487, 704, 828, 819, 158, 12, 756, 474, 125, 216, 4, 611, 252, 982, 795, 295, 1017, 300, 391, 560, 272, 504, 115, 366, 963, 212, 678, 449, 411, 101, 498, 237, 143, 530, 869, 337, 65, 160, 991, 687, 524, 18, 945, 761, 331, 232, 371, 833, 644, 597, 718, 799, 826, 45, 746, 20, 347, 658, 335, 423, 650, 116, 712, 293, 989, 585, 1001, 1021, 681, 861, 543, 913, 870, 837, 796, 967, 938, 703, 976, 73, 433, 503, 570, 674, 318, 394, 473, 934, 285, 510, 370, 645, 1014, 972, 339, 24, 261, 863, 316, 867, 914, 185, 92, 522, 732, 696, 241, 184, 607, 60, 657, 1015, 707, 924, 399, 372, 250, 74, 192, 621, 170, 458, 238, 1008, 638, 353, 582, 455, 961, 46, 860, 632, 471, 313, 653, 48, 724, 903, 554, 398, 877, 100, 654, 845, 505, 564, 690, 98, 290, 421, 264, 663, 906, 312, 137, 1000, 59, 993, 350, 448, 537, 807, 468, 70, 263, 602, 209, 172, 479, 112, 224, 994, 985, 240, 640, 230, 231, 811, 397, 84, 507, 518, 544, 202, 849, 800, 416, 511, 789, 871, 912, 76, 466, 259, 469, 446, 119, 319, 440, 333, 875, 236, 642, 818, 422, 592, 551, 359, 444, 956, 566, 174, 402, 709, 375, 432, 198, 588, 753, 784, 79, 367, 452, 760, 542, 539, 675, 894, 244, 820, 749, 1020, 937, 407, 662, 987, 255, 220, 559, 608, 288, 480, 53, 169, 428, 995, 862, 889, 992, 121, 454, 573, 971, 485, 82, 930, 464, 141, 765, 486, 442, 304, 827, 32, 729, 223, 1002, 85, 853, 517, 419, 627, 461, 803, 382, 990, 844, 523, 839, 814, 697, 973, 180, 682, 623, 336, 547, 86, 280, 134, 836, 97, 189, 866, 379, 774, 1013, 147, 72, 89, 118, 949, 742, 846, 915, 140, 246, 438, 694, 790, 71, 614, 323, 210, 633, 715, 639, 58, 378, 669, 57, 553, 583, 357, 944, 496, 810, 260, 39, 42, 443, 599, 900, 888, 171, 935, 549, 656, 208, 307, 655, 165, 722, 981, 253, 23, 159, 788, 726, 643, 777, 152, 730, 31, 355, 360, 884, 940, 349, 404, 574, 348, 481, 506, 227, 409, 589, 752, 909, 974, 744, 733, 618, 679, 387, 946, 755, 762, 489, 699, 673, 483, 898, 262, 852, 676, 872, 947, 598, 886, 193, 984, 203, 475, 758, 37, 572, 294, 567, 424, 534, 513, 593, 881, 786, 578, 660, 941, 276, 87, 90, 334, 429, 308, 43, 207, 556, 478, 952, 436, 1022, 384, 750, 879, 7, 671, 401, 646, 380, 124, 829, 929, 229, 911, 341, 328, 69, 270, 502, 651, 775, 389, 546, 386, 35, 822, 161, 856, 393, 999, 667, 500, 326, 791, 26, 0, 265, 917, 488, 128, 780, 95, 519, 616, 594, 108, 706, 997, 287, 557, 131, 988, 445, 831, 969, 986, 83, 740, 741, 418, 200, 625, 615, 186, 126, 983, 123, 792, 1011, 332, 600, 629, 958, 711, 136, 727, 835, 5, 922, 155, 580, 514, 622, 555, 274, 695, 668, 256, 110, 689, 292, 759, 854, 630, 919, 197, 771, 109, 680, 529, 51, 352, 376, 410, 635, 538, 16, 964, 721, 8, 365, 840, 628, 139, 683, 19, 979, 895, 156, 880, 392, 425, 925, 490, 610, 787, 414, 182, 406, 268, 303, 855, 821, 315, 27, 942, 890, 111, 885, 77, 739, 472, 763, 684, 842, 427, 965, 239, 396, 652, 291, 851, 456, 309, 467, 773, 38, 67, 297, 196, 242, 385, 40, 1016, 595, 808, 817, 215, 768, 439, 342, 497, 512, 545, 691, 830, 282, 698, 864, 66, 154, 968, 541, 701, 281, 723, 781, 267, 586, 882, 257, 916, 327, 1009, 175, 324, 962, 99, 187, 162, 579, 548, 604, 998, 932, 841, 52, 194, 47, 850, 275, 81, 754, 10, 377, 435, 550, 381, 201, 11, 133, 813, 562, 383, 199, 130, 103, 484, 731, 804, 354, 228, 793, 770, 693, 1006, 296, 64, 516, 284, 977, 286, 306, 3, 78, 951, 794, 531, 577, 996, 565, 41, 943, 734, 271, 933, 927, 902, 164, 563, 2, 29, 356, 568, 247, 22, 463, 613, 766, 499, 700, 437, 596, 767, 330, 368, 876, 146, 91, 606, 68, 764, 465, 322, 310, 525, 620, 910, 289, 122, 233, 343, 672, 278, 363, 9, 636, 641, 905, 823, 892, 738, 714, 950, 878, 17, 535, 299, 54, 129, 664, 441, 277, 298, 685, 218, 351, 205, 13, 515, 163, 33, 453, 710, 177], [298, 804, 140, 640, 598, 830, 654, 520, 715, 978, 104, 166, 94, 953, 445, 933, 472, 63, 2, 95, 475, 750, 790, 246, 390, 1006, 351, 847, 402, 897, 685, 1001, 416, 247, 493, 688, 619, 333, 400, 36, 26, 578, 727, 337, 638, 70, 919, 507, 473, 431, 414, 127, 409, 377, 684, 813, 562, 132, 136, 346, 686, 361, 291, 701, 384, 120, 720, 601, 612, 162, 658, 900, 995, 573, 1022, 652, 141, 42, 210, 879, 912, 711, 959, 381, 492, 44, 172, 888, 937, 395, 117, 827, 422, 229, 194, 554, 199, 17, 215, 31, 556, 815, 887, 761, 248, 242, 322, 108, 998, 802, 575, 163, 911, 939, 676, 344, 310, 343, 230, 541, 388, 629, 511, 355, 253, 302, 865, 184, 930, 683, 398, 954, 89, 687, 721, 542, 543, 962, 47, 25, 454, 284, 1005, 383, 695, 261, 265, 669, 584, 546, 227, 134, 315, 875, 874, 518, 792, 512, 218, 115, 368, 611, 211, 885, 990, 58, 844, 177, 311, 762, 623, 867, 84, 735, 276, 648, 99, 832, 724, 499, 129, 286, 7, 440, 909, 268, 433, 746, 595, 371, 583, 783, 448, 936, 175, 657, 515, 892, 1008, 467, 760, 34, 299, 159, 831, 829, 843, 148, 625, 968, 593, 209, 986, 490, 112, 893, 219, 841, 1011, 737, 87, 403, 231, 258, 780, 469, 330, 235, 946, 277, 290, 234, 549, 97, 842, 487, 233, 67, 781, 934, 385, 318, 336, 432, 561, 378, 816, 437, 256, 308, 682, 548, 764, 280, 679, 517, 369, 32, 68, 359, 646, 161, 530, 489, 702, 993, 796, 788, 704, 207, 396, 149, 317, 523, 436, 450, 969, 660, 417, 460, 563, 88, 859, 22, 362, 160, 811, 941, 555, 278, 146, 8, 353, 838, 908, 240, 1014, 168, 1003, 742, 743, 181, 77, 979, 898, 288, 791, 217, 634, 101, 996, 92, 881, 495, 630, 497, 169, 744, 474, 642, 752, 85, 967, 719, 4, 1023, 301, 103, 774, 109, 952, 833, 342, 590, 580, 976, 424, 61, 6, 846, 477, 62, 107, 956, 367, 613, 826, 565, 463, 118, 853, 509, 594, 404, 907, 531, 293, 782, 992, 948, 678, 285, 11, 457, 597, 461, 244, 80, 697, 753, 663, 707, 602, 212, 987, 413, 951, 756, 871, 131, 615, 527, 825, 52, 155, 119, 505, 262, 958, 655, 903, 798, 839, 430, 835, 394, 466, 425, 406, 981, 57, 524, 224, 91, 239, 102, 896, 974, 252, 223, 739, 803, 927, 455, 845, 270, 98, 545, 824, 552, 1012, 671, 641, 564, 786, 110, 916, 624, 799, 596, 864, 18, 307, 577, 557, 49, 418, 397, 723, 438, 269, 323, 447, 335, 370, 35, 145, 124, 605, 626, 412, 9, 921, 435, 585, 508, 924, 179, 770, 340, 621, 139, 13, 716, 659, 894, 984, 96, 738, 428, 647, 665, 500, 21, 365, 680, 201, 114, 348, 491, 75, 880, 38, 582, 105, 312, 59, 551, 610, 650, 856, 186, 653, 37, 886, 534, 733, 775, 193, 225, 915, 484, 748, 784, 943, 656, 728, 121, 332, 486, 0, 944, 375, 718, 439, 607, 899, 757, 228, 73, 522, 446, 174, 576, 902, 213, 857, 405, 347, 840, 985, 123, 379, 809, 458, 929, 793, 326, 30, 150, 106, 281, 128, 151, 922, 544, 287, 1007, 165, 195, 882, 372, 133, 130, 818, 591, 142, 627, 717, 823, 706, 79, 386, 767, 100, 633, 373, 712, 836, 812, 64, 387, 526, 171, 810, 920, 275, 476, 182, 90, 877, 237, 917, 710, 255, 254, 313, 977, 327, 991, 3, 46, 983, 138, 779, 392, 755, 325, 429, 536, 644, 572, 529, 189, 60, 698, 250, 592, 516, 86, 651, 14, 614, 203, 606, 328, 950, 462, 913, 415, 1017, 423, 661, 1004, 525, 292, 980, 801, 331, 1009, 868, 854, 699, 664, 510, 806, 241, 319, 1013, 928, 399, 1020, 918, 41, 666, 407, 855, 303, 282, 819, 631, 53, 56, 296, 167, 304, 745, 427, 713, 667, 689, 749, 295, 1002, 152, 178, 872, 622, 521, 188, 821, 581, 703, 65, 358, 185, 535, 997, 873, 820, 15, 600, 19, 363, 955, 949, 608, 78, 66, 279, 205, 537, 502, 71, 143, 635, 485, 374, 456, 137, 519, 805, 442, 1016, 334, 848, 550, 514, 904, 232, 540, 496, 238, 957, 196, 411, 884, 122, 316, 586, 800, 794, 420, 12, 895, 444, 271, 849, 5, 725, 204, 691, 479, 973, 352, 604, 776, 675, 734, 1, 589, 673, 401, 192, 464, 777, 914, 480, 273, 547, 558, 714, 116, 772, 76, 345, 693, 560, 24, 10, 994, 441, 20, 866, 618, 700, 740, 257, 645, 1000, 82, 559, 965, 639, 732, 274, 222, 54, 861, 283, 1018, 202, 135, 961, 860, 938, 708, 637, 349, 729, 410, 940, 876, 964, 156, 513, 773, 768, 539, 382, 970, 309, 696, 393, 93, 890, 632, 72, 787, 966, 113, 945, 354, 942, 451, 39, 506, 40, 426, 81, 731, 83, 931, 249, 176, 488, 726, 1021, 690, 453, 769, 27, 220, 376, 709, 588, 305, 649, 206, 538, 216, 324, 208, 51, 579, 1019, 214, 48, 587, 314, 963, 338, 670, 471, 789, 481, 910, 694, 443, 891, 449, 532, 862, 173, 837, 662, 183, 778, 478, 878, 260, 356, 164, 289, 245, 971, 852, 360, 722, 785, 1015, 197, 468, 470, 925, 187, 741, 692, 730, 828, 341, 817, 570, 272, 553, 33, 389, 628, 771, 569, 869, 29, 636, 236, 834, 158, 267, 501, 419, 574, 226, 947, 144, 266, 366, 498, 191, 988, 751, 982, 599, 263, 620, 434, 999, 603, 763, 797, 674, 200, 16, 294, 111, 380, 251, 408, 329, 339, 567, 766, 154, 459, 483, 901, 28, 677, 452, 935, 808, 795, 960, 320, 883, 153, 221, 494, 754, 259, 503, 568, 736, 421, 482, 747, 765, 465, 571, 668, 74, 43, 364, 157, 814, 504, 975, 126, 147, 533, 858, 616, 863, 672, 350, 609, 807, 906, 566, 190, 617, 306, 23, 180, 681, 300, 923, 170, 528, 297, 321, 264, 357, 905, 1010, 926, 643, 851, 758, 870, 125, 889, 989, 45, 391, 55, 972, 932, 822, 705, 759, 850, 243, 198, 50, 69], [4, 568, 159, 546, 889, 177, 298, 942, 363, 711, 959, 388, 637, 949, 975, 1005, 790, 22, 218, 920, 438, 980, 994, 545, 722, 888, 227, 263, 540, 503, 664, 226, 549, 328, 181, 1012, 544, 831, 848, 351, 988, 826, 593, 92, 794, 784, 236, 455, 915, 601, 325, 748, 254, 807, 235, 890, 894, 342, 86, 303, 851, 329, 496, 393, 69, 770, 948, 741, 428, 81, 139, 494, 898, 291, 824, 660, 957, 717, 701, 612, 804, 652, 998, 778, 798, 95, 20, 877, 127, 389, 333, 940, 30, 586, 671, 56, 600, 823, 795, 796, 845, 98, 152, 509, 665, 260, 474, 87, 250, 1008, 230, 628, 265, 972, 425, 484, 394, 255, 79, 257, 203, 724, 730, 275, 343, 610, 991, 205, 860, 397, 607, 173, 909, 175, 536, 779, 251, 26, 483, 1006, 662, 80, 759, 999, 558, 945, 435, 870, 443, 955, 153, 84, 120, 751, 927, 777, 745, 148, 36, 223, 324, 314, 806, 269, 756, 330, 974, 908, 903, 625, 986, 106, 214, 837, 810, 832, 220, 587, 305, 131, 621, 318, 294, 516, 608, 938, 976, 887, 420, 917, 138, 457, 766, 811, 574, 632, 584, 554, 865, 346, 122, 229, 961, 514, 48, 293, 964, 734, 187, 486, 310, 57, 23, 107, 415, 41, 576, 276, 657, 256, 935, 582, 166, 386, 737, 633, 248, 914, 412, 969, 74, 66, 398, 453, 788, 867, 121, 532, 952, 678, 99, 733, 1007, 698, 201, 825, 374, 401, 167, 525, 834, 270, 261, 15, 1003, 417, 385, 1013, 581, 421, 773, 560, 699, 603, 566, 39, 432, 755, 274, 313, 710, 931, 716, 556, 105, 588, 682, 83, 136, 815, 996, 195, 907, 287, 850, 49, 750, 694, 707, 207, 636, 365, 472, 340, 874, 395, 245, 747, 799, 836, 592, 906, 843, 0, 776, 667, 785, 140, 656, 573, 319, 162, 280, 869, 715, 88, 641, 668, 691, 738, 13, 233, 841, 307, 402, 149, 953, 941, 512, 266, 619, 720, 993, 816, 522, 683, 1010, 89, 72, 183, 855, 422, 605, 190, 827, 704, 697, 151, 135, 775, 932, 247, 620, 973, 966, 789, 479, 700, 895, 109, 797, 859, 231, 782, 875, 524, 354, 866, 505, 872, 897, 240, 561, 533, 452, 760, 919, 653, 62, 1021, 344, 646, 685, 594, 627, 426, 384, 146, 883, 258, 380, 963, 930, 731, 163, 578, 78, 928, 666, 1022, 858, 854, 94, 977, 912, 947, 147, 155, 769, 868, 835, 965, 531, 876, 922, 113, 820, 414, 498, 726, 951, 787, 272, 842, 312, 367, 253, 901, 992, 289, 650, 129, 182, 761, 958, 85, 75, 708, 45, 110, 921, 170, 821, 404, 145, 622, 551, 893, 861, 521, 614, 613, 336, 677, 370, 929, 571, 9, 515, 241, 879, 714, 150, 703, 118, 793, 28, 644, 473, 623, 616, 853, 878, 800, 100, 364, 55, 548, 180, 143, 564, 35, 488, 1001, 124, 309, 169, 871, 840, 780, 754, 406, 224, 663, 454, 962, 598, 709, 321, 618, 19, 433, 774, 372, 983, 199, 611, 286, 757, 299, 331, 34, 1014, 647, 31, 67, 721, 160, 746, 547, 873, 476, 771, 705, 939, 1011, 116, 634, 946, 891, 933, 262, 541, 543, 60, 783, 409, 459, 193, 158, 448, 228, 14, 572, 609, 392, 341, 492, 379, 212, 297, 819, 102, 569, 719, 301, 500, 537, 655, 530, 296, 200, 735, 117, 358, 762, 126, 222, 552, 918, 168, 332, 658, 491, 981, 577, 430, 40, 58, 178, 829, 732, 817, 461, 419, 565, 5, 900, 729, 63, 317, 808, 497, 215, 249, 137, 595, 206, 246, 1023, 123, 360, 902, 904, 466, 1016, 542, 2, 535, 405, 847, 70, 640, 366, 899, 24, 635, 267, 557, 725, 244, 885, 1002, 456, 706, 239, 283, 767, 29, 238, 985, 359, 302, 44, 252, 103, 114, 978, 464, 356, 197, 337, 805, 846, 626, 672, 281, 802, 213, 563, 37, 979, 538, 282, 838, 987, 743, 481, 673, 670, 243, 290, 523, 654, 696, 410, 905, 772, 645, 6, 469, 439, 517, 383, 416, 786, 482, 591, 320, 468, 25, 194, 179, 295, 624, 350, 371, 277, 659, 259, 513, 465, 911, 555, 791, 440, 165, 742, 391, 925, 674, 357, 970, 630, 284, 765, 8, 232, 995, 489, 643, 506, 423, 322, 485, 278, 460, 519, 534, 736, 196, 353, 368, 93, 119, 562, 599, 937, 463, 675, 12, 818, 18, 10, 550, 96, 559, 53, 61, 339, 450, 718, 490, 101, 132, 527, 306, 480, 882, 268, 580, 1018, 361, 1019, 585, 781, 174, 638, 651, 209, 493, 924, 345, 446, 744, 511, 64, 431, 880, 375, 413, 68, 50, 639, 355, 376, 11, 82, 687, 211, 396, 447, 451, 590, 1017, 495, 676, 968, 242, 142, 335, 264, 579, 445, 856, 712, 502, 27, 487, 156, 926, 669, 809, 90, 164, 740, 934, 604, 936, 803, 334, 553, 434, 144, 681, 210, 954, 508, 467, 130, 54, 327, 830, 1, 597, 596, 570, 349, 288, 510, 728, 801, 172, 753, 864, 886, 649, 702, 896, 77, 176, 501, 764, 713, 141, 863, 518, 115, 849, 923, 217, 749, 311, 47, 17, 892, 567, 739, 661, 225, 381, 300, 315, 285, 43, 418, 583, 602, 526, 589, 128, 814, 688, 884, 304, 157, 984, 444, 104, 191, 407, 752, 204, 424, 308, 689, 352, 471, 967, 71, 65, 504, 458, 237, 112, 758, 208, 828, 1000, 982, 960, 686, 684, 216, 369, 326, 950, 323, 437, 997, 390, 812, 234, 956, 629, 763, 185, 723, 862, 46, 462, 52, 449, 188, 690, 189, 32, 382, 271, 219, 399, 273, 192, 615, 944, 539, 184, 852, 154, 408, 833, 857, 171, 161, 441, 133, 916, 910, 642, 51, 575, 989, 727, 387, 507, 913, 21, 499, 279, 475, 429, 134, 529, 221, 839, 198, 202, 42, 3, 91, 7, 477, 1020, 1004, 617, 76, 971, 338, 347, 442, 411, 378, 373, 1015, 844, 362, 1009, 16, 813, 768, 108, 348, 478, 528, 606, 33, 186, 111, 73, 403, 693, 59, 125, 792, 377, 436, 470, 38, 427, 400, 943, 990, 648, 881, 97, 316, 631, 692, 695, 822, 680, 292, 520, 679]], [[137, 895, 537, 20, 536, 324, 473, 869, 115, 682, 386, 343, 426, 716, 943, 408, 300, 898, 287, 748, 443, 52, 881, 680, 867, 483, 835, 812, 402, 714, 702, 71, 302, 726, 631, 683, 157, 850, 338, 870, 675, 645, 359, 422, 967, 350, 814, 146, 959, 263, 781, 121, 430, 24, 255, 259, 427, 61, 894, 892, 944, 634, 92, 566, 858, 364, 821, 776, 844, 454, 307, 763, 158, 797, 435, 54, 89, 859, 706, 485, 130, 500, 488, 922, 876, 569, 215, 501, 291, 103, 497, 459, 833, 826, 951, 787, 383, 696, 964, 59, 369, 3, 261, 923, 142, 896, 525, 984, 44, 12, 114, 436, 388, 211, 234, 970, 557, 567, 613, 777, 156, 981, 948, 795, 849, 417, 321, 909, 332, 783, 713, 328, 866, 756, 67, 412, 561, 877, 170, 48, 982, 8, 918, 274, 374, 653, 346, 80, 74, 623, 1000, 514, 220, 21, 717, 883, 104, 878, 549, 950, 421, 735, 977, 605, 722, 574, 641, 688, 434, 1018, 226, 63, 955, 677, 743, 81, 492, 200, 15, 882, 535, 140, 49, 770, 940, 609, 551, 579, 205, 855, 22, 928, 816, 744, 597, 79, 519, 658, 547, 516, 606, 221, 585, 283, 902, 1019, 796, 224, 387, 464, 508, 1015, 903, 228, 672, 596, 518, 668, 650, 534, 788, 10, 437, 40, 360, 993, 929, 528, 913, 541, 753, 353, 550, 55, 352, 615, 667, 19, 423, 637, 243, 9, 976, 76, 233, 341, 1012, 248, 18, 82, 188, 58, 900, 542, 370, 552, 267, 175, 399, 33, 329, 932, 181, 279, 851, 694, 487, 603, 540, 345, 315, 539, 938, 968, 256, 649, 690, 506, 50, 398, 230, 107, 28, 1001, 184, 745, 730, 1008, 94, 334, 698, 911, 890, 247, 373, 577, 707, 956, 725, 665, 367, 357, 861, 700, 661, 558, 336, 679, 768, 681, 118, 496, 916, 840, 479, 531, 446, 37, 91, 857, 486, 213, 349, 720, 172, 1013, 789, 282, 384, 254, 520, 145, 176, 136, 969, 201, 530, 289, 354, 147, 991, 723, 129, 708, 410, 69, 589, 601, 666, 636, 72, 1003, 817, 495, 209, 242, 197, 604, 741, 433, 190, 266, 448, 578, 60, 786, 734, 344, 204, 5, 509, 257, 889, 752, 705, 939, 253, 161, 836, 622, 311, 401, 602, 966, 847, 100, 365, 475, 841, 860, 265, 451, 362, 600, 904, 461, 191, 285, 330, 644, 135, 280, 348, 187, 290, 469, 198, 907, 655, 260, 941, 297, 223, 934, 38, 988, 455, 499, 322, 591, 543, 719, 270, 555, 588, 775, 949, 790, 766, 246, 676, 921, 95, 736, 134, 695, 727, 99, 268, 143, 926, 524, 212, 305, 250, 660, 583, 575, 809, 405, 281, 998, 150, 47, 559, 162, 138, 498, 639, 848, 312, 504, 478, 828, 102, 994, 871, 886, 798, 546, 372, 935, 791, 413, 663, 617, 587, 293, 978, 669, 1007, 831, 155, 897, 489, 581, 113, 985, 441, 503, 1014, 864, 598, 453, 931, 746, 868, 238, 472, 355, 879, 919, 331, 570, 595, 957, 873, 802, 842, 468, 7, 39, 837, 216, 314, 958, 180, 625, 759, 754, 936, 390, 952, 376, 457, 701, 123, 117, 122, 428, 628, 133, 335, 699, 1, 31, 825, 491, 689, 26, 983, 691, 447, 169, 110, 554, 933, 724, 801, 231, 252, 23, 292, 98, 1009, 995, 594, 125, 415, 189, 13, 880, 908, 920, 84, 222, 808, 406, 14, 232, 296, 925, 563, 166, 697, 620, 112, 273, 693, 632, 686, 593, 755, 633, 568, 1020, 27, 66, 762, 177, 887, 109, 183, 16, 872, 992, 396, 301, 304, 394, 512, 590, 404, 830, 85, 414, 73, 843, 996, 576, 927, 832, 942, 607, 823, 1006, 799, 361, 325, 382, 160, 174, 915, 206, 449, 782, 425, 678, 635, 640, 86, 199, 648, 997, 917, 439, 1021, 526, 731, 513, 471, 510, 375, 999, 1010, 761, 288, 465, 874, 295, 337, 151, 371, 962, 17, 195, 704, 493, 0, 419, 32, 538, 70, 397, 467, 692, 738, 979, 11, 711, 463, 139, 517, 657, 481, 106, 45, 963, 779, 432, 750, 93, 901, 239, 674, 974, 409, 35, 56, 824, 319, 523, 818, 313, 227, 553, 391, 616, 219, 111, 511, 1011, 225, 987, 153, 474, 480, 381, 989, 793, 774, 865, 715, 829, 584, 78, 893, 820, 764, 947, 299, 445, 171, 564, 124, 522, 807, 347, 742, 278, 792, 41, 368, 424, 154, 128, 34, 819, 592, 671, 303, 654, 185, 178, 438, 624, 320, 116, 619, 356, 703, 351, 68, 971, 965, 164, 945, 379, 803, 214, 120, 194, 217, 800, 403, 477, 1005, 839, 444, 760, 186, 42, 765, 466, 203, 339, 490, 210, 276, 505, 75, 389, 767, 758, 532, 407, 891, 385, 899, 64, 88, 395, 780, 286, 906, 127, 608, 852, 545, 749, 251, 811, 272, 885, 685, 838, 684, 241, 309, 323, 208, 105, 527, 737, 582, 612, 771, 269, 87, 236, 342, 533, 148, 863, 757, 1002, 1017, 442, 986, 773, 560, 502, 611, 845, 431, 673, 244, 51, 119, 416, 96, 249, 784, 562, 529, 728, 862, 946, 450, 440, 132, 1004, 856, 778, 854, 961, 298, 284, 586, 418, 914, 207, 131, 363, 149, 930, 97, 262, 875, 275, 638, 46, 264, 306, 1016, 57, 340, 476, 815, 662, 721, 411, 810, 429, 573, 83, 954, 740, 192, 1022, 772, 43, 507, 980, 652, 392, 318, 482, 470, 710, 687, 458, 647, 937, 29, 975, 326, 2, 751, 627, 572, 333, 316, 77, 366, 884, 277, 664, 163, 548, 245, 144, 294, 614, 739, 173, 733, 651, 732, 452, 196, 905, 152, 973, 462, 709, 271, 953, 62, 646, 377, 53, 65, 141, 670, 36, 769, 888, 358, 235, 990, 400, 4, 643, 159, 610, 806, 794, 420, 785, 629, 659, 168, 378, 618, 805, 621, 237, 912, 813, 656, 747, 380, 460, 571, 456, 521, 846, 729, 718, 910, 25, 484, 310, 240, 556, 834, 515, 1023, 258, 972, 165, 308, 182, 193, 6, 827, 565, 101, 167, 108, 626, 960, 327, 393, 90, 179, 642, 804, 544, 202, 317, 822, 126, 630, 229, 218, 599, 494, 853, 712, 30, 580, 924], [102, 674, 874, 432, 11, 893, 672, 731, 877, 290, 783, 62, 419, 802, 936, 47, 105, 684, 359, 562, 204, 489, 938, 148, 243, 321, 431, 316, 161, 407, 999, 14, 800, 968, 3, 840, 1000, 380, 445, 788, 128, 917, 86, 943, 1001, 433, 284, 39, 172, 334, 358, 826, 974, 930, 752, 403, 32, 205, 763, 456, 254, 134, 639, 220, 209, 798, 700, 371, 464, 192, 50, 520, 983, 846, 839, 1009, 152, 932, 758, 556, 459, 237, 1023, 956, 565, 596, 72, 741, 300, 558, 782, 894, 1010, 90, 913, 957, 626, 749, 110, 955, 640, 127, 554, 601, 620, 327, 299, 36, 529, 637, 647, 117, 683, 76, 276, 813, 465, 345, 881, 552, 210, 361, 653, 537, 845, 372, 939, 247, 771, 452, 496, 507, 450, 806, 251, 387, 323, 853, 904, 668, 375, 548, 730, 534, 291, 177, 661, 370, 615, 764, 550, 447, 458, 104, 910, 436, 953, 710, 280, 725, 928, 451, 156, 89, 560, 422, 69, 664, 580, 655, 307, 720, 643, 413, 34, 201, 63, 838, 389, 1007, 780, 954, 676, 920, 44, 996, 673, 473, 139, 51, 24, 71, 682, 394, 969, 525, 278, 600, 479, 567, 634, 446, 937, 807, 141, 836, 218, 667, 514, 1003, 663, 524, 404, 590, 213, 592, 914, 707, 471, 80, 363, 315, 262, 268, 901, 186, 803, 962, 628, 945, 539, 488, 146, 81, 332, 277, 650, 476, 58, 907, 6, 339, 185, 561, 604, 140, 727, 779, 88, 891, 566, 75, 472, 994, 756, 12, 849, 5, 631, 757, 400, 70, 212, 292, 652, 811, 959, 871, 406, 680, 7, 457, 388, 1015, 762, 876, 235, 29, 500, 207, 895, 244, 124, 790, 131, 354, 644, 998, 978, 624, 42, 617, 1014, 347, 273, 695, 53, 19, 261, 941, 868, 837, 437, 714, 399, 366, 417, 455, 709, 689, 181, 777, 481, 573, 866, 1021, 900, 575, 485, 776, 657, 975, 1018, 341, 512, 919, 896, 435, 1006, 772, 511, 726, 997, 1002, 30, 882, 702, 150, 120, 331, 816, 328, 423, 184, 817, 805, 154, 242, 497, 165, 703, 931, 194, 353, 33, 564, 977, 274, 852, 246, 475, 675, 921, 961, 56, 760, 248, 660, 965, 1005, 883, 225, 812, 523, 173, 229, 797, 329, 918, 386, 770, 401, 851, 925, 952, 875, 678, 686, 747, 352, 198, 269, 716, 923, 178, 360, 970, 101, 228, 367, 505, 303, 311, 607, 350, 544, 761, 8, 355, 25, 223, 903, 115, 319, 314, 775, 197, 144, 119, 351, 136, 493, 765, 547, 443, 638, 103, 563, 865, 444, 41, 135, 735, 915, 180, 166, 175, 490, 260, 167, 677, 759, 133, 591, 15, 160, 855, 542, 990, 608, 233, 487, 340, 206, 873, 729, 850, 492, 870, 107, 722, 37, 469, 746, 744, 515, 467, 22, 884, 126, 395, 322, 854, 519, 100, 794, 814, 336, 543, 20, 944, 255, 137, 410, 164, 121, 1, 577, 83, 158, 377, 785, 527, 179, 555, 960, 889, 264, 528, 1016, 252, 503, 219, 109, 414, 619, 151, 297, 249, 461, 296, 159, 991, 796, 338, 356, 540, 579, 74, 832, 801, 94, 645, 532, 295, 987, 61, 234, 4, 553, 745, 958, 434, 586, 18, 506, 384, 111, 976, 658, 55, 301, 240, 182, 97, 483, 448, 463, 453, 502, 864, 662, 833, 482, 409, 982, 521, 385, 349, 21, 142, 60, 859, 947, 96, 330, 899, 265, 64, 362, 885, 312, 402, 343, 415, 420, 405, 93, 40, 281, 376, 549, 993, 494, 1011, 927, 705, 984, 462, 31, 480, 23, 283, 145, 699, 1013, 728, 611, 603, 701, 236, 112, 878, 508, 724, 795, 379, 454, 116, 912, 397, 818, 498, 13, 793, 934, 430, 393, 618, 666, 460, 867, 35, 828, 571, 221, 230, 416, 118, 621, 190, 973, 129, 486, 324, 266, 755, 513, 609, 830, 986, 531, 289, 257, 477, 357, 651, 835, 887, 325, 342, 195, 551, 428, 963, 424, 598, 908, 733, 504, 501, 517, 78, 114, 46, 478, 125, 258, 734, 717, 304, 522, 313, 250, 275, 169, 648, 320, 860, 536, 187, 597, 767, 123, 594, 924, 964, 589, 656, 892, 364, 143, 688, 59, 149, 491, 791, 588, 546, 857, 224, 84, 541, 694, 425, 38, 786, 581, 862, 659, 199, 916, 306, 897, 649, 368, 396, 742, 440, 305, 844, 967, 79, 858, 183, 906, 670, 369, 557, 470, 831, 174, 52, 365, 748, 787, 54, 697, 847, 825, 559, 484, 856, 1004, 466, 391, 902, 911, 599, 215, 739, 706, 935, 267, 933, 646, 808, 287, 723, 310, 68, 441, 317, 318, 1020, 613, 992, 773, 736, 429, 719, 202, 147, 905, 82, 629, 285, 605, 294, 253, 572, 449, 226, 981, 576, 732, 693, 499, 335, 926, 848, 217, 138, 170, 426, 753, 948, 642, 516, 17, 685, 227, 196, 168, 743, 245, 259, 929, 203, 155, 704, 880, 898, 861, 627, 122, 279, 574, 909, 804, 373, 569, 468, 712, 408, 538, 985, 43, 392, 713, 171, 418, 827, 971, 10, 630, 298, 570, 333, 163, 162, 99, 26, 95, 232, 946, 568, 73, 625, 635, 272, 211, 193, 602, 132, 641, 509, 809, 214, 57, 28, 45, 412, 950, 681, 383, 92, 979, 344, 66, 822, 49, 820, 106, 823, 834, 216, 382, 890, 842, 1012, 843, 669, 16, 188, 721, 526, 241, 708, 863, 778, 67, 85, 614, 698, 636, 692, 740, 696, 239, 130, 309, 189, 888, 1008, 632, 769, 995, 427, 0, 518, 398, 411, 337, 610, 774, 789, 869, 584, 616, 153, 799, 737, 176, 593, 378, 578, 679, 533, 711, 208, 271, 231, 530, 841, 113, 879, 951, 535, 949, 792, 623, 587, 612, 157, 784, 200, 263, 691, 77, 87, 671, 583, 308, 768, 922, 815, 988, 872, 824, 346, 940, 348, 510, 326, 622, 108, 633, 665, 821, 302, 282, 989, 65, 829, 980, 9, 442, 819, 495, 687, 810, 750, 439, 781, 27, 595, 942, 98, 270, 585, 886, 751, 438, 545, 286, 238, 654, 718, 421, 374, 381, 293, 91, 288, 222, 1019, 474, 966, 1022, 390, 738, 1017, 606, 972, 766, 191, 2, 690, 715, 754, 48, 256, 582], [809, 79, 104, 59, 15, 282, 214, 98, 340, 366, 925, 136, 342, 841, 85, 288, 784, 188, 598, 782, 208, 68, 571, 996, 863, 658, 979, 929, 632, 1013, 433, 1014, 866, 380, 443, 244, 842, 599, 954, 53, 689, 164, 759, 262, 566, 389, 693, 29, 796, 932, 554, 313, 637, 901, 917, 510, 119, 60, 520, 948, 283, 647, 281, 1015, 192, 427, 415, 792, 489, 4, 128, 919, 355, 429, 624, 882, 702, 326, 669, 17, 117, 950, 966, 148, 449, 920, 376, 760, 671, 563, 258, 405, 905, 123, 1004, 106, 776, 833, 965, 537, 556, 943, 190, 146, 879, 143, 601, 913, 903, 900, 42, 461, 310, 553, 421, 105, 981, 185, 672, 744, 548, 2, 220, 172, 73, 391, 416, 991, 334, 149, 444, 964, 163, 951, 818, 363, 1000, 544, 852, 202, 277, 785, 320, 466, 763, 663, 804, 167, 346, 938, 82, 89, 404, 716, 683, 872, 280, 351, 257, 381, 9, 846, 465, 171, 768, 686, 93, 634, 700, 451, 453, 112, 551, 490, 422, 666, 368, 971, 237, 893, 1011, 780, 657, 338, 695, 141, 140, 33, 442, 296, 91, 99, 786, 505, 159, 751, 246, 897, 256, 524, 675, 924, 988, 773, 696, 662, 746, 80, 904, 834, 752, 698, 873, 659, 706, 286, 406, 57, 195, 573, 101, 944, 521, 738, 608, 407, 912, 47, 701, 660, 594, 989, 586, 153, 399, 764, 110, 479, 1007, 687, 87, 132, 371, 550, 986, 704, 248, 173, 559, 218, 158, 204, 18, 58, 692, 799, 228, 349, 795, 165, 595, 997, 902, 6, 364, 459, 203, 875, 250, 7, 464, 498, 721, 906, 527, 668, 679, 881, 259, 963, 66, 1020, 990, 75, 499, 302, 354, 729, 621, 241, 25, 987, 933, 874, 767, 1018, 265, 558, 206, 691, 319, 130, 390, 791, 243, 862, 568, 239, 137, 70, 999, 529, 961, 3, 44, 14, 315, 166, 127, 888, 309, 487, 936, 409, 335, 129, 896, 1003, 37, 627, 279, 138, 650, 8, 561, 396, 967, 501, 254, 947, 956, 840, 67, 41, 370, 690, 519, 549, 565, 43, 590, 867, 145, 350, 307, 304, 276, 992, 438, 358, 538, 474, 710, 640, 124, 252, 585, 432, 781, 891, 821, 76, 186, 835, 419, 189, 539, 824, 707, 743, 231, 348, 1006, 234, 31, 402, 201, 298, 885, 894, 620, 318, 730, 778, 533, 980, 269, 545, 546, 225, 569, 769, 1012, 673, 579, 426, 285, 619, 685, 578, 431, 62, 810, 468, 534, 765, 654, 495, 1016, 790, 22, 611, 516, 512, 290, 316, 229, 242, 383, 38, 245, 531, 960, 688, 982, 591, 430, 352, 294, 923, 374, 596, 367, 435, 670, 437, 857, 848, 597, 473, 323, 831, 797, 492, 161, 774, 154, 908, 580, 581, 382, 779, 884, 845, 219, 332, 215, 247, 217, 275, 240, 337, 775, 998, 120, 469, 642, 198, 757, 263, 333, 974, 541, 664, 515, 783, 452, 887, 373, 851, 156, 532, 388, 928, 121, 291, 536, 48, 423, 27, 40, 870, 656, 661, 635, 976, 582, 1023, 1010, 418, 684, 934, 84, 470, 341, 714, 617, 183, 86, 864, 699, 506, 488, 35, 915, 211, 523, 747, 836, 260, 847, 772, 816, 889, 379, 820, 646, 606, 911, 322, 942, 742, 969, 102, 205, 507, 822, 497, 793, 249, 460, 10, 825, 854, 712, 397, 457, 957, 72, 651, 631, 456, 209, 400, 677, 528, 97, 446, 403, 170, 517, 134, 713, 622, 613, 238, 147, 178, 755, 564, 233, 100, 414, 475, 12, 725, 157, 476, 197, 274, 401, 278, 588, 945, 994, 719, 289, 264, 439, 378, 212, 593, 972, 827, 718, 94, 600, 267, 328, 968, 653, 828, 56, 271, 844, 395, 955, 724, 324, 723, 860, 440, 224, 859, 301, 717, 823, 808, 196, 643, 300, 19, 615, 973, 939, 993, 434, 330, 576, 471, 504, 970, 626, 907, 526, 812, 735, 762, 467, 733, 226, 16, 478, 417, 24, 107, 1005, 500, 255, 55, 193, 168, 216, 577, 892, 734, 96, 880, 722, 815, 221, 1, 522, 361, 175, 169, 92, 336, 65, 737, 34, 393, 266, 603, 535, 813, 604, 52, 926, 152, 1009, 133, 392, 329, 946, 602, 227, 711, 766, 826, 644, 150, 482, 514, 384, 491, 610, 789, 829, 984, 909, 398, 648, 959, 525, 856, 345, 387, 895, 962, 213, 312, 727, 347, 931, 748, 865, 876, 557, 641, 720, 199, 861, 179, 1017, 503, 425, 667, 749, 800, 732, 299, 843, 36, 803, 694, 937, 365, 445, 484, 494, 543, 118, 412, 305, 540, 958, 223, 64, 858, 375, 839, 883, 935, 187, 899, 144, 921, 210, 113, 518, 575, 806, 369, 552, 311, 614, 28, 287, 455, 420, 270, 890, 628, 513, 177, 916, 343, 116, 817, 741, 13, 253, 321, 486, 230, 802, 502, 740, 207, 592, 481, 69, 306, 777, 496, 385, 995, 855, 39, 413, 930, 625, 317, 910, 607, 297, 78, 428, 949, 728, 649, 739, 292, 359, 754, 0, 356, 472, 567, 697, 308, 180, 807, 232, 410, 174, 377, 477, 952, 940, 941, 918, 222, 77, 109, 463, 344, 680, 1001, 726, 45, 125, 616, 115, 715, 273, 447, 914, 284, 191, 160, 111, 386, 295, 811, 983, 636, 511, 454, 927, 293, 139, 612, 11, 530, 20, 788, 801, 542, 587, 61, 5, 303, 184, 572, 151, 162, 753, 142, 95, 51, 871, 850, 448, 1021, 709, 235, 251, 869, 830, 639, 71, 562, 758, 339, 805, 88, 509, 731, 314, 182, 394, 665, 268, 493, 838, 272, 630, 629, 83, 194, 953, 360, 108, 886, 819, 236, 1022, 977, 103, 436, 633, 814, 798, 450, 176, 705, 770, 26, 555, 408, 703, 411, 674, 325, 21, 181, 638, 832, 645, 131, 1002, 63, 574, 261, 584, 23, 750, 678, 898, 560, 30, 49, 74, 570, 652, 90, 853, 480, 975, 609, 794, 485, 508, 32, 135, 122, 50, 483, 362, 54, 126, 589, 978, 756, 985, 849, 682, 761, 837, 441, 327, 623, 114, 458, 922, 1008, 1019, 771, 787, 676, 331, 655, 618, 81, 681, 357, 583, 424, 877, 605, 745, 736, 200, 372, 353, 708, 46, 462, 155, 868, 547, 878], [35, 76, 956, 1016, 920, 417, 242, 114, 564, 372, 803, 565, 73, 434, 744, 892, 812, 365, 681, 861, 303, 403, 504, 749, 734, 591, 1, 717, 942, 185, 730, 326, 693, 368, 581, 806, 543, 259, 155, 530, 246, 822, 552, 2, 811, 705, 872, 382, 291, 67, 354, 436, 290, 659, 361, 169, 549, 944, 393, 470, 934, 20, 774, 800, 237, 575, 701, 886, 558, 733, 792, 513, 17, 112, 78, 851, 220, 53, 704, 170, 674, 569, 299, 31, 41, 124, 731, 780, 58, 570, 957, 166, 497, 503, 984, 487, 592, 567, 758, 783, 606, 947, 849, 657, 68, 612, 1021, 648, 201, 951, 257, 596, 879, 1000, 974, 437, 142, 390, 130, 735, 815, 473, 864, 405, 121, 223, 626, 336, 118, 331, 275, 712, 215, 711, 455, 449, 667, 253, 70, 482, 794, 998, 903, 891, 483, 1023, 598, 635, 571, 611, 241, 141, 379, 577, 460, 106, 665, 1014, 93, 952, 355, 629, 156, 638, 360, 399, 346, 590, 686, 77, 875, 779, 340, 1018, 502, 152, 54, 145, 989, 469, 478, 997, 321, 25, 14, 709, 677, 708, 721, 807, 172, 39, 140, 102, 1020, 202, 535, 509, 656, 645, 198, 979, 727, 553, 284, 6, 8, 277, 928, 75, 472, 1003, 7, 1019, 150, 820, 534, 1010, 870, 831, 268, 669, 225, 154, 420, 461, 832, 33, 651, 492, 936, 761, 276, 765, 397, 308, 556, 95, 525, 887, 194, 955, 814, 263, 747, 697, 776, 248, 834, 190, 639, 280, 108, 732, 24, 330, 742, 576, 540, 877, 678, 468, 174, 175, 446, 836, 216, 307, 912, 341, 728, 935, 616, 440, 798, 848, 48, 691, 574, 608, 830, 138, 842, 101, 617, 171, 725, 863, 790, 466, 661, 786, 862, 343, 931, 69, 12, 465, 937, 94, 739, 531, 119, 573, 810, 238, 103, 302, 852, 406, 789, 426, 164, 371, 965, 769, 716, 554, 462, 1015, 213, 945, 882, 788, 737, 61, 748, 696, 658, 799, 924, 640, 312, 986, 412, 718, 690, 760, 781, 234, 724, 52, 261, 480, 325, 345, 624, 896, 274, 488, 282, 301, 895, 37, 463, 10, 1022, 15, 44, 759, 839, 146, 866, 421, 929, 642, 107, 793, 245, 726, 889, 664, 162, 43, 244, 221, 785, 771, 358, 595, 294, 871, 702, 415, 1013, 547, 29, 985, 381, 675, 177, 663, 593, 881, 408, 181, 447, 655, 178, 271, 218, 205, 740, 544, 319, 505, 90, 620, 384, 878, 364, 60, 561, 210, 314, 557, 414, 529, 182, 485, 3, 980, 641, 964, 1001, 967, 506, 521, 63, 625, 28, 18, 706, 80, 668, 840, 906, 484, 900, 389, 514, 713, 413, 698, 42, 441, 320, 910, 116, 310, 634, 306, 334, 1006, 976, 448, 853, 579, 827, 685, 83, 474, 627, 795, 84, 918, 729, 59, 680, 192, 824, 586, 623, 741, 860, 1012, 526, 411, 894, 147, 51, 987, 927, 115, 893, 409, 819, 960, 219, 843, 407, 260, 21, 64, 197, 660, 62, 823, 904, 383, 695, 555, 666, 572, 428, 366, 1009, 784, 349, 930, 992, 458, 962, 703, 262, 396, 316, 490, 195, 865, 493, 137, 768, 850, 293, 49, 297, 855, 808, 444, 149, 427, 915, 953, 973, 835, 818, 328, 32, 897, 752, 252, 916, 524, 5, 433, 520, 548, 688, 613, 805, 825, 313, 692, 723, 481, 580, 332, 1007, 684, 431, 1008, 943, 847, 342, 151, 189, 615, 443, 50, 256, 498, 687, 250, 885, 772, 369, 837, 494, 196, 983, 630, 619, 545, 159, 813, 40, 333, 180, 392, 423, 229, 231, 908, 222, 283, 751, 1017, 46, 601, 135, 585, 87, 45, 311, 72, 542, 117, 991, 652, 683, 527, 846, 451, 71, 618, 457, 736, 856, 884, 442, 536, 938, 66, 694, 816, 19, 376, 767, 995, 491, 23, 387, 89, 949, 92, 4, 134, 367, 183, 128, 34, 926, 958, 200, 500, 738, 56, 425, 628, 710, 932, 770, 778, 1004, 335, 438, 804, 143, 144, 972, 363, 753, 353, 120, 775, 969, 826, 86, 817, 125, 512, 129, 350, 47, 452, 911, 802, 289, 977, 281, 646, 424, 689, 278, 888, 966, 131, 880, 99, 22, 940, 158, 876, 968, 594, 750, 173, 933, 11, 239, 193, 479, 538, 941, 82, 163, 65, 212, 988, 357, 631, 467, 700, 160, 111, 999, 139, 295, 869, 217, 649, 348, 404, 589, 247, 859, 978, 429, 495, 563, 867, 921, 746, 203, 632, 994, 588, 773, 9, 435, 351, 653, 539, 286, 395, 337, 551, 971, 98, 288, 838, 913, 273, 670, 401, 380, 743, 603, 453, 1005, 232, 133, 609, 267, 643, 206, 600, 214, 925, 873, 227, 559, 329, 755, 550, 496, 511, 909, 854, 602, 671, 30, 475, 990, 74, 829, 923, 801, 797, 157, 352, 388, 699, 287, 917, 796, 258, 347, 754, 719, 501, 398, 243, 165, 902, 621, 489, 981, 975, 841, 787, 191, 338, 188, 516, 597, 418, 391, 654, 153, 105, 374, 279, 419, 439, 883, 450, 1011, 269, 993, 240, 560, 110, 104, 13, 610, 662, 982, 584, 394, 828, 507, 402, 566, 432, 607, 508, 26, 523, 950, 57, 126, 676, 184, 168, 636, 959, 939, 123, 583, 517, 757, 324, 1002, 199, 707, 599, 167, 375, 292, 265, 211, 970, 204, 16, 605, 791, 476, 673, 519, 318, 176, 207, 235, 362, 714, 385, 322, 715, 946, 486, 327, 948, 285, 370, 38, 251, 315, 226, 464, 377, 647, 373, 430, 919, 587, 954, 233, 91, 317, 454, 228, 97, 96, 522, 633, 582, 356, 0, 614, 874, 296, 422, 122, 88, 845, 672, 907, 266, 541, 899, 833, 562, 224, 127, 868, 410, 148, 764, 898, 132, 858, 36, 209, 179, 922, 857, 756, 844, 762, 85, 477, 254, 309, 208, 400, 782, 622, 604, 637, 682, 578, 81, 416, 963, 339, 763, 304, 186, 996, 456, 305, 499, 323, 272, 905, 100, 533, 113, 568, 187, 27, 230, 359, 537, 518, 961, 528, 79, 270, 679, 459, 914, 378, 644, 766, 809, 161, 344, 236, 510, 55, 109, 298, 777, 445, 515, 249, 890, 471, 386, 300, 722, 821, 546, 255, 720, 901, 745, 136, 264, 532, 650]], [[857, 448, 146, 873, 164, 1004, 18, 240, 287, 680, 341, 498, 962, 64, 573, 191, 281, 782, 781, 623, 808, 546, 63, 209, 280, 435, 986, 965, 278, 1006, 41, 976, 855, 422, 829, 728, 404, 686, 859, 533, 802, 67, 258, 814, 348, 852, 219, 127, 190, 326, 1014, 115, 206, 39, 545, 493, 619, 839, 0, 125, 716, 678, 898, 999, 713, 178, 793, 61, 151, 328, 259, 195, 996, 327, 516, 94, 894, 423, 45, 881, 431, 71, 269, 910, 797, 106, 156, 655, 345, 1023, 351, 432, 480, 497, 420, 318, 732, 128, 42, 392, 366, 640, 515, 510, 166, 564, 931, 336, 756, 708, 742, 251, 334, 97, 346, 576, 740, 815, 324, 356, 463, 652, 882, 920, 991, 776, 750, 255, 339, 791, 668, 637, 96, 122, 362, 303, 660, 818, 760, 124, 244, 866, 702, 747, 246, 225, 563, 704, 183, 761, 993, 995, 372, 571, 487, 933, 437, 847, 439, 298, 313, 393, 182, 707, 202, 95, 975, 826, 759, 874, 690, 647, 179, 353, 701, 441, 290, 768, 478, 194, 845, 274, 603, 530, 83, 661, 185, 320, 33, 505, 227, 134, 745, 1010, 1011, 987, 648, 551, 150, 10, 381, 288, 231, 517, 557, 201, 715, 503, 903, 17, 491, 961, 561, 871, 495, 36, 76, 234, 953, 558, 289, 694, 160, 200, 264, 830, 282, 621, 785, 525, 37, 484, 333, 538, 410, 502, 654, 985, 566, 997, 479, 940, 86, 205, 14, 337, 325, 38, 489, 562, 338, 880, 27, 454, 123, 567, 488, 449, 91, 261, 47, 1012, 616, 172, 572, 942, 243, 506, 330, 13, 705, 335, 807, 501, 688, 672, 120, 197, 1002, 60, 876, 1005, 589, 226, 374, 853, 842, 542, 767, 394, 142, 684, 105, 415, 758, 48, 950, 537, 21, 267, 459, 959, 248, 955, 464, 375, 541, 236, 779, 310, 188, 553, 629, 595, 369, 59, 683, 659, 964, 540, 809, 147, 77, 272, 631, 192, 419, 978, 344, 597, 43, 466, 899, 22, 549, 119, 388, 945, 752, 610, 944, 307, 152, 89, 470, 365, 712, 977, 636, 865, 811, 239, 317, 204, 438, 764, 161, 268, 101, 301, 657, 912, 645, 118, 884, 718, 68, 520, 28, 817, 872, 696, 75, 787, 137, 514, 868, 368, 583, 275, 213, 722, 867, 607, 1016, 612, 642, 52, 418, 233, 476, 883, 878, 425, 56, 739, 11, 772, 921, 521, 417, 475, 843, 892, 9, 719, 890, 302, 1013, 726, 184, 569, 483, 263, 1001, 744, 51, 771, 103, 956, 943, 602, 615, 596, 665, 1021, 812, 593, 1019, 26, 136, 1003, 90, 402, 292, 145, 790, 509, 99, 960, 490, 620, 352, 7, 658, 426, 84, 824, 252, 832, 66, 253, 98, 851, 416, 135, 400, 692, 736, 879, 577, 245, 297, 363, 687, 407, 457, 399, 265, 801, 235, 173, 315, 193, 141, 775, 316, 691, 780, 983, 897, 714, 548, 783, 733, 108, 669, 765, 238, 598, 169, 299, 695, 825, 170, 579, 347, 656, 65, 628, 904, 4, 528, 1, 294, 30, 709, 786, 800, 260, 673, 524, 456, 228, 743, 143, 580, 924, 100, 828, 923, 496, 241, 413, 856, 762, 485, 532, 662, 343, 157, 559, 72, 594, 436, 611, 864, 917, 405, 840, 46, 395, 922, 176, 81, 536, 421, 721, 398, 387, 6, 901, 891, 751, 932, 323, 918, 44, 314, 938, 198, 984, 465, 989, 900, 217, 870, 270, 757, 210, 973, 908, 242, 582, 8, 471, 247, 446, 796, 633, 930, 163, 895, 443, 748, 408, 639, 952, 112, 155, 587, 162, 296, 171, 3, 460, 216, 109, 585, 885, 821, 208, 896, 175, 104, 774, 916, 224, 727, 285, 414, 534, 925, 590, 322, 212, 181, 409, 87, 131, 822, 174, 379, 635, 560, 838, 2, 833, 792, 934, 110, 697, 1009, 877, 794, 755, 608, 947, 5, 237, 737, 451, 601, 969, 979, 384, 609, 53, 308, 937, 504, 380, 1007, 670, 613, 703, 458, 85, 711, 909, 556, 513, 725, 199, 677, 698, 552, 158, 474, 663, 958, 928, 606, 675, 632, 138, 230, 508, 875, 273, 795, 622, 450, 492, 906, 349, 440, 312, 321, 992, 218, 889, 913, 370, 599, 763, 472, 1022, 29, 284, 820, 69, 749, 586, 848, 915, 250, 837, 113, 133, 769, 634, 666, 957, 377, 886, 54, 340, 373, 447, 232, 554, 481, 82, 371, 649, 982, 893, 988, 50, 681, 167, 789, 19, 860, 397, 550, 618, 256, 1015, 331, 676, 518, 731, 592, 342, 214, 523, 257, 679, 511, 735, 919, 382, 168, 777, 406, 664, 177, 453, 936, 433, 994, 186, 121, 717, 625, 816, 376, 20, 741, 391, 359, 784, 107, 262, 34, 581, 93, 383, 429, 249, 445, 62, 798, 941, 88, 102, 970, 574, 823, 411, 40, 424, 500, 905, 914, 222, 766, 526, 980, 24, 305, 565, 555, 887, 300, 819, 211, 835, 671, 468, 254, 144, 401, 963, 295, 111, 396, 32, 911, 196, 951, 507, 575, 723, 674, 861, 653, 57, 55, 706, 1017, 729, 139, 477, 949, 1000, 803, 946, 467, 539, 834, 319, 469, 981, 1020, 617, 189, 547, 998, 543, 519, 203, 605, 354, 850, 773, 966, 902, 92, 849, 79, 693, 25, 364, 180, 806, 271, 132, 685, 831, 720, 644, 954, 360, 734, 461, 544, 283, 624, 117, 869, 630, 863, 23, 651, 499, 389, 724, 972, 844, 710, 78, 682, 73, 1018, 311, 588, 990, 412, 486, 220, 535, 827, 846, 646, 154, 74, 1008, 159, 888, 638, 788, 754, 15, 277, 16, 215, 355, 442, 746, 165, 390, 129, 462, 444, 810, 627, 604, 58, 378, 286, 836, 578, 148, 221, 49, 948, 512, 805, 967, 854, 494, 293, 149, 813, 730, 935, 361, 427, 968, 31, 207, 529, 770, 570, 114, 229, 939, 130, 329, 140, 643, 304, 929, 126, 276, 434, 799, 358, 430, 473, 614, 70, 804, 482, 386, 385, 455, 187, 841, 974, 907, 291, 116, 332, 862, 452, 428, 689, 700, 626, 699, 778, 753, 357, 738, 309, 584, 12, 568, 367, 926, 600, 522, 927, 641, 266, 306, 403, 527, 350, 667, 531, 223, 858, 591, 650, 279, 971, 35, 153, 80], [953, 284, 45, 799, 783, 151, 565, 264, 664, 956, 640, 551, 804, 784, 35, 732, 699, 289, 845, 171, 713, 229, 1007, 239, 135, 569, 635, 314, 362, 955, 410, 829, 165, 446, 572, 930, 395, 499, 814, 103, 587, 234, 478, 345, 943, 192, 216, 312, 430, 847, 604, 708, 562, 632, 583, 116, 824, 821, 211, 175, 275, 162, 514, 268, 456, 858, 964, 501, 379, 767, 101, 907, 864, 657, 722, 87, 547, 585, 744, 355, 285, 85, 683, 576, 1017, 510, 361, 593, 887, 665, 999, 393, 652, 244, 865, 861, 602, 752, 626, 867, 603, 235, 860, 450, 327, 88, 899, 644, 833, 728, 47, 487, 508, 715, 558, 349, 788, 637, 68, 352, 786, 866, 500, 53, 144, 131, 1003, 122, 654, 432, 161, 474, 482, 178, 613, 222, 525, 435, 986, 334, 102, 339, 638, 653, 535, 8, 988, 857, 392, 157, 471, 911, 203, 531, 816, 890, 406, 509, 932, 764, 609, 36, 982, 150, 483, 44, 407, 633, 848, 782, 299, 249, 773, 220, 366, 371, 421, 297, 464, 906, 791, 620, 1010, 1000, 817, 93, 80, 838, 586, 400, 305, 197, 869, 716, 424, 903, 801, 106, 383, 11, 477, 753, 846, 532, 737, 287, 276, 468, 476, 574, 412, 888, 771, 413, 1019, 420, 183, 774, 840, 213, 74, 915, 650, 429, 245, 28, 874, 120, 294, 742, 1013, 819, 319, 940, 329, 129, 779, 978, 969, 755, 180, 948, 941, 288, 679, 594, 704, 274, 258, 792, 714, 825, 403, 734, 896, 109, 540, 968, 717, 33, 246, 219, 949, 554, 281, 326, 839, 321, 260, 820, 636, 556, 226, 619, 893, 862, 830, 230, 931, 251, 810, 641, 114, 140, 207, 961, 697, 17, 736, 323, 598, 966, 544, 895, 663, 997, 457, 187, 507, 194, 15, 270, 703, 751, 823, 121, 646, 523, 69, 629, 164, 648, 1008, 618, 985, 210, 980, 55, 39, 625, 947, 994, 522, 454, 272, 748, 317, 762, 43, 470, 859, 427, 265, 469, 897, 19, 479, 365, 972, 257, 504, 434, 520, 123, 37, 541, 300, 957, 588, 1002, 336, 684, 645, 923, 467, 146, 356, 1015, 739, 962, 89, 458, 405, 308, 350, 526, 452, 23, 99, 528, 974, 575, 218, 749, 199, 108, 304, 416, 449, 623, 605, 770, 0, 29, 673, 243, 489, 168, 295, 12, 721, 600, 423, 984, 61, 481, 975, 666, 145, 928, 939, 292, 979, 660, 426, 1011, 67, 512, 137, 1018, 81, 904, 524, 437, 992, 315, 70, 170, 695, 462, 149, 376, 160, 167, 360, 130, 919, 872, 777, 27, 26, 989, 404, 747, 386, 534, 134, 152, 891, 1021, 730, 359, 465, 142, 835, 624, 542, 215, 402, 1014, 253, 621, 325, 811, 370, 1001, 367, 291, 84, 552, 227, 826, 104, 527, 40, 293, 182, 918, 331, 225, 963, 209, 563, 718, 273, 442, 991, 94, 681, 401, 787, 174, 9, 64, 205, 46, 639, 79, 851, 128, 564, 124, 112, 855, 958, 519, 132, 741, 313, 1012, 680, 195, 316, 381, 667, 141, 622, 495, 794, 490, 505, 803, 384, 76, 72, 1005, 995, 409, 954, 760, 223, 929, 884, 606, 676, 448, 50, 608, 24, 221, 908, 832, 269, 310, 441, 1022, 879, 698, 296, 517, 388, 971, 905, 883, 705, 942, 852, 398, 475, 202, 254, 643, 577, 671, 917, 827, 970, 354, 881, 189, 320, 601, 546, 425, 252, 115, 729, 440, 364, 579, 719, 328, 105, 208, 768, 780, 599, 6, 693, 926, 878, 156, 417, 1023, 397, 607, 447, 894, 186, 228, 335, 658, 647, 757, 945, 937, 224, 486, 473, 597, 785, 455, 553, 303, 18, 357, 496, 871, 578, 615, 723, 433, 843, 32, 436, 836, 682, 959, 726, 57, 181, 694, 761, 677, 924, 453, 62, 515, 649, 48, 746, 805, 568, 492, 340, 91, 22, 614, 212, 973, 385, 655, 1004, 548, 263, 537, 571, 7, 738, 117, 66, 444, 290, 822, 346, 936, 691, 550, 282, 533, 242, 516, 634, 561, 659, 993, 60, 155, 776, 506, 628, 740, 459, 617, 996, 651, 13, 311, 71, 463, 710, 612, 690, 557, 763, 511, 422, 849, 201, 301, 581, 78, 163, 380, 185, 338, 198, 580, 701, 56, 214, 484, 309, 498, 466, 177, 204, 536, 900, 745, 702, 539, 241, 812, 337, 828, 217, 126, 176, 95, 3, 815, 238, 831, 922, 727, 4, 41, 21, 374, 414, 179, 592, 772, 113, 886, 560, 172, 306, 983, 946, 692, 347, 596, 2, 798, 368, 237, 797, 950, 280, 92, 591, 750, 322, 390, 51, 77, 351, 394, 841, 707, 491, 775, 802, 378, 674, 485, 850, 898, 480, 754, 927, 302, 438, 382, 960, 65, 987, 256, 590, 279, 567, 408, 800, 853, 981, 793, 232, 582, 870, 138, 913, 58, 662, 233, 266, 97, 910, 153, 952, 689, 502, 262, 977, 196, 231, 731, 497, 844, 377, 688, 14, 711, 538, 925, 439, 573, 428, 159, 670, 912, 493, 460, 431, 261, 503, 318, 630, 967, 75, 765, 876, 877, 998, 389, 25, 332, 901, 31, 724, 687, 52, 559, 549, 733, 672, 147, 86, 59, 286, 277, 808, 119, 250, 885, 240, 111, 914, 396, 169, 98, 278, 173, 555, 49, 324, 902, 38, 669, 1016, 807, 611, 461, 709, 127, 166, 834, 566, 247, 920, 342, 154, 696, 494, 589, 756, 10, 298, 206, 795, 545, 107, 372, 158, 348, 610, 668, 73, 818, 190, 661, 916, 255, 267, 1, 200, 875, 806, 725, 419, 248, 656, 83, 758, 938, 766, 584, 1009, 675, 543, 882, 518, 415, 513, 283, 133, 90, 343, 909, 391, 373, 488, 743, 856, 921, 193, 411, 530, 735, 307, 63, 30, 642, 333, 271, 631, 889, 191, 796, 42, 1020, 139, 399, 34, 521, 148, 344, 863, 769, 809, 595, 933, 837, 236, 712, 935, 965, 1006, 778, 5, 678, 873, 369, 418, 880, 472, 759, 54, 789, 188, 790, 443, 143, 16, 341, 813, 136, 892, 781, 944, 627, 118, 100, 951, 934, 685, 842, 259, 96, 854, 445, 20, 353, 451, 387, 616, 700, 868, 706, 330, 529, 125, 720, 686, 110, 990, 363, 184, 358, 976, 375, 570, 82], [334, 732, 796, 198, 299, 859, 339, 884, 254, 534, 205, 107, 293, 9, 532, 692, 477, 961, 781, 31, 509, 490, 841, 950, 937, 521, 505, 470, 771, 541, 595, 768, 756, 465, 814, 946, 746, 115, 120, 907, 831, 287, 507, 895, 812, 340, 663, 258, 782, 745, 199, 697, 752, 975, 514, 853, 34, 1007, 38, 649, 721, 811, 371, 551, 11, 574, 673, 367, 318, 992, 46, 787, 250, 833, 954, 27, 614, 499, 412, 161, 847, 685, 635, 1018, 428, 618, 687, 931, 976, 313, 476, 684, 253, 657, 702, 60, 977, 389, 502, 829, 607, 390, 888, 124, 648, 303, 1003, 998, 481, 133, 294, 381, 834, 666, 589, 8, 442, 594, 757, 520, 699, 951, 53, 760, 45, 933, 497, 464, 379, 433, 851, 845, 794, 891, 429, 517, 231, 311, 202, 553, 44, 280, 805, 399, 354, 445, 83, 1008, 153, 123, 1000, 366, 858, 149, 604, 41, 972, 619, 300, 91, 118, 628, 865, 150, 172, 70, 19, 127, 939, 806, 578, 718, 345, 713, 176, 994, 151, 789, 945, 219, 803, 821, 145, 980, 228, 132, 307, 295, 795, 556, 849, 584, 382, 546, 872, 177, 35, 3, 988, 651, 385, 450, 187, 400, 669, 747, 2, 952, 686, 536, 705, 395, 85, 881, 131, 443, 613, 680, 82, 96, 291, 185, 723, 897, 424, 411, 751, 272, 266, 548, 69, 72, 409, 353, 108, 417, 625, 168, 722, 396, 279, 343, 80, 237, 225, 615, 634, 117, 890, 902, 102, 838, 125, 867, 901, 922, 179, 783, 587, 49, 444, 978, 936, 744, 419, 957, 797, 413, 380, 480, 659, 862, 410, 855, 109, 969, 999, 189, 317, 800, 336, 871, 372, 200, 148, 769, 828, 786, 224, 824, 23, 249, 425, 66, 36, 569, 74, 203, 208, 868, 903, 924, 777, 762, 830, 827, 348, 759, 816, 791, 966, 1012, 660, 30, 930, 886, 403, 194, 89, 929, 968, 384, 822, 904, 558, 719, 836, 274, 616, 870, 887, 645, 448, 241, 106, 983, 662, 77, 706, 501, 898, 316, 233, 344, 227, 488, 506, 538, 819, 229, 982, 861, 600, 960, 286, 78, 452, 739, 281, 431, 188, 248, 335, 942, 182, 359, 591, 47, 333, 405, 215, 668, 302, 277, 273, 166, 516, 562, 422, 644, 720, 661, 575, 278, 1022, 216, 547, 940, 693, 472, 627, 798, 630, 455, 134, 537, 923, 268, 171, 1013, 356, 712, 643, 262, 709, 28, 598, 438, 467, 456, 776, 733, 259, 629, 646, 139, 893, 207, 892, 142, 90, 432, 991, 20, 793, 232, 479, 593, 330, 974, 876, 461, 585, 483, 953, 304, 711, 580, 679, 941, 446, 246, 572, 296, 270, 588, 638, 734, 350, 738, 48, 135, 698, 843, 68, 802, 708, 1017, 122, 938, 701, 421, 586, 963, 158, 98, 565, 213, 110, 624, 257, 484, 94, 191, 724, 528, 743, 469, 540, 605, 642, 885, 676, 10, 136, 331, 175, 93, 485, 17, 504, 126, 696, 160, 146, 56, 576, 95, 990, 377, 564, 622, 498, 491, 526, 934, 236, 247, 1009, 860, 513, 128, 55, 844, 164, 436, 87, 854, 935, 42, 549, 856, 156, 842, 454, 571, 695, 81, 478, 404, 269, 458, 647, 804, 1006, 665, 292, 971, 970, 435, 114, 338, 104, 99, 682, 54, 560, 178, 877, 799, 808, 341, 984, 397, 837, 801, 50, 671, 780, 573, 64, 962, 408, 889, 914, 84, 325, 577, 691, 130, 1016, 773, 192, 256, 210, 815, 727, 283, 920, 563, 423, 360, 582, 103, 740, 766, 137, 6, 271, 73, 155, 144, 623, 201, 43, 567, 925, 943, 51, 52, 690, 473, 59, 543, 948, 306, 391, 169, 911, 985, 374, 512, 468, 726, 670, 653, 875, 494, 457, 947, 180, 818, 414, 387, 165, 896, 352, 100, 37, 290, 407, 683, 364, 209, 5, 267, 392, 324, 1011, 471, 86, 260, 181, 159, 346, 761, 402, 987, 906, 703, 840, 725, 319, 430, 612, 1002, 882, 932, 765, 850, 147, 503, 531, 530, 611, 519, 857, 475, 361, 79, 58, 949, 792, 742, 275, 917, 728, 704, 852, 57, 434, 141, 365, 451, 523, 39, 315, 218, 116, 71, 894, 320, 369, 717, 919, 323, 15, 244, 749, 915, 33, 332, 370, 813, 869, 65, 276, 406, 633, 736, 416, 453, 810, 119, 700, 314, 583, 656, 995, 784, 608, 463, 1010, 326, 230, 993, 447, 621, 508, 239, 297, 26, 544, 617, 729, 298, 832, 688, 222, 383, 1021, 170, 466, 105, 492, 599, 18, 220, 355, 101, 775, 597, 193, 652, 846, 195, 212, 24, 525, 92, 162, 0, 163, 809, 420, 441, 678, 234, 328, 474, 522, 772, 658, 748, 75, 767, 157, 329, 284, 790, 29, 40, 905, 835, 265, 310, 626, 996, 22, 252, 289, 337, 602, 493, 913, 631, 427, 495, 88, 866, 221, 394, 489, 707, 401, 650, 1023, 111, 1004, 710, 883, 140, 14, 235, 735, 154, 527, 609, 288, 217, 190, 449, 579, 639, 375, 542, 62, 386, 566, 965, 4, 240, 255, 533, 958, 986, 226, 183, 737, 529, 973, 535, 437, 321, 368, 879, 305, 674, 243, 426, 282, 112, 731, 632, 61, 596, 173, 864, 770, 823, 826, 873, 561, 559, 664, 1005, 820, 667, 376, 921, 755, 758, 848, 13, 308, 928, 1001, 1, 393, 550, 785, 25, 214, 1019, 640, 462, 681, 511, 440, 956, 515, 675, 21, 967, 357, 378, 825, 76, 285, 351, 817, 113, 944, 312, 63, 741, 989, 152, 309, 764, 500, 655, 245, 486, 779, 439, 545, 715, 654, 900, 197, 12, 67, 636, 754, 557, 788, 778, 568, 620, 839, 908, 415, 926, 910, 774, 730, 496, 167, 603, 880, 641, 570, 863, 981, 997, 358, 606, 1014, 714, 242, 763, 373, 592, 362, 347, 552, 460, 750, 186, 174, 899, 264, 955, 261, 716, 524, 637, 238, 874, 418, 251, 672, 32, 301, 204, 694, 959, 909, 398, 807, 138, 878, 16, 610, 916, 1020, 927, 363, 121, 554, 581, 349, 482, 590, 912, 689, 129, 510, 143, 322, 601, 555, 7, 518, 342, 263, 459, 677, 327, 539, 196, 223, 1015, 206, 211, 918, 184, 964, 487, 979, 753, 97, 388], [675, 594, 642, 776, 676, 866, 946, 947, 979, 912, 668, 449, 581, 29, 47, 825, 702, 486, 414, 246, 126, 717, 932, 530, 206, 891, 494, 885, 307, 679, 875, 8, 692, 19, 548, 90, 30, 113, 764, 485, 927, 829, 303, 712, 859, 740, 434, 223, 805, 478, 888, 647, 708, 282, 582, 492, 761, 653, 906, 455, 342, 121, 72, 1022, 137, 882, 265, 933, 952, 410, 240, 739, 539, 453, 468, 792, 976, 955, 195, 534, 627, 46, 801, 657, 376, 184, 987, 45, 571, 931, 37, 487, 301, 919, 763, 758, 1003, 353, 474, 70, 441, 42, 201, 470, 198, 908, 465, 242, 499, 250, 514, 381, 94, 767, 178, 214, 715, 263, 231, 85, 706, 53, 515, 957, 262, 830, 551, 269, 1018, 856, 928, 500, 972, 603, 839, 328, 646, 851, 152, 901, 575, 64, 483, 352, 259, 129, 89, 540, 685, 894, 609, 910, 503, 630, 923, 893, 511, 926, 992, 911, 622, 962, 556, 956, 532, 566, 62, 524, 719, 187, 1016, 755, 920, 612, 302, 341, 589, 844, 824, 869, 135, 984, 509, 531, 606, 102, 234, 254, 236, 26, 248, 577, 54, 197, 264, 375, 0, 726, 25, 529, 789, 645, 365, 536, 769, 354, 141, 180, 1012, 145, 858, 838, 899, 671, 700, 871, 36, 861, 391, 533, 535, 368, 237, 517, 66, 378, 205, 724, 576, 157, 843, 733, 672, 836, 9, 542, 704, 889, 513, 247, 495, 553, 454, 522, 458, 770, 669, 104, 771, 634, 2, 371, 666, 118, 691, 754, 325, 257, 680, 631, 439, 459, 74, 109, 746, 978, 804, 323, 473, 442, 271, 501, 788, 902, 78, 219, 143, 482, 549, 982, 898, 890, 683, 336, 161, 366, 239, 1021, 699, 169, 44, 392, 117, 253, 149, 684, 114, 735, 608, 60, 736, 822, 787, 784, 210, 110, 466, 356, 593, 182, 568, 437, 643, 430, 578, 418, 1000, 274, 444, 855, 150, 167, 252, 461, 260, 817, 811, 103, 359, 385, 426, 892, 270, 452, 688, 20, 604, 373, 727, 921, 61, 916, 565, 96, 125, 797, 543, 88, 1015, 526, 613, 79, 780, 343, 816, 743, 588, 423, 344, 930, 456, 802, 934, 820, 76, 636, 221, 181, 812, 694, 258, 834, 488, 1020, 960, 1005, 355, 163, 633, 504, 268, 330, 11, 728, 87, 15, 409, 415, 7, 417, 716, 939, 272, 572, 985, 106, 996, 283, 614, 477, 277, 766, 682, 164, 860, 537, 584, 687, 457, 448, 166, 116, 393, 587, 476, 389, 925, 884, 276, 815, 173, 399, 827, 907, 809, 841, 115, 95, 408, 10, 377, 421, 853, 147, 339, 374, 971, 847, 975, 185, 883, 747, 56, 857, 752, 638, 574, 639, 611, 695, 318, 191, 580, 518, 321, 924, 131, 810, 854, 75, 119, 471, 43, 808, 4, 711, 872, 284, 840, 775, 993, 650, 310, 670, 595, 986, 407, 51, 586, 397, 130, 818, 464, 689, 786, 288, 592, 83, 490, 914, 222, 742, 989, 936, 224, 852, 404, 304, 596, 896, 431, 40, 772, 133, 823, 528, 13, 299, 134, 765, 878, 944, 93, 983, 1019, 806, 204, 579, 1008, 949, 701, 384, 652, 720, 358, 654, 3, 950, 226, 557, 887, 541, 324, 158, 651, 435, 320, 863, 519, 212, 331, 188, 1010, 937, 387, 751, 82, 402, 33, 6, 635, 988, 794, 27, 233, 411, 244, 194, 527, 293, 573, 348, 215, 107, 624, 559, 446, 279, 142, 619, 362, 721, 278, 862, 350, 918, 101, 216, 969, 795, 753, 660, 599, 386, 249, 84, 394, 347, 388, 472, 160, 120, 32, 475, 460, 673, 484, 151, 521, 963, 401, 656, 607, 243, 748, 913, 462, 16, 193, 340, 77, 21, 155, 686, 915, 632, 176, 718, 31, 814, 819, 451, 357, 281, 640, 312, 275, 229, 39, 425, 777, 964, 768, 81, 363, 360, 148, 69, 881, 779, 510, 922, 413, 661, 329, 562, 52, 953, 664, 538, 370, 516, 807, 1, 317, 108, 662, 138, 335, 948, 998, 280, 756, 55, 493, 57, 693, 154, 744, 71, 170, 618, 333, 778, 311, 361, 124, 172, 628, 298, 67, 369, 463, 295, 105, 723, 136, 218, 757, 583, 567, 412, 973, 290, 749, 999, 696, 192, 139, 171, 111, 940, 880, 709, 351, 601, 380, 995, 864, 585, 297, 762, 903, 935, 895, 338, 327, 616, 245, 759, 22, 970, 128, 382, 1004, 227, 416, 241, 738, 617, 774, 729, 783, 286, 251, 707, 730, 502, 773, 319, 905, 49, 122, 220, 23, 255, 433, 552, 621, 379, 153, 943, 558, 821, 225, 480, 848, 235, 667, 725, 334, 86, 750, 658, 598, 174, 12, 489, 610, 867, 1017, 313, 732, 605, 813, 498, 18, 162, 179, 1002, 900, 497, 50, 600, 713, 306, 991, 400, 432, 938, 403, 123, 396, 876, 722, 659, 296, 929, 785, 345, 217, 261, 555, 1011, 760, 1006, 390, 828, 655, 681, 703, 874, 945, 427, 941, 520, 968, 196, 649, 405, 564, 230, 112, 146, 665, 38, 909, 28, 737, 100, 974, 337, 697, 203, 690, 332, 80, 508, 897, 832, 267, 591, 445, 443, 300, 835, 165, 289, 615, 994, 563, 868, 144, 850, 305, 545, 420, 232, 959, 322, 904, 560, 24, 623, 256, 428, 505, 507, 132, 63, 199, 954, 186, 826, 436, 629, 796, 308, 364, 48, 798, 569, 620, 467, 873, 238, 698, 803, 570, 59, 842, 438, 831, 734, 316, 745, 200, 228, 990, 97, 782, 429, 273, 189, 781, 799, 677, 561, 177, 14, 1009, 678, 710, 35, 65, 865, 98, 92, 41, 674, 554, 837, 961, 127, 793, 705, 550, 977, 91, 980, 209, 966, 512, 496, 208, 156, 525, 159, 450, 447, 406, 951, 291, 213, 1023, 546, 981, 506, 523, 140, 419, 5, 292, 17, 870, 800, 849, 367, 644, 590, 846, 395, 287, 917, 315, 58, 73, 202, 175, 714, 886, 294, 424, 479, 383, 349, 597, 491, 266, 1001, 626, 1014, 68, 790, 942, 1007, 547, 544, 1013, 741, 190, 965, 877, 422, 625, 648, 372, 791, 469, 183, 637, 440, 731, 211, 833, 663, 879, 845, 285, 967, 398, 34, 309, 326, 997, 346, 314, 207, 99, 641, 602, 481, 958, 168]], [[400, 490, 982, 203, 966, 696, 8, 318, 56, 314, 234, 198, 162, 382, 860, 471, 84, 283, 808, 822, 78, 419, 524, 950, 506, 54, 7, 739, 843, 698, 71, 515, 798, 161, 990, 12, 252, 1005, 43, 989, 875, 410, 784, 221, 796, 706, 651, 334, 279, 740, 726, 576, 1013, 290, 580, 632, 958, 492, 270, 453, 452, 885, 942, 764, 830, 815, 734, 112, 50, 347, 190, 491, 607, 121, 609, 610, 415, 844, 852, 441, 86, 615, 45, 271, 849, 1022, 227, 257, 612, 254, 214, 549, 338, 826, 742, 386, 475, 111, 839, 477, 805, 951, 204, 34, 38, 141, 626, 661, 373, 1014, 500, 312, 14, 306, 464, 905, 170, 310, 422, 640, 63, 893, 947, 923, 404, 881, 900, 148, 941, 275, 717, 521, 30, 760, 823, 927, 356, 123, 377, 137, 82, 937, 501, 645, 224, 106, 743, 48, 1006, 339, 145, 908, 962, 608, 724, 700, 5, 675, 265, 226, 541, 485, 964, 116, 961, 972, 31, 707, 467, 308, 232, 892, 200, 436, 147, 183, 349, 666, 697, 556, 27, 261, 87, 998, 655, 687, 371, 831, 169, 423, 605, 229, 295, 483, 985, 369, 76, 376, 790, 26, 912, 212, 499, 25, 558, 548, 904, 928, 130, 981, 365, 156, 37, 213, 838, 59, 518, 876, 755, 1000, 915, 416, 946, 851, 765, 209, 443, 186, 62, 968, 540, 392, 444, 709, 503, 417, 664, 569, 494, 768, 385, 690, 727, 944, 332, 153, 925, 763, 819, 40, 363, 693, 292, 498, 979, 220, 879, 806, 398, 511, 430, 196, 11, 761, 683, 773, 446, 269, 99, 837, 836, 206, 101, 774, 374, 789, 49, 591, 454, 6, 970, 804, 230, 354, 786, 668, 429, 718, 704, 546, 529, 592, 460, 769, 703, 268, 522, 205, 297, 222, 604, 725, 959, 940, 70, 317, 331, 847, 816, 459, 74, 924, 732, 0, 309, 673, 341, 488, 906, 820, 335, 411, 239, 952, 303, 294, 599, 462, 803, 32, 397, 243, 304, 746, 218, 340, 891, 505, 642, 469, 554, 1019, 800, 963, 767, 794, 246, 242, 956, 278, 184, 573, 119, 620, 771, 143, 18, 61, 853, 516, 987, 939, 333, 372, 744, 801, 757, 888, 753, 967, 77, 207, 390, 993, 451, 567, 291, 324, 510, 1017, 432, 470, 719, 999, 442, 735, 41, 538, 589, 868, 133, 733, 902, 445, 188, 775, 250, 581, 359, 507, 484, 44, 931, 311, 65, 530, 60, 797, 236, 566, 235, 449, 714, 974, 754, 217, 590, 421, 864, 88, 248, 653, 758, 878, 828, 1011, 468, 166, 563, 51, 617, 277, 407, 783, 193, 514, 140, 802, 1009, 127, 895, 710, 577, 929, 379, 813, 138, 264, 886, 474, 702, 182, 988, 983, 388, 861, 689, 872, 181, 1010, 90, 938, 189, 330, 238, 665, 267, 694, 896, 662, 679, 992, 228, 752, 105, 107, 126, 749, 517, 300, 98, 81, 672, 253, 23, 550, 537, 486, 701, 930, 916, 1004, 409, 482, 713, 437, 643, 921, 244, 728, 3, 75, 532, 109, 21, 342, 984, 146, 671, 256, 128, 535, 901, 560, 473, 185, 497, 174, 565, 53, 858, 396, 194, 812, 788, 527, 840, 684, 621, 738, 814, 587, 348, 737, 163, 624, 439, 114, 799, 329, 914, 210, 343, 911, 448, 327, 721, 274, 731, 408, 108, 634, 10, 1007, 980, 670, 579, 414, 132, 751, 730, 564, 387, 571, 649, 890, 889, 502, 299, 139, 496, 777, 919, 688, 785, 187, 520, 787, 657, 122, 289, 996, 165, 509, 770, 595, 750, 570, 855, 884, 585, 778, 976, 741, 756, 948, 249, 370, 288, 611, 534, 149, 160, 129, 920, 791, 934, 575, 24, 1002, 68, 863, 16, 401, 115, 150, 629, 870, 197, 708, 381, 583, 481, 134, 125, 208, 772, 627, 366, 692, 827, 681, 1, 616, 135, 854, 736, 909, 871, 425, 543, 832, 52, 547, 361, 85, 413, 949, 173, 880, 322, 344, 72, 391, 539, 368, 79, 223, 100, 94, 825, 216, 955, 465, 555, 866, 280, 159, 20, 561, 480, 80, 158, 39, 118, 526, 96, 456, 647, 191, 792, 22, 211, 656, 856, 834, 102, 282, 782, 91, 723, 350, 614, 266, 542, 171, 762, 867, 262, 440, 544, 2, 110, 525, 92, 337, 36, 943, 922, 835, 523, 433, 33, 403, 177, 660, 276, 897, 352, 513, 586, 66, 658, 695, 913, 13, 325, 305, 420, 559, 971, 426, 654, 375, 172, 241, 887, 1016, 602, 648, 977, 686, 155, 225, 625, 622, 124, 180, 353, 817, 918, 46, 1012, 493, 598, 637, 659, 458, 508, 251, 623, 58, 383, 19, 258, 399, 833, 973, 95, 272, 476, 711, 779, 136, 320, 438, 1023, 367, 601, 406, 933, 73, 296, 682, 588, 678, 302, 997, 957, 705, 364, 669, 936, 435, 776, 945, 663, 67, 841, 298, 237, 285, 917, 641, 898, 157, 650, 574, 969, 857, 389, 55, 578, 260, 954, 873, 882, 729, 781, 447, 850, 402, 281, 519, 584, 720, 596, 531, 903, 233, 613, 428, 478, 1003, 1018, 593, 293, 691, 434, 677, 618, 117, 321, 259, 472, 568, 9, 759, 103, 273, 780, 395, 552, 167, 793, 466, 355, 151, 144, 1020, 795, 766, 231, 336, 606, 582, 215, 685, 652, 323, 326, 842, 457, 818, 633, 745, 572, 69, 394, 975, 603, 932, 986, 199, 899, 533, 479, 869, 179, 97, 47, 829, 907, 487, 315, 455, 1015, 995, 638, 245, 450, 809, 495, 716, 328, 255, 512, 154, 358, 28, 4, 597, 594, 201, 463, 551, 877, 811, 646, 346, 313, 393, 862, 910, 676, 412, 284, 113, 810, 557, 287, 553, 824, 807, 1008, 168, 427, 240, 545, 960, 357, 57, 301, 142, 42, 1001, 307, 360, 89, 378, 351, 859, 17, 247, 461, 164, 175, 639, 680, 748, 131, 489, 15, 747, 263, 562, 845, 965, 821, 674, 722, 894, 848, 528, 176, 874, 846, 319, 286, 192, 29, 630, 195, 712, 883, 935, 504, 64, 431, 316, 865, 384, 628, 926, 35, 93, 635, 362, 994, 380, 978, 1021, 619, 104, 424, 202, 699, 152, 83, 667, 405, 953, 178, 636, 345, 120, 715, 536, 991, 418, 644, 219, 600, 631], [201, 275, 612, 118, 277, 104, 542, 335, 948, 333, 203, 263, 671, 224, 103, 888, 66, 1020, 703, 131, 599, 495, 563, 235, 770, 871, 942, 996, 389, 887, 176, 1005, 989, 713, 119, 293, 478, 1004, 374, 156, 926, 727, 63, 340, 723, 268, 580, 592, 718, 204, 489, 31, 297, 964, 827, 851, 161, 481, 621, 172, 446, 310, 223, 307, 298, 149, 669, 805, 372, 520, 1015, 124, 646, 228, 993, 192, 656, 427, 980, 404, 416, 1011, 289, 122, 381, 555, 704, 106, 232, 640, 847, 252, 603, 722, 936, 188, 439, 164, 79, 59, 353, 687, 354, 858, 143, 506, 678, 558, 738, 376, 342, 392, 444, 927, 363, 874, 978, 706, 200, 220, 387, 27, 789, 829, 861, 238, 650, 431, 983, 533, 151, 711, 528, 538, 464, 160, 266, 945, 733, 399, 62, 826, 5, 961, 976, 501, 476, 237, 783, 735, 667, 920, 966, 956, 320, 432, 562, 685, 857, 434, 290, 56, 904, 883, 809, 99, 1017, 556, 523, 360, 40, 649, 595, 43, 917, 123, 605, 591, 403, 708, 193, 824, 102, 344, 662, 759, 880, 429, 623, 358, 992, 410, 471, 923, 304, 273, 743, 325, 260, 294, 891, 458, 660, 696, 288, 133, 751, 796, 530, 300, 746, 53, 643, 166, 499, 182, 468, 724, 184, 516, 279, 579, 802, 100, 778, 882, 95, 863, 250, 107, 693, 137, 377, 884, 356, 146, 914, 46, 697, 242, 257, 730, 258, 787, 384, 230, 570, 572, 494, 742, 25, 893, 507, 817, 648, 155, 418, 521, 482, 383, 661, 1009, 959, 326, 393, 534, 20, 366, 854, 578, 180, 894, 617, 226, 483, 589, 12, 690, 1, 969, 406, 673, 513, 791, 217, 573, 469, 165, 114, 480, 526, 218, 92, 52, 33, 361, 78, 13, 663, 328, 870, 974, 178, 317, 251, 633, 29, 225, 436, 426, 449, 543, 582, 776, 984, 219, 958, 321, 285, 601, 531, 117, 42, 994, 514, 417, 236, 638, 339, 913, 919, 1013, 183, 786, 443, 247, 807, 614, 470, 2, 581, 276, 597, 895, 454, 682, 142, 329, 244, 695, 109, 910, 39, 3, 694, 797, 768, 775, 159, 57, 999, 221, 869, 714, 502, 524, 658, 373, 840, 970, 602, 319, 918, 760, 9, 1000, 905, 835, 583, 272, 227, 641, 334, 213, 607, 441, 928, 58, 121, 466, 699, 941, 954, 337, 1021, 286, 608, 868, 832, 705, 547, 412, 14, 48, 38, 457, 849, 889, 504, 16, 397, 853, 318, 391, 302, 729, 327, 686, 101, 544, 347, 965, 401, 653, 455, 153, 594, 585, 642, 405, 398, 433, 896, 816, 453, 915, 1001, 493, 780, 448, 437, 613, 330, 430, 689, 364, 822, 1018, 451, 767, 842, 766, 815, 264, 782, 477, 116, 282, 762, 624, 72, 208, 677, 86, 973, 808, 371, 202, 788, 324, 435, 36, 635, 616, 921, 598, 375, 773, 396, 91, 960, 726, 175, 497, 190, 98, 753, 879, 865, 338, 626, 892, 24, 440, 515, 195, 932, 725, 576, 1002, 803, 732, 748, 557, 701, 261, 30, 750, 314, 134, 424, 359, 934, 96, 916, 885, 140, 474, 205, 0, 593, 813, 707, 265, 111, 790, 1019, 76, 77, 11, 21, 496, 944, 256, 17, 171, 370, 331, 185, 740, 112, 428, 752, 937, 352, 698, 245, 535, 938, 554, 684, 878, 772, 804, 168, 343, 902, 737, 73, 981, 644, 931, 529, 423, 537, 127, 212, 600, 666, 125, 859, 761, 296, 818, 197, 540, 971, 610, 509, 552, 154, 105, 308, 548, 525, 741, 719, 950, 844, 35, 408, 207, 135, 793, 801, 745, 488, 1008, 83, 26, 898, 312, 108, 559, 672, 382, 70, 908, 486, 903, 241, 419, 148, 152, 518, 734, 798, 158, 270, 55, 977, 995, 65, 952, 839, 681, 890, 814, 627, 479, 654, 951, 191, 267, 295, 512, 784, 8, 32, 979, 680, 862, 825, 517, 596, 442, 756, 50, 710, 712, 929, 630, 911, 157, 388, 731, 1012, 315, 943, 561, 736, 6, 912, 799, 44, 90, 655, 634, 922, 838, 346, 716, 18, 550, 460, 848, 998, 402, 909, 567, 897, 51, 323, 484, 75, 590, 500, 728, 115, 194, 702, 545, 97, 511, 316, 313, 68, 64, 450, 169, 568, 23, 198, 962, 625, 664, 249, 631, 467, 532, 400, 47, 34, 214, 132, 1023, 877, 619, 795, 305, 269, 255, 652, 639, 1014, 505, 136, 492, 843, 541, 84, 243, 830, 259, 421, 10, 620, 395, 860, 129, 586, 367, 575, 461, 187, 254, 565, 721, 975, 271, 209, 679, 28, 549, 233, 139, 777, 771, 715, 126, 615, 407, 475, 179, 632, 196, 1022, 7, 447, 845, 990, 61, 820, 341, 94, 546, 604, 128, 587, 674, 306, 907, 229, 647, 864, 425, 167, 700, 947, 819, 692, 834, 881, 215, 850, 774, 508, 755, 873, 886, 747, 821, 792, 536, 539, 577, 1003, 691, 130, 588, 811, 177, 301, 785, 794, 181, 472, 991, 385, 551, 997, 841, 569, 901, 292, 69, 386, 81, 368, 462, 71, 240, 836, 754, 519, 210, 369, 622, 186, 19, 720, 456, 651, 967, 609, 490, 560, 60, 846, 379, 925, 284, 659, 394, 147, 351, 503, 968, 141, 744, 618, 522, 935, 445, 350, 872, 781, 899, 946, 628, 348, 779, 280, 957, 675, 41, 564, 574, 924, 287, 963, 253, 463, 831, 668, 390, 764, 800, 415, 189, 74, 657, 1016, 362, 67, 606, 510, 949, 336, 222, 933, 828, 810, 87, 571, 955, 763, 22, 120, 411, 54, 757, 765, 246, 855, 303, 49, 378, 311, 498, 611, 852, 37, 749, 930, 173, 1006, 309, 357, 283, 986, 409, 636, 211, 113, 231, 867, 487, 452, 174, 322, 299, 438, 88, 906, 45, 281, 900, 688, 162, 833, 93, 89, 953, 163, 145, 584, 278, 940, 206, 465, 216, 234, 332, 527, 676, 758, 875, 239, 144, 4, 739, 683, 985, 262, 349, 987, 459, 637, 670, 110, 380, 491, 837, 982, 665, 856, 170, 629, 1007, 291, 345, 473, 939, 82, 414, 413, 248, 972, 199, 1010, 823, 988, 709, 876, 769, 566, 553, 355, 717, 138, 85, 274, 365, 422, 420, 15, 866, 80, 645, 806, 150, 485, 812], [757, 430, 466, 649, 412, 266, 210, 477, 746, 428, 592, 551, 415, 309, 867, 183, 21, 799, 509, 186, 267, 343, 535, 53, 737, 507, 543, 967, 34, 8, 656, 614, 435, 962, 554, 906, 7, 2, 184, 555, 844, 283, 497, 886, 232, 464, 981, 517, 85, 658, 898, 823, 940, 421, 260, 413, 494, 254, 557, 612, 578, 342, 392, 620, 27, 214, 1015, 676, 791, 909, 949, 766, 610, 796, 865, 162, 449, 180, 527, 340, 236, 854, 133, 587, 863, 702, 173, 694, 237, 331, 46, 920, 255, 264, 176, 329, 45, 901, 473, 653, 317, 964, 574, 519, 358, 65, 878, 929, 372, 883, 359, 1004, 925, 932, 456, 680, 418, 508, 22, 119, 203, 856, 666, 753, 781, 294, 970, 213, 446, 729, 1020, 501, 802, 947, 73, 288, 188, 132, 630, 820, 285, 955, 141, 520, 829, 434, 536, 381, 209, 63, 564, 648, 1, 602, 472, 441, 1016, 577, 126, 525, 735, 980, 326, 455, 515, 583, 594, 971, 212, 376, 707, 919, 356, 987, 502, 78, 631, 72, 881, 687, 638, 847, 686, 77, 677, 550, 859, 625, 304, 263, 39, 945, 468, 249, 892, 120, 322, 573, 59, 196, 683, 986, 722, 357, 895, 758, 161, 689, 163, 190, 189, 44, 590, 532, 530, 375, 238, 437, 278, 242, 705, 846, 122, 445, 724, 615, 599, 874, 113, 366, 695, 714, 635, 798, 75, 700, 9, 244, 887, 296, 500, 142, 11, 348, 855, 572, 618, 994, 105, 595, 659, 83, 566, 928, 756, 852, 741, 930, 3, 927, 905, 416, 211, 89, 503, 708, 759, 4, 273, 654, 531, 147, 861, 642, 1022, 706, 471, 742, 17, 1023, 13, 483, 860, 424, 57, 961, 690, 490, 819, 346, 69, 858, 0, 561, 600, 175, 782, 982, 843, 447, 124, 931, 715, 384, 1002, 262, 353, 591, 128, 117, 711, 657, 369, 585, 542, 1009, 522, 803, 731, 733, 727, 91, 115, 140, 565, 200, 241, 866, 910, 807, 32, 770, 225, 701, 764, 166, 325, 674, 332, 121, 832, 924, 570, 417, 942, 328, 234, 915, 1013, 609, 954, 611, 282, 295, 661, 723, 320, 19, 991, 64, 118, 274, 164, 80, 916, 144, 771, 495, 584, 431, 146, 840, 539, 403, 795, 1006, 235, 312, 646, 297, 23, 246, 217, 231, 698, 923, 720, 914, 233, 192, 247, 261, 514, 206, 540, 641, 313, 402, 899, 730, 769, 409, 697, 139, 154, 1018, 568, 889, 62, 979, 660, 70, 355, 933, 112, 151, 390, 367, 839, 552, 785, 228, 549, 765, 976, 911, 226, 868, 744, 556, 291, 71, 54, 968, 51, 768, 1007, 460, 318, 28, 448, 569, 158, 939, 336, 420, 534, 199, 825, 651, 380, 608, 983, 957, 388, 767, 389, 1014, 153, 314, 440, 812, 891, 636, 393, 597, 169, 461, 897, 323, 66, 181, 624, 104, 90, 41, 439, 371, 894, 386, 966, 174, 344, 40, 643, 696, 341, 805, 576, 429, 426, 307, 20, 893, 529, 943, 836, 82, 1008, 257, 880, 99, 885, 395, 202, 110, 12, 88, 333, 478, 52, 692, 270, 903, 837, 833, 682, 645, 269, 725, 670, 997, 926, 518, 411, 853, 851, 250, 30, 306, 103, 365, 580, 827, 622, 838, 778, 397, 734, 385, 252, 824, 563, 14, 571, 432, 92, 749, 1003, 35, 55, 792, 136, 817, 370, 1001, 364, 197, 732, 793, 988, 97, 709, 335, 606, 492, 632, 87, 310, 712, 38, 170, 748, 806, 721, 316, 405, 382, 671, 810, 368, 941, 900, 533, 596, 305, 123, 451, 419, 243, 131, 86, 977, 145, 544, 363, 752, 797, 229, 10, 934, 259, 688, 108, 205, 546, 1010, 148, 324, 745, 650, 290, 879, 786, 452, 187, 978, 775, 814, 516, 300, 679, 716, 538, 673, 315, 68, 601, 275, 94, 308, 276, 773, 790, 182, 960, 116, 541, 299, 377, 5, 743, 125, 222, 864, 67, 76, 902, 6, 787, 81, 1000, 589, 703, 350, 303, 298, 754, 607, 100, 918, 487, 185, 588, 553, 634, 43, 15, 975, 1012, 272, 457, 378, 523, 871, 327, 293, 877, 354, 655, 134, 58, 334, 157, 253, 251, 848, 168, 178, 907, 442, 784, 956, 908, 215, 407, 1011, 396, 496, 710, 760, 678, 605, 665, 884, 414, 738, 973, 755, 179, 219, 193, 562, 37, 423, 111, 486, 152, 493, 462, 952, 821, 443, 521, 18, 818, 84, 800, 33, 652, 159, 480, 191, 581, 330, 427, 245, 699, 989, 667, 912, 281, 996, 399, 736, 379, 873, 79, 761, 751, 319, 627, 239, 481, 828, 279, 207, 499, 60, 637, 619, 337, 475, 224, 811, 16, 433, 633, 890, 750, 394, 794, 972, 129, 227, 512, 965, 842, 459, 101, 675, 177, 454, 998, 869, 613, 990, 102, 888, 404, 95, 425, 946, 1017, 347, 406, 582, 640, 127, 339, 763, 728, 685, 135, 265, 579, 438, 801, 167, 850, 374, 951, 248, 935, 491, 156, 995, 862, 944, 545, 467, 691, 876, 373, 813, 762, 463, 875, 772, 663, 150, 904, 747, 171, 704, 575, 484, 718, 1005, 130, 223, 816, 963, 616, 779, 984, 719, 218, 913, 488, 808, 422, 138, 834, 776, 559, 31, 458, 485, 804, 921, 96, 788, 489, 992, 629, 993, 617, 882, 639, 547, 469, 198, 621, 598, 974, 383, 360, 603, 410, 482, 258, 352, 98, 444, 29, 114, 25, 513, 669, 985, 506, 401, 56, 321, 815, 870, 937, 349, 149, 49, 476, 137, 644, 835, 195, 194, 780, 106, 465, 511, 256, 289, 560, 287, 739, 953, 662, 345, 777, 271, 450, 311, 959, 726, 208, 524, 950, 398, 717, 221, 362, 872, 498, 453, 48, 604, 93, 286, 826, 526, 351, 160, 845, 623, 831, 143, 917, 922, 626, 301, 672, 338, 302, 220, 479, 24, 50, 61, 668, 470, 201, 230, 593, 1019, 361, 436, 789, 969, 268, 740, 783, 74, 647, 896, 664, 628, 681, 165, 510, 155, 693, 713, 408, 822, 841, 830, 537, 774, 292, 504, 47, 240, 938, 172, 809, 216, 109, 999, 391, 558, 387, 1021, 948, 849, 107, 548, 474, 684, 26, 505, 936, 284, 400, 42, 586, 958, 528, 36, 280, 204, 857, 567, 277], [992, 672, 1000, 675, 699, 919, 299, 958, 23, 463, 854, 883, 660, 443, 139, 420, 837, 736, 896, 314, 848, 589, 860, 979, 217, 861, 794, 325, 733, 698, 530, 116, 205, 502, 745, 788, 967, 188, 222, 369, 737, 503, 903, 731, 512, 234, 741, 857, 432, 202, 200, 728, 3, 899, 170, 559, 136, 909, 41, 364, 162, 377, 882, 171, 293, 284, 479, 208, 26, 121, 843, 624, 801, 622, 724, 132, 231, 125, 495, 310, 426, 908, 668, 871, 693, 95, 681, 519, 995, 836, 772, 918, 437, 839, 365, 739, 600, 54, 321, 711, 273, 730, 734, 595, 115, 74, 949, 407, 858, 565, 197, 630, 193, 155, 580, 636, 73, 564, 230, 707, 339, 638, 29, 27, 555, 993, 542, 366, 599, 2, 611, 570, 971, 721, 218, 574, 1005, 216, 892, 179, 1, 264, 308, 709, 415, 1020, 923, 914, 488, 1023, 228, 920, 957, 670, 783, 451, 349, 198, 126, 44, 787, 395, 153, 827, 539, 529, 779, 631, 77, 1006, 533, 685, 691, 944, 492, 643, 131, 667, 647, 886, 658, 219, 689, 374, 175, 815, 986, 552, 191, 429, 751, 878, 635, 117, 444, 214, 692, 372, 408, 964, 814, 881, 465, 496, 725, 669, 142, 571, 105, 134, 111, 510, 749, 916, 558, 800, 260, 174, 30, 773, 63, 782, 774, 6, 357, 490, 181, 261, 398, 614, 140, 974, 951, 301, 581, 934, 809, 639, 756, 893, 965, 928, 76, 806, 101, 435, 455, 19, 895, 822, 487, 403, 744, 467, 472, 500, 79, 255, 362, 927, 107, 1016, 541, 478, 906, 982, 237, 966, 471, 726, 729, 985, 789, 344, 723, 942, 281, 536, 265, 791, 535, 969, 477, 1013, 538, 18, 644, 856, 585, 447, 272, 1022, 241, 450, 315, 64, 627, 66, 256, 885, 401, 889, 792, 831, 210, 109, 719, 169, 48, 808, 224, 655, 340, 119, 122, 59, 973, 873, 523, 433, 342, 632, 924, 378, 888, 666, 417, 987, 833, 199, 524, 754, 876, 106, 853, 594, 9, 244, 158, 690, 820, 528, 884, 700, 623, 154, 785, 318, 484, 333, 955, 319, 13, 405, 764, 930, 609, 567, 438, 178, 640, 798, 394, 31, 824, 453, 28, 586, 43, 938, 481, 146, 245, 701, 391, 994, 22, 137, 0, 483, 147, 596, 304, 60, 662, 263, 112, 518, 936, 266, 679, 262, 280, 935, 625, 904, 910, 712, 954, 629, 799, 526, 894, 797, 439, 917, 842, 727, 336, 353, 877, 248, 811, 240, 775, 997, 489, 850, 172, 130, 24, 761, 156, 32, 49, 207, 359, 90, 418, 47, 981, 1004, 400, 975, 352, 5, 40, 183, 286, 862, 509, 356, 209, 279, 763, 977, 546, 1021, 572, 926, 164, 805, 597, 807, 708, 499, 642, 867, 677, 652, 829, 257, 348, 215, 849, 553, 300, 238, 963, 185, 562, 769, 151, 141, 747, 56, 192, 14, 844, 39, 120, 812, 770, 551, 38, 307, 7, 341, 654, 82, 1003, 898, 868, 294, 840, 498, 242, 337, 456, 466, 108, 959, 298, 947, 865, 776, 511, 1012, 212, 1010, 863, 687, 329, 501, 507, 454, 292, 355, 550, 475, 190, 852, 664, 688, 414, 33, 742, 182, 592, 513, 554, 569, 291, 743, 990, 312, 710, 962, 305, 232, 864, 203, 86, 320, 229, 573, 999, 671, 113, 12, 440, 720, 870, 446, 16, 813, 253, 306, 143, 223, 69, 220, 780, 251, 167, 271, 409, 180, 309, 996, 10, 768, 102, 289, 204, 15, 828, 838, 653, 617, 1001, 159, 89, 804, 943, 346, 686, 902, 616, 907, 328, 834, 651, 166, 577, 988, 194, 1015, 716, 520, 795, 766, 715, 392, 758, 645, 810, 68, 173, 384, 880, 646, 759, 983, 62, 419, 351, 953, 393, 104, 367, 144, 605, 128, 20, 243, 650, 302, 911, 80, 331, 1008, 421, 576, 704, 296, 145, 282, 945, 327, 695, 802, 869, 36, 259, 714, 713, 84, 258, 825, 522, 275, 473, 474, 823, 387, 246, 65, 568, 803, 51, 347, 380, 598, 746, 817, 537, 582, 1011, 476, 370, 52, 460, 195, 57, 678, 354, 901, 53, 543, 859, 350, 532, 781, 1014, 99, 313, 765, 648, 937, 133, 110, 557, 326, 1017, 268, 55, 846, 960, 540, 549, 527, 835, 239, 694, 750, 619, 96, 757, 607, 187, 748, 457, 42, 150, 847, 890, 58, 777, 225, 912, 287, 406, 335, 78, 206, 922, 1009, 784, 317, 412, 753, 93, 363, 544, 941, 21, 931, 680, 989, 11, 458, 25, 735, 991, 343, 323, 290, 276, 482, 661, 760, 4, 706, 427, 968, 545, 504, 462, 593, 841, 925, 674, 91, 816, 961, 413, 152, 948, 196, 270, 767, 448, 905, 34, 322, 587, 755, 933, 381, 866, 431, 157, 752, 921, 480, 676, 88, 311, 946, 696, 452, 184, 821, 233, 612, 548, 845, 493, 560, 819, 390, 786, 103, 189, 515, 449, 375, 211, 575, 932, 75, 740, 665, 566, 940, 491, 330, 303, 615, 497, 732, 972, 461, 416, 70, 160, 213, 100, 1018, 470, 397, 269, 249, 124, 929, 628, 703, 94, 468, 85, 411, 603, 970, 163, 428, 900, 606, 915, 72, 295, 534, 718, 8, 338, 939, 832, 1002, 129, 771, 35, 702, 138, 956, 684, 402, 277, 250, 123, 590, 556, 494, 976, 161, 998, 368, 274, 588, 656, 872, 283, 358, 547, 334, 659, 45, 422, 891, 641, 87, 71, 436, 267, 657, 583, 382, 505, 579, 118, 637, 633, 425, 176, 235, 738, 563, 386, 227, 332, 578, 790, 683, 913, 525, 61, 879, 373, 247, 584, 388, 521, 621, 984, 663, 389, 434, 442, 285, 360, 383, 851, 149, 514, 97, 634, 610, 506, 127, 1007, 445, 591, 46, 697, 396, 399, 221, 601, 469, 165, 385, 135, 796, 517, 613, 254, 430, 361, 874, 673, 37, 168, 288, 618, 516, 626, 404, 602, 410, 376, 887, 316, 324, 649, 464, 620, 297, 226, 950, 177, 875, 818, 236, 441, 561, 978, 83, 826, 423, 778, 345, 980, 830, 1019, 531, 897, 793, 92, 608, 186, 379, 50, 485, 148, 722, 114, 604, 486, 459, 855, 682, 424, 67, 17, 762, 98, 508, 371, 81, 705, 952, 717, 201, 252, 278]]]
    winning_perm_model_sd = []
    for i in range(5):
        winning_perm_model_sd.append(
            permute(args.arch, model, winning_permutation[i], sd[i], w[i], nchannels, nclasses, args.width))
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
        # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/auto/{source_file_name}'
        destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA_InstanceOptimized_v2/oracle/SA/grid/{exp_no}/{source_file_name}'
        pickle_out = pickle.dumps(dict_after)
        upload_pkl(bucket_name, pickle_out, destination_blob_name)
    print()
    print("basin_mean_after", statistics.mean(barrier_test_basin))
    print("basin_std_after", statistics.stdev(barrier_test_basin))






    # # ########################################## SA original models: model1 and model2
    sd = []
    # for j in [7, 8, 9, 15, 16]:
    for j in [20, 11, 12, 13, 19]:
        bucket_name = 'permutation-mlp'
        destination_blob_name = 'model_best.th'
        source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{j}/{destination_blob_name}'
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
            random.seed(i)
            rnd = random.sample(lst, z)
            random_permuted_index.append(rnd)
        init_state.append(random_permuted_index)

    exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    winning_permutation = barrier_SA(args.arch, model, sd, w, init_state,
                                     args.tmax, args.tmin, args.steps,
                                     train_inputs, train_targets,
                                     nchannels, nclasses, args.width)
    print("winning_permutation", winning_permutation)
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
        # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/auto/{source_file_name}'
        destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA_InstanceOptimized_v2/grid/{exp_no}/{source_file_name}'
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


def evaluate_model(args, model, inputs, targets):
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
            if args.arch == 'mlp':
                input = data.to(device).view(data.size(0), -1)
            else:
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



def evaluate_model_small(args, model, inputs, targets):
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
            if args.arch == 'mlp':
                input = data.to(device).view(data.size(0), -1)
            else:
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
        result_train.append(evaluate_model(args, model, train_inputs, train_targets)['top1'])
        result_test.append(evaluate_model(args, model, test_inputs, test_targets)['top1'])

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
        model = MLP1_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.0.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.0.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.2.bias"][:] = torch.Tensor(b2_p)
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
        w2_p = w2[:, idx1] ### to take care of prv permutation
        w2_p = w2_p[idx2, :] ## to apply new permutation
        b2_p = b2[idx2]

        idx2 = perm_ind[1]
        w3_p = w3[:, idx2]
        b3_p = b3

        ##################### save model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP2_layer(n_units=nunits, n_channels=nchannels, n_classes=nclasses)
        model = model.to(device)

        model.state_dict()["layers.0.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.0.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.4.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.4.bias"][:] = torch.Tensor(b3_p)
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

        model.state_dict()["layers.0.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.0.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.4.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.4.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["layers.6.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["layers.6.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["layers.8.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["layers.8.bias"][:] = torch.Tensor(b5_p)
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

        model.state_dict()["layers.0.weight"][:] = torch.Tensor(w1_p)
        model.state_dict()["layers.0.bias"][:] = torch.Tensor(b1_p)
        model.state_dict()["layers.2.weight"][:] = torch.Tensor(w2_p)
        model.state_dict()["layers.2.bias"][:] = torch.Tensor(b2_p)
        model.state_dict()["layers.4.weight"][:] = torch.Tensor(w3_p)
        model.state_dict()["layers.4.bias"][:] = torch.Tensor(b3_p)
        model.state_dict()["layers.6.weight"][:] = torch.Tensor(w4_p)
        model.state_dict()["layers.6.bias"][:] = torch.Tensor(b4_p)
        model.state_dict()["layers.8.weight"][:] = torch.Tensor(w5_p)
        model.state_dict()["layers.8.bias"][:] = torch.Tensor(b5_p)
        model.state_dict()["layers.10.weight"][:] = torch.Tensor(w6_p)
        model.state_dict()["layers.10.bias"][:] = torch.Tensor(b6_p)
        model.state_dict()["layers.12.weight"][:] = torch.Tensor(w7_p)
        model.state_dict()["layers.12.bias"][:] = torch.Tensor(b7_p)
        model.state_dict()["layers.14.weight"][:] = torch.Tensor(w8_p)
        model.state_dict()["layers.14.bias"][:] = torch.Tensor(b8_p)
        model.state_dict()["layers.16.weight"][:] = torch.Tensor(w9_p)
        model.state_dict()["layers.16.bias"][:] = torch.Tensor(b9_p)
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
            permuted_models = []
            for i in range(5):
                permuted_models.append(permute(arch, model, self.state[i], sd2[i], w2[i], nchannels, nclasses, nunits))
            #### form one model which is the average of 5 permuted models
            permuted_avg = copy.deepcopy(model)
            new_params = OrderedDict()
            for key in sd2[0].keys():
                param = 0
                for i in range(len(permuted_models)):
                    param = param + permuted_models[i][key]
                # print(permuted_models[i].state_dict()[key])
                # print(key, i , param.shape, param)
                new_params[key] = param / len(permuted_models)
            permuted_avg.load_state_dict(new_params)
            eval_train = evaluate_model_small(args, permuted_avg, train_inputs, train_targets)['top1']
            # eval_train = evaluate_model(args, permuted_avg, train_inputs, train_targets)['top1']
            cost = 1 - eval_train
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