import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
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
parser.add_argument('--dataset', default='SVHN', type=str,
                    help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
parser.add_argument('--batchsize', default=64, type=int,
                    help='input batch size (default: 64)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--nlayers', default=18, type=int)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--steps', default=50000, type=int)
parser.add_argument('--tmax', default=25000, type=float)
parser.add_argument('--tmin', default=2.5, type=float)
parser.add_argument('--pair', default=1, type=int)
parser.add_argument('--rand_seed', default=1, type=int)
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
        model = vgg.__dict__[args.arch](nchannels, args.width, nclasses)

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


    chosen_seed = [i for i in range(1,21)]

    ############## form 10 pairs
    random.seed(1)
    def pop_random(lst):
        idx = random.randrange(0, len(lst))
        return lst.pop(idx)

    pairs = []
    lst = chosen_seed
    while lst:
        rand1 = pop_random(lst)
        rand2 = pop_random(lst)
        pair = rand1, rand2
        pairs.append(pair)

    dict_before = {}
    print("paired models: ", pairs)

    ################## original barrier
    # for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # for x in [10]:
    # pair = pairs[x-1]
    pair = pairs[args.pair-1]
    print("pair", pair)
    model1_id = pair[0]
    model2_id = pair[1]
    ############################# load selected models
    bucket_name = 'permutation-mlp'
    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model1_id}/{destination_blob_name}'
    download_blob(bucket_name, source_file_name, destination_blob_name)

    checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    sd1 = checkpoint

    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model2_id}/{destination_blob_name}'
    download_blob(bucket_name, source_file_name, destination_blob_name)

    checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    sd2 = checkpoint


    def key_transformation(old_key):
        if 'module' in old_key:
            return old_key[7:]
        return old_key

    new_state_dict1 = OrderedDict()
    for key, value in sd1.items():
        new_key = key_transformation(key)
        new_state_dict1[new_key] = value

    new_state_dict2 = OrderedDict()
    for key, value in sd2.items():
        new_key = key_transformation(key)
        new_state_dict2[new_key] = value
    sd1 = new_state_dict1
    sd2 = new_state_dict2


    conv_arch = False
    for key in sd1:
        print(key, sd1[key].shape)
        if "conv" in key or "running_mean" in key:
            conv_arch = True
    # from torchsummary import summary
    # summary(model, (1, 32, 32))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    # ########################################## original barrier
    dict_before = get_barrier(model, sd1, sd2, train_inputs, train_targets, test_inputs, test_targets)
    barrier_train = dict_before['barrier_train']
    lmc_train = dict_before['train_lmc']

    test_avg_org_models = dict_before['test_avg_models'][0]
    train_avg_org_models = dict_before['train_avg_models'][0]
    print("barrier_train_original", barrier_train)
    print("lmc_train_original", lmc_train)

    source_file_name = f'dict_before_{args.pair}.pkl'
    destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/original/{source_file_name}'
    pickle_out = pickle.dumps(dict_before)
    upload_pkl(bucket_name, pickle_out, destination_blob_name)
    ########################################## oracle barrier
    bucket_name = 'permutation-mlp'
    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{21}/{destination_blob_name}'
    download_blob(bucket_name, source_file_name, destination_blob_name)

    checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    sd1 = checkpoint

    def key_transformation(old_key):
        if 'module' in old_key:
            return old_key[7:]
        return old_key

    new_state_dict1 = OrderedDict()
    for key, value in sd1.items():
        new_key = key_transformation(key)
        new_state_dict1[new_key] = value
    sd1 = new_state_dict1
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
    if conv_arch:
        params = []
        len_perm = []
        for key in sd2.keys():
            param = model.state_dict()[key]
            if "num_batches_tracked" not in key:
                params.append(param.cpu().detach().numpy())
                if len(param.shape) == 4:
                    len_perm.append(param.shape[0])
                if len(param.shape) == 2:
                    len_perm.append(param.shape[0])

    print("len_perm", len(len_perm))
    print("len_perm", len_perm)


    random_permuted_index = []
    for z in len_perm:
        lst = [y for y in range(z)]
        random.seed(model2_id)
        # random.seed(args.rand_seed)
        rnd = random.sample(lst, z)
        random_permuted_index.append(rnd)
    init_state1 = random_permuted_index

    # print(sd1["features.0.weight"][0:2])
    permuted_oracle_sd1 = permute(args.arch, model, init_state1, sd1, w1, nchannels, nclasses, args.width)
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


    # ##################################################
    bucket_name = 'permutation-mlp'
    destination_blob_name = 'model_best.th'
    source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{21}/{destination_blob_name}'
    download_blob(bucket_name, source_file_name, destination_blob_name)

    checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    sd1 = checkpoint

    def key_transformation(old_key):
        if 'module' in old_key:
            return old_key[7:]
        return old_key

    new_state_dict1 = OrderedDict()
    for key, value in sd1.items():
        new_key = key_transformation(key)
        new_state_dict1[new_key] = value

    sd1 = new_state_dict1


    dict_oracle = get_barrier(model, sd1, permuted_oracle_sd1, train_inputs, train_targets, test_inputs, test_targets)
    barrier_train = dict_oracle['barrier_train']
    lmc_train = dict_oracle['train_lmc']

    print("barrier_train_oracle", barrier_train)
    print("lmc_train_oracle", lmc_train)

    # source_file_name = f'dict_oracle_{args.pair}_{args.rand_seed}.pkl'
    source_file_name = f'dict_oracle_{args.pair}.pkl'
    destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/oracle/before/{source_file_name}'
    pickle_out = pickle.dumps(dict_oracle)
    upload_pkl(bucket_name, pickle_out, destination_blob_name)



    # # # ########################################## SA original models: model1 and model2
    # # load selected models
    # bucket_name = 'permutation-mlp'
    # destination_blob_name = 'model_best.th'
    # source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model1_id}/{destination_blob_name}'
    # download_blob(bucket_name, source_file_name, destination_blob_name)
    #
    # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # sd1 = checkpoint
    #
    # destination_blob_name = 'model_best.th'
    # source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model2_id}/{destination_blob_name}'
    # download_blob(bucket_name, source_file_name, destination_blob_name)
    #
    # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # sd2 = checkpoint
    #
    #
    # def key_transformation(old_key):
    #     if 'module' in old_key:
    #         return old_key[7:]
    #     return old_key
    #
    # new_state_dict1 = OrderedDict()
    # for key, value in sd1.items():
    #     new_key = key_transformation(key)
    #     new_state_dict1[new_key] = value
    #
    # new_state_dict2 = OrderedDict()
    # for key, value in sd2.items():
    #     new_key = key_transformation(key)
    #     new_state_dict2[new_key] = value
    # sd1 = new_state_dict1
    # sd2 = new_state_dict2
    #
    # w2 = []
    # for key in sd2.keys():
    #     param = sd2[key]
    #     w2.append(param.cpu().detach().numpy())
    #
    #
    # # create permutation list for mlp
    # if args.arch == 'mlp':
    #     len_perm = []
    #     for i in range(int(len(w2) / 2 - 1)):
    #         len_perm.append(args.width)
    # # create permutation list for conv nets
    # if conv_arch:
    #     params = []
    #     len_perm = []
    #     for key in sd2.keys():
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
    # random_permuted_index = []
    # for z in len_perm:
    #     lst = [y for y in range(z)]
    #     random.seed(model2_id)
    #     rnd = random.sample(lst, z)
    #     random_permuted_index.append(rnd)
    # init_state1 = random_permuted_index
    # exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    # winning_permutation = barrier_SA(args.arch, model, sd1, sd2, w2, init_state1,
    #                                  args.tmax, args.tmin, args.steps,
    #                                  train_inputs, train_targets,
    #                                  train_avg_org_models, nchannels, nclasses,
    #                                  args.width)
    #
    # print("winning_permutation", winning_permutation)
    # winning_perm_model_sd = permute(args.arch, model, winning_permutation, sd2, w2, nchannels, nclasses, args.width)
    #
    # dict_after = get_barrier(model, sd1, winning_perm_model_sd, train_inputs, train_targets, test_inputs, test_targets)
    #
    # add_element(dict_after, 'winning_permutation', winning_permutation)
    # add_element(dict_after, 'winning_perm_model_sd', winning_perm_model_sd)
    #
    # barrier_train = dict_after['barrier_train']
    # lmc_train = dict_after['train_lmc']
    #
    # print("barrier_train_SA", barrier_train)
    # print("lmc_train_SA", lmc_train)
    #
    # source_file_name = f'dict_after_pair{pair}.pkl'
    # # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/auto/{source_file_name}'
    # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/SA/grid/{exp_no}/{source_file_name}'
    # pickle_out = pickle.dumps(dict_after)
    # upload_pkl(bucket_name, pickle_out, destination_blob_name)
    #
    # # ########################################## SA oracle: model1 and permuted model1
    # # load selected models
    # bucket_name = 'permutation-mlp'
    # destination_blob_name = 'model_best.th'
    # source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model1_id}/{destination_blob_name}'
    # download_blob(bucket_name, source_file_name, destination_blob_name)
    #
    # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # sd1 = checkpoint
    #
    # destination_blob_name = 'model_best.th'
    # source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{model2_id}/{destination_blob_name}'
    # download_blob(bucket_name, source_file_name, destination_blob_name)
    #
    # checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
    # sd2 = checkpoint
    #
    #
    # def key_transformation(old_key):
    #     if 'module' in old_key:
    #         return old_key[7:]
    #     return old_key
    #
    # new_state_dict1 = OrderedDict()
    # for key, value in sd1.items():
    #     new_key = key_transformation(key)
    #     new_state_dict1[new_key] = value
    #
    # new_state_dict2 = OrderedDict()
    # for key, value in sd2.items():
    #     new_key = key_transformation(key)
    #     new_state_dict2[new_key] = value
    # sd1 = new_state_dict1
    # sd2 = new_state_dict2
    #
    # w2 = []
    # for key in permuted_oracle_sd1.keys():
    #     param = permuted_oracle_sd1[key]
    #     w2.append(param.cpu().detach().numpy())
    # exp_no = f'tmax{args.tmax}_tmin{args.tmin}_steps{args.steps}'
    #
    # winning_permutation = barrier_SA(args.arch, model, sd1, permuted_oracle_sd1, w2, init_state1,
    #                                  args.tmax, args.tmin, args.steps,
    #                                  train_inputs, train_targets,
    #                                  train_avg_org_models, nchannels, nclasses,
    #                                  args.width)
    #
    # print("winning_permutation", winning_permutation)
    # winning_perm_model_sd = permute(args.arch, model, winning_permutation, sd2, w2, nchannels, nclasses, args.width)
    #
    # dict_after = get_barrier(model, sd1, winning_perm_model_sd, train_inputs, train_targets, test_inputs, test_targets)
    #
    # add_element(dict_after, 'winning_permutation', winning_permutation)
    # add_element(dict_after, 'winning_perm_model_sd', winning_perm_model_sd)
    #
    # barrier_train = dict_after['barrier_train']
    # lmc_train = dict_after['train_lmc']
    #
    # print("barrier_train_SA_oracle", barrier_train)
    # print("lmc_train_SA_oracle", lmc_train)
    #
    # # source_file_name = f'dict_after_pair{pair}_{args.rand_seed}.pkl'
    # source_file_name = f'dict_after_pair{pair}.pkl'
    # # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/oracle/SA/auto/{source_file_name}'
    # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/barrier_10pairs/oracle/SA/grid/{exp_no}/{source_file_name}'
    # pickle_out = pickle.dumps(dict_after)
    # upload_pkl(bucket_name, pickle_out, destination_blob_name)
    #
    #







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
    elif arch == 'vgg11_bn':

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
        key = 'features.4.weight'
        param = sd[key]


        w_p = param[:, idx1, :, :]
        w_p = w_p[idx2, :, :, :]
        sd[key][:] = w_p

        key = 'features.4.bias'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.5.weight'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.5.bias'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.5.running_mean'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p

        key = 'features.5.running_var'
        param = sd[key]


        w_p = param[idx2]
        sd[key][:] = w_p
        ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'features.8.weight'
        param = sd[key]


        w_p = param[:, idx2, :, :]
        w_p = w_p[idx3, :, :, :]
        sd[key][:] = w_p

        key = 'features.8.bias'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.9.weight'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.9.bias'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.9.running_mean'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p

        key = 'features.9.running_var'
        param = sd[key]


        w_p = param[idx3]
        sd[key][:] = w_p
        ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'features.11.weight'
        param = sd[key]


        w_p = param[:, idx3, :, :]
        w_p = w_p[idx4, :, :, :]
        sd[key][:] = w_p

        key = 'features.11.bias'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.12.weight'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.12.bias'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.12.running_mean'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p

        key = 'features.12.running_var'
        param = sd[key]


        w_p = param[idx4]
        sd[key][:] = w_p
        ##################################### layer 5
        idx5 = perm_ind[4]
        key = 'features.15.weight'
        param = sd[key]


        w_p = param[:, idx4, :, :]
        w_p = w_p[idx5, :, :, :]
        sd[key][:] = w_p

        key = 'features.15.bias'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.16.weight'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.16.bias'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.16.running_mean'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p

        key = 'features.16.running_var'
        param = sd[key]


        w_p = param[idx5]
        sd[key][:] = w_p
        ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'features.18.weight'
        param = sd[key]


        w_p = param[:, idx5, :, :]
        w_p = w_p[idx6, :, :, :]
        sd[key][:] = w_p

        key = 'features.18.bias'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.19.weight'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.19.bias'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.19.running_mean'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'features.19.running_var'
        param = sd[key]


        w_p = param[idx6]
        sd[key][:] = w_p
        ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'features.22.weight'
        param = sd[key]


        w_p = param[:, idx6, :, :]
        w_p = w_p[idx7, :, :, :]
        sd[key][:] = w_p

        key = 'features.22.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.23.weight'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.23.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.23.running_mean'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.23.running_var'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p
        ##################################### layer 8
        idx8 = perm_ind[7]
        key = 'features.25.weight'
        param = sd[key]


        w_p = param[:, idx7, :, :]
        w_p = w_p[idx8, :, :, :]
        sd[key][:] = w_p

        key = 'features.25.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.26.weight'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.26.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.26.running_mean'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.26.running_var'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p
        ##################################### layer 9 ===== linear
        idx9 = perm_ind[8]
        key = 'classifier.1.weight'
        param = sd[key]


        w_p = param[:, idx8]
        w_p = w_p[idx9, :]
        sd[key][:] = w_p

        key = 'classifier.1.bias'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p
        ##################################### layer 10 ===== linear
        idx10 = perm_ind[9]
        key = 'classifier.4.weight'
        param = sd[key]


        w_p = param[:, idx9]
        w_p = w_p[idx10, :]
        sd[key][:] = w_p

        key = 'classifier.4.bias'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p
        # ##################################### layer 11 ===== linear
        key = 'classifier.6.weight'
        param = sd[key]


        w_p = param[:, idx10]
        # w_p = w_p[idx16, :]
        sd[key][:] = w_p

        key = 'classifier.6.bias'
        param = sd[key]


        # w_p = param   ############################## no change
        # sd[key][:] = w_p

        model = vgg.__dict__[args.arch](nchannels, args.width, nclasses)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    elif arch == 'vgg13_bn':

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
        key = 'features.21.weight'
        param = sd[key]


        w_p = param[:, idx6, :, :]
        w_p = w_p[idx7, :, :, :]
        sd[key][:] = w_p

        key = 'features.21.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.22.weight'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.22.bias'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.22.running_mean'
        param = sd[key]


        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'features.22.running_var'
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
        key = 'features.28.weight'
        param = sd[key]


        w_p = param[:, idx8, :, :]
        w_p = w_p[idx9, :, :, :]
        sd[key][:] = w_p

        key = 'features.28.bias'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.29.weight'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.29.bias'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.29.running_mean'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'features.29.running_var'
        param = sd[key]


        w_p = param[idx9]
        sd[key][:] = w_p
        ##################################### layer 10
        idx10 = perm_ind[9]
        key = 'features.31.weight'
        param = sd[key]


        w_p = param[:, idx9, :, :]
        w_p = w_p[idx10, :, :, :]
        sd[key][:] = w_p

        key = 'features.31.bias'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.32.weight'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.32.bias'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.32.running_mean'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'features.32.running_var'
        param = sd[key]


        w_p = param[idx10]
        sd[key][:] = w_p
        ##################################### layer 14 ===== linear
        idx11 = perm_ind[10]
        key = 'classifier.1.weight'
        param = sd[key]


        w_p = param[:, idx10]
        w_p = w_p[idx11, :]
        sd[key][:] = w_p

        key = 'classifier.1.bias'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p
        ##################################### layer 15 ===== linear
        idx12 = perm_ind[11]
        key = 'classifier.4.weight'
        param = sd[key]


        w_p = param[:, idx11]
        w_p = w_p[idx12, :]
        sd[key][:] = w_p

        key = 'classifier.4.bias'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p
        # ##################################### layer 16 ===== linear
        key = 'classifier.6.weight'
        param = sd[key]


        w_p = param[:, idx12]
        # w_p = w_p[idx16, :]
        sd[key][:] = w_p

        key = 'classifier.6.bias'
        param = sd[key]


        # w_p = param   ############################## no change
        # sd[key][:] = w_p

        model = vgg.__dict__[args.arch](nchannels, args.width, nclasses)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

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

        model = vgg.__dict__[args.arch](nchannels, args.width, nclasses)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    elif arch == 'vgg19_bn':
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
        key = 'features.23.weight'
        param = sd[key]


        w_p = param[:, idx7, :, :]
        w_p = w_p[idx8, :, :, :]
        sd[key][:] = w_p

        key = 'features.23.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.24.weight'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.24.bias'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.24.running_mean'
        param = sd[key]


        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'features.24.running_var'
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
        key = 'features.33.weight'
        param = sd[key]


        w_p = param[:, idx10, :, :]
        w_p = w_p[idx11, :, :, :]
        sd[key][:] = w_p

        key = 'features.33.bias'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.34.weight'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.34.bias'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.34.running_mean'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'features.34.running_var'
        param = sd[key]


        w_p = param[idx11]
        sd[key][:] = w_p
        ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'features.36.weight'
        param = sd[key]


        w_p = param[:, idx11, :, :]
        w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        key = 'features.36.bias'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.37.weight'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.37.bias'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.37.running_mean'
        param = sd[key]


        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'features.37.running_var'
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

        ##################################### layer 14
        idx14 = perm_ind[13]
        key = 'features.43.weight'
        param = sd[key]


        w_p = param[:, idx13, :, :]
        w_p = w_p[idx14, :, :, :]
        sd[key][:] = w_p

        key = 'features.43.bias'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'features.44.weight'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'features.44.bias'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'features.44.running_mean'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p

        key = 'features.44.running_var'
        param = sd[key]


        w_p = param[idx14]
        sd[key][:] = w_p

        ##################################### layer 15
        idx15 = perm_ind[14]
        key = 'features.46.weight'
        param = sd[key]


        w_p = param[:, idx14, :, :]
        w_p = w_p[idx15, :, :, :]
        sd[key][:] = w_p

        key = 'features.46.bias'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'features.47.weight'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'features.47.bias'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'features.47.running_mean'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'features.47.running_var'
        param = sd[key]


        w_p = param[idx15]
        sd[key][:] = w_p

        ##################################### layer 16
        idx16 = perm_ind[15]
        key = 'features.49.weight'
        param = sd[key]


        w_p = param[:, idx15, :, :]
        w_p = w_p[idx16, :, :, :]
        sd[key][:] = w_p

        key = 'features.49.bias'
        param = sd[key]


        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'features.50.weight'
        param = sd[key]


        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'features.50.bias'
        param = sd[key]


        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'features.50.running_mean'
        param = sd[key]


        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'features.50.running_var'
        param = sd[key]


        w_p = param[idx16]
        sd[key][:] = w_p
        ##################################### layer 17 ===== linear
        idx17 = perm_ind[16]
        key = 'classifier.1.weight'
        param = sd[key]


        w_p = param[:, idx16]
        w_p = w_p[idx17, :]
        sd[key][:] = w_p

        key = 'classifier.1.bias'
        param = sd[key]


        w_p = param[idx17]
        sd[key][:] = w_p
        ##################################### layer 18 ===== linear
        idx18 = perm_ind[17]
        key = 'classifier.4.weight'
        param = sd[key]


        w_p = param[:, idx17]
        w_p = w_p[idx18, :]
        sd[key][:] = w_p

        key = 'classifier.4.bias'
        param = sd[key]

        w_p = param[idx18]
        sd[key][:] = w_p
        # ##################################### layer 19 ===== linear
        key = 'classifier.6.weight'
        param = sd[key]


        w_p = param[:, idx18]
        # w_p = w_p[idx19, :]
        sd[key][:] = w_p

        key = 'classifier.6.bias'
        param = sd[key]


        # w_p = param   ############################## no change
        # sd[key][:] = w_p

        model = vgg.__dict__[args.arch](nchannels, args.width, nclasses)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)




    elif arch == 'resnet18':
        # ##################################### layer 1
        # idx1 = perm_ind[0]
        # key = 'conv1.weight'  # [64, 3, 3, 3]
        # param = sd[key]
        # w_p = param[idx1, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'bn1.weight'
        # param = sd[key]
        # # print(idx1)
        # # print(param)
        # w_p = param[idx1]
        # # print(w_p)
        # sd[key][:] = w_p
        #
        # key = 'bn1.bias'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_mean'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_var'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        # # # ##################################### layer 2
        idx2 = perm_ind[1]
        key = 'layer1.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx1, :, :]
        # w_p = w_p[idx2, :, :, :]
        w_p = param[idx2, :, :, :] ## start from idx2
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
        # ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'layer1.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx2, :, :]
        # w_p = w_p[idx3, :, :, :]
        # w_p = param[idx3, :, :, :]  ## start from idx3
        sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        # # # ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'layer1.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx3, :, :]
        # w_p = w_p[idx4, :, :, :]
        w_p = param[idx4, :, :, :]  ## start from idx4
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
        # # ##################################### layer 5
        idx5 = perm_ind[4]
        key = 'layer1.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx4, :, :]
        # w_p = w_p[idx5, :, :, :]
        # w_p = param[idx5, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'layer2.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx5, :, :]
        # w_p = w_p[idx6, :, :, :]
        w_p = param[idx6, :, :, :]  ### start from idx6
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
        # ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'layer2.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx6, :, :]
        # w_p = w_p[idx7, :, :, :]
        # w_p = param[idx7, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.weight'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        # # # ##################################### layer 8
        # idx8 = perm_ind[7]
        # key = 'layer2.0.shortcut.0.weight'
        # param = sd[key]
        # # print(param.shape) ## (128, 64, 1, 1)
        # # w_p = param[:, idx7, :, :] ###
        # # print(len(idx5), len(idx6)) ## 64,128
        # w_p = param[:, idx5, :, :]  ## layer2.0.conv1.weight
        # # w_p = w_p[idx7, :, :, :]  #### layer2.0.conv2.weight
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx6]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx7]
        # sd[key][:] = w_p
        # ##################################### layer 9
        idx9 = perm_ind[8]
        key = 'layer2.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx7, :, :]
        # w_p = w_p[idx9, :, :, :]
        w_p = param[idx9, :, :, :]
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
        # ##################################### layer 10
        idx10 = perm_ind[9]
        key = 'layer2.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx9, :, :]
        # w_p = w_p[idx10, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx10]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx10]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx10]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx10]
        # sd[key][:] = w_p
        # ##################################### layer 11
        idx11 = perm_ind[10]
        key = 'layer3.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx10, :, :]
        # w_p = w_p[idx11, :, :, :]
        w_p = param[idx11, :, :, :]
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
        # ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'layer3.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx11, :, :]
        # w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx12]
        # sd[key][:] = w_p
        # ##################################### layer 13 ===================== shortcut
        # idx13 = perm_ind[12]
        # key = 'layer3.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx12, :, :]
        # w_p = param[:, idx10, :, :]  ## layer3.0.conv1.weight
        # w_p = w_p[idx12, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        # ##################################### layer 14
        idx14 = perm_ind[13]
        key = 'layer3.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx12, :, :]
        # w_p = w_p[idx14, :, :, :]
        w_p = param[idx14, :, :, :]
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
        # ##################################### layer 15
        idx15 = perm_ind[14]
        key = 'layer3.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx14, :, :]
        # w_p = w_p[idx15, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx15]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx15]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx15]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx15]
        # sd[key][:] = w_p
        # ##################################### layer 16
        idx16 = perm_ind[15]
        key = 'layer4.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx15, :, :]
        # w_p = w_p[idx16, :, :, :]
        w_p = param[idx16, :, :, :]
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
        # ##################################### layer 17
        idx17 = perm_ind[16]
        key = 'layer4.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx16, :, :]
        # w_p = w_p[idx17, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx17]
        # sd[key][:] = w_p
        # ##################################### layer 18   SHORTCUT
        # idx18 = perm_ind[17]
        # key = 'layer4.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx17, :, :]
        # w_p = param[:, idx15, :, :]  ### layer4.0.conv1.weight
        # w_p = w_p[idx17, :, :, :]  ### layer4.0.conv2.weight
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx17]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx17]
        # sd[key][:] = w_p
        # ##################################### layer 19
        idx19 = perm_ind[18]
        key = 'layer4.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx17, :, :]
        # w_p = w_p[idx19, :, :, :]
        w_p = param[idx19, :, :, :]
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
        # w_p = w_p[idx20, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx20]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx20]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx20]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx20]
        # sd[key][:] = w_p
        # # ##################################### layer 21 ===== linear
        # key = 'linear.weight'
        # param = sd[key]
        #
        # w_p = param[:, idx20]
        # # w_p = w_p[idx16, :]
        # sd[key][:] = w_p
        #
        # key = 'linear.bias'
        # param = sd[key]
        #
        # # w_p = param   ############################## no change
        # # sd[key][:] = w_p


        model = ResNet18(nclasses, args.width, nchannels)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    elif arch == 'resnet34':
        # ##################################### layer 1
        # idx1 = perm_ind[0]
        # key = 'conv1.weight'  # [64, 3, 3, 3]
        # param = sd[key]
        # w_p = param[idx1, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'bn1.weight'
        # param = sd[key]
        # # print(idx1)
        # # print(param)
        # w_p = param[idx1]
        # # print(w_p)
        # sd[key][:] = w_p
        #
        # key = 'bn1.bias'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_mean'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_var'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        # # # ##################################### layer 2
        idx2 = perm_ind[1]
        key = 'layer1.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx1, :, :]
        # w_p = w_p[idx2, :, :, :]
        w_p = param[idx2, :, :, :] ## start from idx2
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
        # ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'layer1.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx2, :, :]
        # w_p = w_p[idx3, :, :, :]
        # w_p = param[idx3, :, :, :]  ## start from idx3
        sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx3]
        # sd[key][:] = w_p
        # # # ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'layer1.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx3, :, :]
        # w_p = w_p[idx4, :, :, :]
        w_p = param[idx4, :, :, :]  ## start from idx4
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
        # # ##################################### layer 5
        idx5 = perm_ind[4]
        key = 'layer1.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx4, :, :]
        # w_p = w_p[idx5, :, :, :]
        # w_p = param[idx5, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        # # # ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'layer1.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx5, :, :]
        # w_p = w_p[idx6, :, :, :]
        w_p = param[idx6, :, :, :]  ## start from idx6
        sd[key][:] = w_p

        key = 'layer1.2.bn1.weight'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.bias'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p
        # # ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'layer1.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx6, :, :]
        # w_p = w_p[idx7, :, :, :]
        # w_p = param[idx7, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer1.2.bn2.weight'
        # param = sd[key]
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn2.bias'
        # param = sd[key]
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx7]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx7]
        # sd[key][:] = w_p



        ##################################### layer 8
        idx8 = perm_ind[7]
        key = 'layer2.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx7, :, :]
        # w_p = w_p[idx8, :, :, :]
        w_p = param[idx8, :, :, :]  ### start from idx6
        sd[key][:] = w_p

        key = 'layer2.0.bn1.weight'
        param = sd[key]
        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.bias'
        param = sd[key]
        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx8]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx8]
        sd[key][:] = w_p
        # ##################################### layer 9
        idx9 = perm_ind[8]
        key = 'layer2.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx8, :, :]
        # w_p = w_p[idx9, :, :, :]
        # w_p = param[idx9, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.weight'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        # # # ##################################### layer 10
        # idx10 = perm_ind[9]
        # key = 'layer2.0.shortcut.0.weight'
        # param = sd[key]
        # w_p = param[:, idx7, :, :]
        # # w_p = w_p[idx9, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx9]
        # sd[key][:] = w_p
        # ##################################### layer 11
        idx11 = perm_ind[10]
        key = 'layer2.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx9, :, :]
        # w_p = w_p[idx11, :, :, :]
        w_p = param[idx11, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.weight'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.bias'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx11]
        sd[key][:] = w_p
        # ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'layer2.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx11, :, :]
        # w_p = w_p[idx12, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx12]
        # sd[key][:] = w_p
        # ##################################### layer 13
        idx13 = perm_ind[12]
        key = 'layer2.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx11, :, :]
        # w_p = w_p[idx13, :, :, :]
        w_p = param[idx13, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.weight'
        param = sd[key]

        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.bias'
        param = sd[key]

        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.running_mean'
        param = sd[key]

        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.running_var'
        param = sd[key]

        w_p = param[idx13]
        sd[key][:] = w_p
        # ##################################### layer 14
        idx14 = perm_ind[13]
        key = 'layer2.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx13, :, :]
        # w_p = w_p[idx14, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.2.bn2.weight'
        # param = sd[key]
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        # ##################################### layer 15
        idx15 = perm_ind[14]
        key = 'layer2.3.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx13, :, :]
        # w_p = w_p[idx15, :, :, :]
        w_p = param[idx15, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.weight'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.bias'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.running_mean'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.running_var'
        param = sd[key]

        w_p = param[idx15]
        sd[key][:] = w_p
        # ##################################### layer 16
        idx16 = perm_ind[15]
        key = 'layer2.3.conv2.weight'
        param = sd[key]
        w_p = param[:, idx15, :, :]
        # w_p = w_p[idx16, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.3.bn2.weight'
        # param = sd[key]
        # w_p = param[idx16]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn2.bias'
        # param = sd[key]
        #
        # w_p = param[idx16]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn2.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx16]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn2.running_var'
        # param = sd[key]
        #
        # w_p = param[idx16]
        # sd[key][:] = w_p
        # ##################################### layer 17
        idx17 = perm_ind[16]
        key = 'layer3.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx16, :, :]
        # w_p = w_p[idx17, :, :, :]
        w_p = param[idx17, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.weight'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.bias'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p
        # ##################################### layer 18
        idx18 = perm_ind[17]
        key = 'layer3.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx17, :, :]
        # w_p = w_p[idx18, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx18]
        # sd[key][:] = w_p
        # ##################################### layer 19 ===================== shortcut
        # idx19 = perm_ind[18]
        # key = 'layer3.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx18, :, :]
        # w_p = param[:, idx16, :, :]  ## layer3.0.conv1.weight
        # w_p = w_p[idx18, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        # ##################################### layer 20
        idx20 = perm_ind[19]
        key = 'layer3.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx18, :, :]
        # w_p = w_p[idx20, :, :, :]
        w_p = param[idx20, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.weight'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.bias'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_mean'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_var'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p
        # ##################################### layer 21
        idx21 = perm_ind[20]
        key = 'layer3.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx20, :, :]
        # w_p = w_p[idx21, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx21]
        # sd[key][:] = w_p
        # ##################################### layer 22
        idx22 = perm_ind[21]
        key = 'layer3.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx21, :, :]
        # w_p = w_p[idx22, :, :, :]
        w_p = param[idx22, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.weight'
        param = sd[key]
        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.bias'
        param = sd[key]
        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx22]
        sd[key][:] = w_p
        # ##################################### layer 23
        idx23 = perm_ind[22]
        key = 'layer3.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx22, :, :]
        # w_p = w_p[idx23, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.2.bn2.weight'
        # param = sd[key]
        # w_p = param[idx23]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn2.bias'
        # param = sd[key]
        # w_p = param[idx23]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx23]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx23]
        # sd[key][:] = w_p
        # ##################################### layer 24
        idx24 = perm_ind[23]
        key = 'layer3.3.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx23, :, :]
        # w_p = w_p[idx24, :, :, :]
        w_p = param[idx24, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.weight'
        param = sd[key]
        w_p = param[idx24]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.bias'
        param = sd[key]
        w_p = param[idx24]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.running_mean'
        param = sd[key]
        w_p = param[idx24]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.running_var'
        param = sd[key]
        w_p = param[idx24]
        sd[key][:] = w_p
        # ##################################### layer 25
        idx25 = perm_ind[24]
        key = 'layer3.3.conv2.weight'
        param = sd[key]
        w_p = param[:, idx24, :, :]
        # w_p = w_p[idx25, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.3.bn2.weight'
        # param = sd[key]
        # w_p = param[idx25]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn2.bias'
        # param = sd[key]
        # w_p = param[idx25]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx25]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx25]
        # sd[key][:] = w_p

        # ##################################### layer 26
        idx26 = perm_ind[25]
        key = 'layer3.4.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx25, :, :]
        # w_p = w_p[idx26, :, :, :]
        w_p = param[idx26, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.weight'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.bias'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.running_mean'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.running_var'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p
        # ##################################### layer 27
        idx27 = perm_ind[26]
        key = 'layer3.4.conv2.weight'
        param = sd[key]
        w_p = param[:, idx26, :, :]
        # w_p = w_p[idx27, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.4.bn2.weight'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn2.bias'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p

        # ##################################### layer 28
        idx28 = perm_ind[27]
        key = 'layer3.5.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx27, :, :]
        # w_p = w_p[idx28, :, :, :]
        w_p = param[idx28, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.weight'
        param = sd[key]
        w_p = param[idx28]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.bias'
        param = sd[key]
        w_p = param[idx28]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.running_mean'
        param = sd[key]
        w_p = param[idx28]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.running_var'
        param = sd[key]
        w_p = param[idx28]
        sd[key][:] = w_p
        # ##################################### layer 29
        idx29 = perm_ind[26]
        key = 'layer3.5.conv2.weight'
        param = sd[key]
        w_p = param[:, idx28, :, :]
        # w_p = w_p[idx29, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.5.bn2.weight'
        # param = sd[key]
        # w_p = param[idx29]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn2.bias'
        # param = sd[key]
        # w_p = param[idx29]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx29]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx29]
        # sd[key][:] = w_p
        # ##################################### layer 30
        idx30 = perm_ind[29]
        key = 'layer4.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx29, :, :]
        # w_p = w_p[idx30, :, :, :]
        w_p = param[idx30, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.weight'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.bias'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p
        # ##################################### layer 31
        idx31 = perm_ind[30]
        key = 'layer4.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx30, :, :]
        # w_p = w_p[idx31, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.0.bn2.weight'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.bias'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        # ##################################### layer 32   SHORTCUT
        # idx32 = perm_ind[31]
        # key = 'layer4.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx31, :, :]
        # w_p = param[:, idx29, :, :]  ### layer4.0.conv1.weight
        # w_p = w_p[idx31, :, :, :]  ### layer4.0.conv2.weight
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx31]
        # sd[key][:] = w_p
        # ##################################### layer 33
        idx33 = perm_ind[32]
        key = 'layer4.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx31, :, :]
        # w_p = w_p[idx33, :, :, :]
        w_p = param[idx33, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.weight'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.bias'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_mean'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_var'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p
        ##################################### layer 34
        idx34 = perm_ind[33]
        key = 'layer4.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx33, :, :]
        # w_p = w_p[idx34, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.1.bn2.weight'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.bias'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        # ##################################### layer 35
        idx35 = perm_ind[34]
        key = 'layer4.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx31, :, :]
        # w_p = w_p[idx35, :, :, :]
        w_p = param[idx35, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.weight'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.bias'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p
        ##################################### layer 36
        idx36 = perm_ind[35]
        key = 'layer4.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx35, :, :]
        # w_p = w_p[idx36, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.2.bn2.weight'
        # param = sd[key]
        # w_p = param[idx36]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn2.bias'
        # param = sd[key]
        # w_p = param[idx36]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn2.running_mean'
        # param = sd[key]
        # w_p = param[idx36]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn2.running_var'
        # param = sd[key]
        # w_p = param[idx36]
        # sd[key][:] = w_p
        # # ##################################### layer 37 ===== linear
        # key = 'linear.weight'
        # param = sd[key]
        #
        # w_p = param[:, idx36]
        # # w_p = w_p[idx16, :]
        # sd[key][:] = w_p
        #
        # key = 'linear.bias'
        # param = sd[key]
        #
        # # w_p = param   ############################## no change
        # # sd[key][:] = w_p


        model = ResNet34(nclasses, args.width, nchannels)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    elif arch == 'resnet50':
        # ##################################### layer 1
        # idx1 = perm_ind[0]
        # key = 'conv1.weight'  # [64, 3, 3, 3]
        # param = sd[key]
        # w_p = param[idx1, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'bn1.weight'
        # param = sd[key]
        # # print(idx1)
        # # print(param)
        # w_p = param[idx1]
        # # print(w_p)
        # sd[key][:] = w_p
        #
        # key = 'bn1.bias'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_mean'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        #
        # key = 'bn1.running_var'
        # param = sd[key]
        # w_p = param[idx1]
        # sd[key][:] = w_p
        # # # ##################################### layer 2
        idx2 = perm_ind[1]
        key = 'layer1.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx1, :, :]
        # w_p = w_p[idx2, :, :, :]
        w_p = param[idx2, :, :, :] ## start from idx2
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
        # ##################################### layer 3
        idx3 = perm_ind[2]
        key = 'layer1.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx2, :, :]
        w_p = w_p[idx3, :, :, :]
        # w_p = param[idx3, :, :, :]  ## start from idx3
        sd[key][:] = w_p
        #
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
        # ##################################### layer 4
        idx4 = perm_ind[3]
        key = 'layer1.0.conv3.weight'
        param = sd[key]
        w_p = param[:, idx3, :, :]
        # w_p = w_p[idx4, :, :, :]
        # w_p = param[idx4, :, :, :]  ## start from idx3
        sd[key][:] = w_p
        #
        # key = 'layer1.0.bn3.weight'
        # param = sd[key]
        # w_p = param[idx4]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn3.bias'
        # param = sd[key]
        # w_p = param[idx4]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx4]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx4]
        # sd[key][:] = w_p
        # # # ##################################### layer 5
        # idx5 = perm_ind[4]
        # key = 'layer1.0.shortcut.0.weight'
        # param = sd[key]
        # w_p = param[:, idx3, :, :]
        # # w_p = w_p[idx5, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx5]
        # sd[key][:] = w_p
        #
        # key = 'layer1.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx5]
        # sd[key][:] = w_p
        # # # ##################################### layer 6
        idx6 = perm_ind[5]
        key = 'layer1.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx4, :, :]
        # w_p = w_p[idx6, :, :, :]
        w_p = param[idx6, :, :, :]  ## start from idx6
        sd[key][:] = w_p

        key = 'layer1.1.bn1.weight'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.bias'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.running_mean'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p

        key = 'layer1.1.bn1.running_var'
        param = sd[key]
        w_p = param[idx6]
        sd[key][:] = w_p
        # # ##################################### layer 7
        idx7 = perm_ind[6]
        key = 'layer1.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx6, :, :]
        w_p = w_p[idx7, :, :, :]
        # w_p = param[idx7, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.weight'
        param = sd[key]
        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.bias'
        param = sd[key]
        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.running_mean'
        param = sd[key]
        w_p = param[idx7]
        sd[key][:] = w_p

        key = 'layer1.1.bn2.running_var'
        param = sd[key]
        w_p = param[idx7]
        sd[key][:] = w_p
        # ##################################### layer 8
        idx8 = perm_ind[7]
        key = 'layer1.1.conv3.weight'
        param = sd[key]
        w_p = param[:, idx7, :, :]
        # w_p = w_p[idx8, :, :, :]
        # w_p = param[idx8, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer1.1.bn3.weight'
        # param = sd[key]
        # w_p = param[idx8]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn3.bias'
        # param = sd[key]
        # w_p = param[idx8]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx8]
        # sd[key][:] = w_p
        #
        # key = 'layer1.1.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx8]
        # sd[key][:] = w_p
        # # # ##################################### layer 9
        idx9 = perm_ind[8]
        key = 'layer1.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx8, :, :]
        # w_p = w_p[idx9, :, :, :]
        w_p = param[idx9, :, :, :]  ## start from idx9
        sd[key][:] = w_p

        key = 'layer1.2.bn1.weight'
        param = sd[key]
        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.bias'
        param = sd[key]
        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx9]
        sd[key][:] = w_p

        key = 'layer1.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx9]
        sd[key][:] = w_p
        # # ##################################### layer 10
        idx10 = perm_ind[9]
        key = 'layer1.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx9, :, :]
        w_p = w_p[idx10, :, :, :]
        # w_p = param[idx10, :, :, :]
        sd[key][:] = w_p

        key = 'layer1.2.bn2.weight'
        param = sd[key]
        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer1.2.bn2.bias'
        param = sd[key]
        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer1.2.bn2.running_mean'
        param = sd[key]
        w_p = param[idx10]
        sd[key][:] = w_p

        key = 'layer1.2.bn2.running_var'
        param = sd[key]
        w_p = param[idx10]
        sd[key][:] = w_p
        # ##################################### layer 11
        idx11 = perm_ind[10]
        key = 'layer1.2.conv3.weight'
        param = sd[key]
        w_p = param[:, idx10, :, :]
        # w_p = w_p[idx11, :, :, :]
        # w_p = param[idx11, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer1.2.bn3.weight'
        # param = sd[key]
        # w_p = param[idx11]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn3.bias'
        # param = sd[key]
        # w_p = param[idx11]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx11]
        # sd[key][:] = w_p
        #
        # key = 'layer1.2.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx11]
        # sd[key][:] = w_p
        ##################################### layer 12
        idx12 = perm_ind[11]
        key = 'layer2.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx11, :, :]
        # w_p = w_p[idx12, :, :, :]
        w_p = param[idx12, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.weight'
        param = sd[key]
        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.bias'
        param = sd[key]
        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx12]
        sd[key][:] = w_p

        key = 'layer2.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx12]
        sd[key][:] = w_p
        # ##################################### layer 13
        idx13 = perm_ind[12]
        key = 'layer2.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx12, :, :]
        w_p = w_p[idx13, :, :, :]
        # w_p = param[idx13, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.weight'
        param = sd[key]
        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.bias'
        param = sd[key]
        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.running_mean'
        param = sd[key]
        w_p = param[idx13]
        sd[key][:] = w_p

        key = 'layer2.0.bn2.running_var'
        param = sd[key]
        w_p = param[idx13]
        sd[key][:] = w_p
        # ##################################### layer 14
        idx14 = perm_ind[13]
        key = 'layer2.0.conv3.weight'
        param = sd[key]
        w_p = param[:, idx13, :, :]
        # w_p = w_p[idx14, :, :, :]
        # w_p = param[idx14, :, :, :]
        sd[key][:] = w_p
        #
        # key = 'layer2.0.bn3.weight'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn3.bias'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn3.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.bn3.running_var'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        # # # ##################################### layer 15
        # idx15 = perm_ind[14]
        # key = 'layer2.0.shortcut.0.weight'
        # param = sd[key]
        # w_p = param[:, idx12, :, :]
        # # w_p = w_p[idx14, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        #
        # key = 'layer2.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx14]
        # sd[key][:] = w_p
        # ##################################### layer 16
        idx16 = perm_ind[15]
        key = 'layer2.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx14, :, :]
        # w_p = w_p[idx16, :, :, :]
        w_p = param[idx16, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.weight'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.bias'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_mean'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p

        key = 'layer2.1.bn1.running_var'
        param = sd[key]

        w_p = param[idx16]
        sd[key][:] = w_p
        # ##################################### layer 17
        idx17 = perm_ind[16]
        key = 'layer2.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx16, :, :]
        w_p = w_p[idx17, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.weight'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.bias'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.running_mean'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p

        key = 'layer2.1.bn2.running_var'
        param = sd[key]
        w_p = param[idx17]
        sd[key][:] = w_p
        ##################################### layer 18
        idx18 = perm_ind[17]
        key = 'layer2.1.conv3.weight'
        param = sd[key]
        w_p = param[:, idx17, :, :]
        # w_p = w_p[idx18, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.1.bn3.weight'
        # param = sd[key]
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn3.bias'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn3.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        #
        # key = 'layer2.1.bn3.running_var'
        # param = sd[key]
        #
        # w_p = param[idx18]
        # sd[key][:] = w_p
        # ##################################### layer 19
        idx19 = perm_ind[18]
        key = 'layer2.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx18, :, :]
        # w_p = w_p[idx19, :, :, :]
        w_p = param[idx19, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.weight'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.bias'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.running_mean'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p

        key = 'layer2.2.bn1.running_var'
        param = sd[key]

        w_p = param[idx19]
        sd[key][:] = w_p
        # ##################################### layer 20
        idx20 = perm_ind[19]
        key = 'layer2.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx19, :, :]
        w_p = w_p[idx20, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.2.bn2.weight'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer2.2.bn2.bias'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer2.2.bn2.running_mean'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p

        key = 'layer2.2.bn2.running_var'
        param = sd[key]
        w_p = param[idx20]
        sd[key][:] = w_p
        # ##################################### layer 21
        idx21 = perm_ind[20]
        key = 'layer2.2.conv3.weight'
        param = sd[key]
        w_p = param[:, idx20, :, :]
        # w_p = w_p[idx21, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.2.bn3.weight'
        # param = sd[key]
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn3.bias'
        # param = sd[key]
        #
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn3.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx21]
        # sd[key][:] = w_p
        #
        # key = 'layer2.2.bn3.running_var'
        # param = sd[key]
        #
        # w_p = param[idx21]
        # sd[key][:] = w_p
        # ##################################### layer 22
        idx22 = perm_ind[21]
        key = 'layer2.3.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx21, :, :]
        # w_p = w_p[idx22, :, :, :]
        w_p = param[idx22, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.weight'
        param = sd[key]

        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.bias'
        param = sd[key]

        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.running_mean'
        param = sd[key]

        w_p = param[idx22]
        sd[key][:] = w_p

        key = 'layer2.3.bn1.running_var'
        param = sd[key]

        w_p = param[idx22]
        sd[key][:] = w_p
        # ##################################### layer 23
        idx23 = perm_ind[22]
        key = 'layer2.3.conv2.weight'
        param = sd[key]
        w_p = param[:, idx22, :, :]
        w_p = w_p[idx23, :, :, :]
        sd[key][:] = w_p

        key = 'layer2.3.bn2.weight'
        param = sd[key]
        w_p = param[idx23]
        sd[key][:] = w_p

        key = 'layer2.3.bn2.bias'
        param = sd[key]
        w_p = param[idx23]
        sd[key][:] = w_p

        key = 'layer2.3.bn2.running_mean'
        param = sd[key]
        w_p = param[idx23]
        sd[key][:] = w_p

        key = 'layer2.3.bn2.running_var'
        param = sd[key]
        w_p = param[idx23]
        sd[key][:] = w_p
        ##################################### layer 24
        idx24 = perm_ind[23]
        key = 'layer2.3.conv3.weight'
        param = sd[key]
        w_p = param[:, idx23, :, :]
        # w_p = w_p[idx24, :, :, :]
        sd[key][:] = w_p

        # key = 'layer2.3.bn3.weight'
        # param = sd[key]
        # w_p = param[idx24]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn3.bias'
        # param = sd[key]
        #
        # w_p = param[idx24]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn3.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx24]
        # sd[key][:] = w_p
        #
        # key = 'layer2.3.bn3.running_var'
        # param = sd[key]
        #
        # w_p = param[idx24]
        # sd[key][:] = w_p
        # ##################################### layer 25
        idx25 = perm_ind[24]
        key = 'layer3.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx24, :, :]
        # w_p = w_p[idx25, :, :, :]
        w_p = param[idx25, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.weight'
        param = sd[key]
        w_p = param[idx25]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.bias'
        param = sd[key]
        w_p = param[idx25]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx25]
        sd[key][:] = w_p

        key = 'layer3.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx25]
        sd[key][:] = w_p
        # ##################################### layer 26
        idx26 = perm_ind[25]
        key = 'layer3.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx25, :, :]
        w_p = w_p[idx26, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.weight'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.bias'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.running_mean'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p

        key = 'layer3.0.bn2.running_var'
        param = sd[key]
        w_p = param[idx26]
        sd[key][:] = w_p
        ##################################### layer 27
        idx27 = perm_ind[26]
        key = 'layer3.0.conv3.weight'
        param = sd[key]
        w_p = param[:, idx26, :, :]
        # w_p = w_p[idx27, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.0.bn3.weight'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn3.bias'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx27]
        # sd[key][:] = w_p
        # ##################################### layer 28 ===================== shortcut
        # idx28 = perm_ind[18]
        # key = 'layer3.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx27, :, :]
        # w_p = param[:, idx25, :, :]  ## layer3.0.conv1.weight
        # w_p = w_p[idx27, :, :, :]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx27]
        # sd[key][:] = w_p
        #
        # key = 'layer3.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx27]
        # sd[key][:] = w_p
        # ##################################### layer 29
        idx29 = perm_ind[28]
        key = 'layer3.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx27, :, :]
        # w_p = w_p[idx29, :, :, :]
        w_p = param[idx29, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.weight'
        param = sd[key]
        w_p = param[idx29]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.bias'
        param = sd[key]
        w_p = param[idx29]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_mean'
        param = sd[key]
        w_p = param[idx29]
        sd[key][:] = w_p

        key = 'layer3.1.bn1.running_var'
        param = sd[key]
        w_p = param[idx29]
        sd[key][:] = w_p
        # ##################################### layer 30
        idx30 = perm_ind[29]
        key = 'layer3.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx29, :, :]
        w_p = w_p[idx30, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.weight'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.bias'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.running_mean'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p

        key = 'layer3.1.bn2.running_var'
        param = sd[key]
        w_p = param[idx30]
        sd[key][:] = w_p
        ##################################### layer 31
        idx31 = perm_ind[30]
        key = 'layer3.1.conv3.weight'
        param = sd[key]
        w_p = param[:, idx30, :, :]
        # w_p = w_p[idx31, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.1.bn3.weight'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn3.bias'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        #
        # key = 'layer3.1.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx31]
        # sd[key][:] = w_p
        # ##################################### layer 32
        idx32 = perm_ind[31]
        key = 'layer3.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx31, :, :]
        # w_p = w_p[idx32, :, :, :]
        w_p = param[idx32, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.weight'
        param = sd[key]
        w_p = param[idx32]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.bias'
        param = sd[key]
        w_p = param[idx32]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx32]
        sd[key][:] = w_p

        key = 'layer3.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx32]
        sd[key][:] = w_p
        # ##################################### layer 33
        idx33 = perm_ind[32]
        key = 'layer3.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx32, :, :]
        w_p = w_p[idx33, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.2.bn2.weight'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer3.2.bn2.bias'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer3.2.bn2.running_mean'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p

        key = 'layer3.2.bn2.running_var'
        param = sd[key]
        w_p = param[idx33]
        sd[key][:] = w_p
        # ##################################### layer 34
        idx34 = perm_ind[33]
        key = 'layer3.2.conv3.weight'
        param = sd[key]
        w_p = param[:, idx33, :, :]
        # w_p = w_p[idx34, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.2.bn3.weight'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn3.bias'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        #
        # key = 'layer3.2.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx34]
        # sd[key][:] = w_p
        # ##################################### layer 35
        idx35 = perm_ind[34]
        key = 'layer3.3.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx34, :, :]
        # w_p = w_p[idx35, :, :, :]
        w_p = param[idx35, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.weight'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.bias'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.running_mean'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p

        key = 'layer3.3.bn1.running_var'
        param = sd[key]
        w_p = param[idx35]
        sd[key][:] = w_p
        # ##################################### layer 36
        idx36 = perm_ind[35]
        key = 'layer3.3.conv2.weight'
        param = sd[key]
        w_p = param[:, idx35, :, :]
        w_p = w_p[idx36, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.3.bn2.weight'
        param = sd[key]
        w_p = param[idx36]
        sd[key][:] = w_p

        key = 'layer3.3.bn2.bias'
        param = sd[key]
        w_p = param[idx36]
        sd[key][:] = w_p

        key = 'layer3.3.bn2.running_mean'
        param = sd[key]
        w_p = param[idx36]
        sd[key][:] = w_p

        key = 'layer3.3.bn2.running_var'
        param = sd[key]
        w_p = param[idx36]
        sd[key][:] = w_p
        # ##################################### layer 37
        idx37 = perm_ind[36]
        key = 'layer3.3.conv3.weight'
        param = sd[key]
        w_p = param[:, idx36, :, :]
        # w_p = w_p[idx37, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.3.bn3.weight'
        # param = sd[key]
        # w_p = param[idx37]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn3.bias'
        # param = sd[key]
        # w_p = param[idx37]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx37]
        # sd[key][:] = w_p
        #
        # key = 'layer3.3.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx37]
        # sd[key][:] = w_p
        # ##################################### layer 38
        idx38 = perm_ind[37]
        key = 'layer3.4.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx37, :, :]
        # w_p = w_p[idx38, :, :, :]
        w_p = param[idx38, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.weight'
        param = sd[key]
        w_p = param[idx38]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.bias'
        param = sd[key]
        w_p = param[idx38]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.running_mean'
        param = sd[key]
        w_p = param[idx38]
        sd[key][:] = w_p

        key = 'layer3.4.bn1.running_var'
        param = sd[key]
        w_p = param[idx38]
        sd[key][:] = w_p
        # ##################################### layer 39
        idx39 = perm_ind[38]
        key = 'layer3.4.conv2.weight'
        param = sd[key]
        w_p = param[:, idx38, :, :]
        w_p = w_p[idx39, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.4.bn2.weight'
        param = sd[key]
        w_p = param[idx39]
        sd[key][:] = w_p

        key = 'layer3.4.bn2.bias'
        param = sd[key]
        w_p = param[idx39]
        sd[key][:] = w_p

        key = 'layer3.4.bn2.running_mean'
        param = sd[key]
        w_p = param[idx39]
        sd[key][:] = w_p

        key = 'layer3.4.bn2.running_var'
        param = sd[key]
        w_p = param[idx39]
        sd[key][:] = w_p
        ##################################### layer 40
        idx40 = perm_ind[39]
        key = 'layer3.4.conv3.weight'
        param = sd[key]
        w_p = param[:, idx39, :, :]
        # w_p = w_p[idx40, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.4.bn3.weight'
        # param = sd[key]
        # w_p = param[idx40]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn3.bias'
        # param = sd[key]
        # w_p = param[idx40]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx40]
        # sd[key][:] = w_p
        #
        # key = 'layer3.4.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx40]
        # sd[key][:] = w_p
        # ##################################### layer 41
        idx41 = perm_ind[40]
        key = 'layer3.5.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx40, :, :]
        # w_p = w_p[idx41, :, :, :]
        w_p = param[idx41, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.weight'
        param = sd[key]
        w_p = param[idx41]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.bias'
        param = sd[key]
        w_p = param[idx41]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.running_mean'
        param = sd[key]
        w_p = param[idx41]
        sd[key][:] = w_p

        key = 'layer3.5.bn1.running_var'
        param = sd[key]
        w_p = param[idx41]
        sd[key][:] = w_p
        # ##################################### layer 42
        idx42 = perm_ind[41]
        key = 'layer3.5.conv2.weight'
        param = sd[key]
        w_p = param[:, idx41, :, :]
        w_p = w_p[idx42, :, :, :]
        sd[key][:] = w_p

        key = 'layer3.5.bn2.weight'
        param = sd[key]
        w_p = param[idx42]
        sd[key][:] = w_p

        key = 'layer3.5.bn2.bias'
        param = sd[key]
        w_p = param[idx42]
        sd[key][:] = w_p

        key = 'layer3.5.bn2.running_mean'
        param = sd[key]
        w_p = param[idx42]
        sd[key][:] = w_p

        key = 'layer3.5.bn2.running_var'
        param = sd[key]
        w_p = param[idx42]
        sd[key][:] = w_p
        ##################################### layer 43
        idx43 = perm_ind[42]
        key = 'layer3.5.conv3.weight'
        param = sd[key]
        w_p = param[:, idx42, :, :]
        # w_p = w_p[idx43, :, :, :]
        sd[key][:] = w_p

        # key = 'layer3.5.bn3.weight'
        # param = sd[key]
        # w_p = param[idx43]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn3.bias'
        # param = sd[key]
        # w_p = param[idx43]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx43]
        # sd[key][:] = w_p
        #
        # key = 'layer3.5.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx43]
        # sd[key][:] = w_p
        # ##################################### layer 44
        idx44 = perm_ind[43]
        key = 'layer4.0.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx43, :, :]
        # w_p = w_p[idx44, :, :, :]
        w_p = param[idx44, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.weight'
        param = sd[key]
        w_p = param[idx44]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.bias'
        param = sd[key]
        w_p = param[idx44]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_mean'
        param = sd[key]
        w_p = param[idx44]
        sd[key][:] = w_p

        key = 'layer4.0.bn1.running_var'
        param = sd[key]
        w_p = param[idx44]
        sd[key][:] = w_p
        # ##################################### layer 45
        idx45 = perm_ind[44]
        key = 'layer4.0.conv2.weight'
        param = sd[key]
        w_p = param[:, idx44, :, :]
        w_p = w_p[idx45, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.weight'
        param = sd[key]
        w_p = param[idx45]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.bias'
        param = sd[key]
        w_p = param[idx45]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.running_mean'
        param = sd[key]
        w_p = param[idx45]
        sd[key][:] = w_p

        key = 'layer4.0.bn2.running_var'
        param = sd[key]
        w_p = param[idx45]
        sd[key][:] = w_p
        # ##################################### layer 46
        idx46 = perm_ind[45]
        key = 'layer4.0.conv3.weight'
        param = sd[key]
        w_p = param[:, idx45, :, :]
        # w_p = w_p[idx46, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.0.bn3.weight'
        # param = sd[key]
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn3.bias'
        # param = sd[key]
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx46]
        # sd[key][:] = w_p

        # ##################################### layer 47   SHORTCUT
        # idx47 = perm_ind[46]
        # key = 'layer4.0.shortcut.0.weight'
        # param = sd[key]
        #
        # # w_p = param[:, idx46, :, :]
        # w_p = param[:, idx44, :, :]  ### layer4.0.conv1.weight
        # w_p = w_p[idx45, :, :, :]  ### layer4.0.conv2.weight
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.weight'
        # param = sd[key]
        #
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.bias'
        # param = sd[key]
        #
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_mean'
        # param = sd[key]
        #
        # w_p = param[idx46]
        # sd[key][:] = w_p
        #
        # key = 'layer4.0.shortcut.1.running_var'
        # param = sd[key]
        #
        # w_p = param[idx46]
        # sd[key][:] = w_p
        # ##################################### layer 48
        idx48 = perm_ind[47]
        key = 'layer4.1.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx46, :, :]
        # w_p = w_p[idx48, :, :, :]
        w_p = param[idx48, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.weight'
        param = sd[key]
        w_p = param[idx48]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.bias'
        param = sd[key]
        w_p = param[idx48]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_mean'
        param = sd[key]
        w_p = param[idx48]
        sd[key][:] = w_p

        key = 'layer4.1.bn1.running_var'
        param = sd[key]
        w_p = param[idx48]
        sd[key][:] = w_p
        # ##################################### layer 49
        idx49 = perm_ind[48]
        key = 'layer4.1.conv2.weight'
        param = sd[key]
        w_p = param[:, idx48, :, :]
        w_p = w_p[idx49, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.weight'
        param = sd[key]
        w_p = param[idx49]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.bias'
        param = sd[key]
        w_p = param[idx49]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.running_mean'
        param = sd[key]
        w_p = param[idx49]
        sd[key][:] = w_p

        key = 'layer4.1.bn2.running_var'
        param = sd[key]
        w_p = param[idx49]
        sd[key][:] = w_p
        # ##################################### layer 50
        idx50 = perm_ind[49]
        key = 'layer4.1.conv3.weight'
        param = sd[key]
        w_p = param[:, idx49, :, :]
        # w_p = w_p[idx50, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.1.bn3.weight'
        # param = sd[key]
        # w_p = param[idx50]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn3.bias'
        # param = sd[key]
        # w_p = param[idx50]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx50]
        # sd[key][:] = w_p
        #
        # key = 'layer4.1.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx50]
        # sd[key][:] = w_p
        # ##################################### layer 51
        idx51 = perm_ind[50]
        key = 'layer4.2.conv1.weight'
        param = sd[key]
        # w_p = param[:, idx50, :, :]
        # w_p = w_p[idx51, :, :, :]
        w_p = param[idx51, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.weight'
        param = sd[key]
        w_p = param[idx51]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.bias'
        param = sd[key]
        w_p = param[idx51]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.running_mean'
        param = sd[key]
        w_p = param[idx51]
        sd[key][:] = w_p

        key = 'layer4.2.bn1.running_var'
        param = sd[key]
        w_p = param[idx51]
        sd[key][:] = w_p
        ##################################### layer 52
        idx52 = perm_ind[51]
        key = 'layer4.2.conv2.weight'
        param = sd[key]
        w_p = param[:, idx51, :, :]
        w_p = w_p[idx52, :, :, :]
        sd[key][:] = w_p

        key = 'layer4.2.bn2.weight'
        param = sd[key]
        w_p = param[idx52]
        sd[key][:] = w_p

        key = 'layer4.2.bn2.bias'
        param = sd[key]
        w_p = param[idx52]
        sd[key][:] = w_p

        key = 'layer4.2.bn2.running_mean'
        param = sd[key]
        w_p = param[idx52]
        sd[key][:] = w_p

        key = 'layer4.2.bn2.running_var'
        param = sd[key]
        w_p = param[idx52]
        sd[key][:] = w_p
        ##################################### layer 53
        idx53 = perm_ind[53]
        key = 'layer4.2.conv3.weight'
        param = sd[key]
        w_p = param[:, idx52, :, :]
        # w_p = w_p[idx53, :, :, :]
        sd[key][:] = w_p

        # key = 'layer4.2.bn3.weight'
        # param = sd[key]
        # w_p = param[idx53]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn3.bias'
        # param = sd[key]
        # w_p = param[idx53]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn3.running_mean'
        # param = sd[key]
        # w_p = param[idx53]
        # sd[key][:] = w_p
        #
        # key = 'layer4.2.bn3.running_var'
        # param = sd[key]
        # w_p = param[idx53]
        # sd[key][:] = w_p
        # # ##################################### layer 54 ===== linear
        # key = 'linear.weight'
        # param = sd[key]
        #
        # w_p = param[:, idx53]
        # # w_p = w_p[idx16, :]
        # sd[key][:] = w_p
        #
        # key = 'linear.bias'
        # param = sd[key]
        #
        # # w_p = param   ############################## no change
        # # sd[key][:] = w_p


        model = ResNet50(nclasses, args.width, nchannels)
        model.load_state_dict(sd)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model.state_dict()






def barrier_SA(arch, model, sd1, sd2, w2, init_state, tmax, tmin, steps, train_inputs, train_targets, train_avg_org_models, nchannels, nclasses, nunits):
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
            for i in range(len(self.state)):
                x = self.state[i]
                a = random.randint(0, len(x) - 1)
                b = random.randint(0, len(x) - 1)
                self.state[i][a], self.state[i][b] = self.state[i][b], self.state[i][a]
            return self.energy() - initial_energy

        def energy(self):
            """Calculates the length of the route."""
            permuted_model_sd = permute(arch, model, self.state, sd2, w2, nchannels, nclasses, nunits)
            ###### LMC
            sd1_ = sd1
            sd2_ = permuted_model_sd
            model.load_state_dict(interpolate_state_dicts(sd1_, sd2_, 0.50))
            result = evaluate_model_small(args, model, train_inputs, train_targets)['top1']
            ####### inf between model 1 and model2, and calculate the barrier to the closest end
            e = train_avg_org_models - result
            # print(e)
            return e


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