import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import argparse
import csv
from google.cloud import storage
from models import *
import csv
import google
import tarfile
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import pickle

# This function trains a fully connected neural net with a singler hidden layer on the given dataset and calculates
# various measures on the learned network.
def main():
    from google.cloud import storage
    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_schedule', default=0, type=int)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--nlayers', default=50, type=int)
    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--scratch', default=1, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args_seed_folder = int(args.seed / 1)
    save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    ######## models
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    if args.dataset == 'imagenet': nclasses = 1000

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

    if "resnet" in args.arch and args.nlayers == 18:
        model = ResNet18(nclasses, args.width, nchannels)
    if "resnet" in args.arch and args.nlayers == 34:
        model = ResNet34(nclasses, args.width, nchannels)
    if "resnet" in args.arch and args.nlayers == 50:
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
    # model = nn.DataParallel(model)
    model = model.to(device)

    print(model)

    ### save all training arguments
    bucket_name = 'permutation-mlp'
    source_file_name = f'args.pkl'
    destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_out = pickle.dumps(args)
    blob.upload_from_string(pickle_out)

    if args.scratch == 1:
        # create a copy of the initial model to be used later
        init_model = copy.deepcopy(model)
        torch.save(init_model.state_dict(), 'model_0.th')
        bucket_name = 'permutation-mlp'
        source_file_name = 'model_0.th'
        destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        init_epoch = 0
    else:
        bucket_name = 'permutation-mlp'
        destination_blob_name = 'model_best.th'
        source_file_name = f'Neurips21_Arxiv/{save_dir}/Train/{args.seed}/{destination_blob_name}'
        print("checkpoint is now loaded!", source_file_name)
        download_blob(bucket_name, source_file_name, destination_blob_name)

        checkpoint = torch.load('model_best.th', map_location=torch.device('cuda'))
        sd1 = checkpoint
        model.load_state_dict(sd1)
        print("checkpoint is now loaded!", source_file_name)

        init_epoch = 999

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum)
    # optimizer = optim.SGD(model.parameters(), args.lr,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # loading data
    if args.dataset == 'MNIST':
        #### get mnist: solution 3
        os.system("wget www.di.ens.fr/~lelarge/MNIST.tar.gz")
        os.system("tar -xvzf MNIST.tar.gz")

        from torchvision.datasets import MNIST
        from torchvision import transforms

        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        train_loader = torch.utils.data.DataLoader(
            MNIST(root='./', download=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((32, 32)),
                                                                           normalize,
                                                                           ]), train=True),
            batch_size=args.batchsize, shuffle=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            MNIST(root='./', download=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((32, 32)),
                                                                           normalize,
                                                                           ]), train=False),
            batch_size=args.batchsize, shuffle=True,
            num_workers=4, pin_memory=True)
    else:
        train_dataset = load_data('train', args.dataset, args.datadir, nchannels)
        val_dataset = load_data('val', args.dataset, args.datadir, nchannels)

        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

    ########### download train/test data/labels to the bucket

    if (args.dataset == 'MNIST'):
        if "mlp" in args.arch:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST_Train_input_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Train_target_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_input_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_target_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
        else:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST3d_Train_input_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Train_target_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_input_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_target_org.pkl'
            destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
    elif (args.dataset == 'CIFAR10'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'SVHN'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'CIFAR100'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_input_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_target_org.pkl'
        destination_blob_name = f'ICLR21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    # elif (args.dataset == 'ImageNet'):
    ###########

    # training the model
    stats = {}
    best_prec1 = 0
    for epoch in range(init_epoch, args.epochs):
        # train for one epoch
        tr_err, tr_loss = train(args, model, device, train_loader, criterion, optimizer, epoch)
        val_err, val_loss, val_margin = validate(args, model, device, val_loader, criterion)

        if args.lr_schedule == 1:
            lr_scheduler.step()
        #### save checkpoint to the Google bucket
        if epoch % 9 == 0:
            torch.save(model.state_dict(), 'model_' + str(epoch+1) + '.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_' + str(epoch+1) + '.th'
            destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)

        print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')
        # to_csv = [epoch+1, tr_err, tr_loss, val_err, val_loss]
        # with open(f"eval.csv", "a") as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow(to_csv)
        # bucket_name = 'permutation-mlp'
        # source_file_name = 'eval.csv'
        # destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
        # upload_blob(bucket_name, source_file_name, destination_blob_name)


        # train_err, train_loss = evaluate_model(args, model, train_inputs, train_targets, train_loader, criterion)
        # test_err, test_loss = evaluate_model(args, model, test_inputs, test_targets, val_loader, criterion)
        #
        # print(train_err, tr_err)
        # print(train_loss, tr_loss)
        # print(test_err, val_err)





        add_element(stats, 'train_err', tr_err)
        add_element(stats, 'train_loss', tr_loss)
        add_element(stats, 'test_err', val_err)
        add_element(stats, 'test_loss', val_loss)
        for param_group in optimizer.param_groups:
            # print('LR', param_group['lr'])
            add_element(stats, 'lr', param_group['lr'])

        #### save best
        is_best = (1-tr_err) > best_prec1
        best_prec1 = max((1-tr_err), best_prec1)
        if is_best:
            torch.save(model.state_dict(), 'model_best.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_best.th'
            destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)

            add_element(stats, 'best_epoch', epoch)
            add_element(stats, 'best_train_err', tr_err)
            add_element(stats, 'best_test_err', val_err)


        from google.cloud import storage
        bucket_name = 'permutation-mlp'
        source_file_name = f'stats.pkl'
        destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_out = pickle.dumps(stats)
        blob.upload_from_string(pickle_out)


        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_loss < args.stopcond:
            torch.save(model.state_dict(), 'model_final.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_final.th'
            destination_blob_name = f'Neurips21_Arxiv/{save_dir}/Train/{args_seed_folder}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)
            break



    # calculate the training error and margin of the learned model
    tr_err, tr_loss, tr_margin = validate(args, model, device, train_loader, criterion)
    print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {tr_margin:.3f}\t ',
            f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')



    # measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
    # for key, value in measure.items():
    #     print(f'{key:s}:\t {float(value):3.3}')


# train the model for one epoch on the given set
def train(args, model, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        if args.arch == 'mlp':
            data, target = data.to(device).view(data.size(0), -1), target.to(device)
        else:
            data = data.to(device)
            target = target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# evaluate the model on the given set
def validate(args, model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if args.arch == 'mlp':
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data = data.to(device)
                target = target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
        val_margin = np.percentile( margin.cpu().numpy(), 5 )

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin


def evaluate_model(args, model, inputs, targets, loader, criterion):
    device = 'cuda'
    sum_loss, sum_correct = 0, 0

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for (data, target) in zip(inputs, targets):
            if args.arch == 'mlp':
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data = data.to(device)
                target = target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

    return 1 - (sum_correct / len(loader.dataset)), sum_loss / len(loader.dataset)


# Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
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
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset_name == 'imagenet':
        tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    if dataset_name == "imagenet":
        imagenet_folder = "ICLR21/small_imagenet"
        imagenet_tarfile = "small_imagenet.tar.gz"
        file = os.path.join(imagenet_folder, imagenet_tarfile)
        if not os.path.isfile(file):
            print("Downloading imagenet...")
            download_blob("permutation-mlp", file, file)
            print("done.")
        if not os.path.isdir(os.path.join(imagenet_folder, split)):
            print("Unpacking...")
            with tarfile.open(file) as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Unpacking - "):
                    tar.extract(member, path=imagenet_folder)
            print("done.")
        dataset = ImageFolder(os.path.join(imagenet_folder, split), transform=tr_transform)

    else:
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

    return dataset



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

def download_blob(bucket_name, source_blob_name, destination_file_name,
                  blob_path_prefix=""):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    try:
        blob.download_to_filename(destination_file_name)
    except google.api_core.exceptions.NotFound as e:
        os.remove(destination_file_name)
        print(e)
        raise FileNotFoundError

def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


if __name__ == '__main__':
    main()
