"""
This file downloads the data (if needed) and creates data loaders

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from torch.utils.data import ConcatDataset, DataLoader, sampler
from torchvision import transforms, datasets
import patient, ecg_mit_bih, dataset_configs, ecg_dataset_pytorch
data_folder = "./data"


def fminst_data():
    """ Get Fashion MNIST data """
    compose = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
        ])
    out_dir = data_folder
    train_data = datasets.FashionMNIST(root=out_dir, train=True, transform=compose, download=True)
    test_data = datasets.FashionMNIST(root=out_dir, train=False, transform=compose, download=True)
    return ConcatDataset([train_data, test_data])


def mnist_data():
    """ Get MNIST data """
    compose = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([.5, ], [.5, ])
        ])
    out_dir = data_folder
    train_data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
    test_data = datasets.MNIST(root=out_dir, train=False, transform=compose, download=True)
    return ConcatDataset([train_data, test_data])


def cifar10_data():
    """ Get CIFAR10 data """
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    out_dir = data_folder
    train_data = datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)
    test_data = datasets.CIFAR10(root=out_dir, train=False, transform=compose, download=True)
    return ConcatDataset([train_data, test_data])


def ecg_mit_bih():
    composed = transforms.Compose([ecg_dataset_pytorch.ToTensor()])
    train_configs = dataset_configs.DatasetConfigs('train', one_vs_all=True, lstm_setting=False,
                                        over_sample_minority_class=False,
                                        under_sample_majority_class=False, only_take_heartbeat_of_type='F',
                                        classified_heartbeat='F')



    test_configs = dataset_configs.DatasetConfigs('test', one_vs_all=True, lstm_setting=False,
                                        over_sample_minority_class=False,
                                        under_sample_majority_class=False, only_take_heartbeat_of_type='F',
                                        classified_heartbeat='F')

    train_data = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(transform=composed, configs=train_configs)
    test_data = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(transform=composed, configs=test_configs)
    return ConcatDataset([train_data, test_data])

def get_loader(batchSize=100, percentage=1, dataset="mnist"):
    """
    This function returns data loaders for the given datasets
    :param batchSize: the size of each batch
    :type batchSize: int
    :param percentage: the percentage of data to use. Must be in range [0, 1]
    :type percentage: float
    :param dataset: the type of dataset to use. One of ("mnist", "fmnist", "cifar")
    :type dataset: str
    :return: data loader with the percentage of data specified
    :rtype: torch.utils.data.DataLoader
    """
    if dataset == "mnist":
        data = mnist_data()
    elif dataset == "fmnist":
        data = fminst_data()
    elif dataset == "cifar":
        data = cifar10_data()
    elif dataset == "ecg":
        data = ecg_mit_bih()
    else:
        raise Exception("dataset name not correct (or not implemented)")
    # get the size of updated data, based on percentage
    indices = [i for i in range(int(percentage * len(data)))]
    loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices))
    return loader
