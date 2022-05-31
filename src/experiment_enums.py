"""
This file contains the enums with the details to run experiments. Some simple examples are given below. In order to
create your own experiment, simply fill all the keys.

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from enum import Enum
from models.generators import GeneratorNet, GeneratorNetCifar10, EcgCNNGenerator, ECGLSTMGenerator
from models.discriminators import DiscriminatorNet, DiscriminatorNetCifar10, EcgCNNDiscriminator, ECGLSTMDiscriminator, IMVTensorLSTM
from torch import nn, optim
from experiment import Experiment

class ExperimentEnums(Enum):

    # ECGLSTM_adversary_wgan_outdistribution = {
    #     "explainable": True,
    #     "explanationType": "adversary",
    #     "generator": ECGLSTMGenerator,
    #     "discriminator": ECGLSTMDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 64,
    #     "percentage": 1,
    #     "g_optim": optim.SGD,
    #     "d_optim": optim.Adam,
    #     "loss": None,
    #     "glr": 1e-3,
    #     "dlr": 1e-3,
    #     "epochs": 200,
    #     "alpha": 2
    # }

    ECGLSTM_Ig_alpha2_switchlast10= {
        "explainable": True,
        "explanationType": "ig",
        "generator": ECGLSTMGenerator,
        "discriminator": ECGLSTMDiscriminator,
        "dataset": "ecg",
        "batchSize": 64,
        "percentage": 1,
        "g_optim": optim.SGD,
        "d_optim": optim.Adam,
        "glr": 0.1,
        "dlr": 1e-3,
        "epochs": 200,
        "alpha": 2
    }

    # ECGCNN_Saliency_alpha5_classifier = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 64,
    #     "percentage": 1,
    #     "g_optim": optim.SGD,
    #     "d_optim": optim.Adam,
    #     "glr": 0.1,
    #     "dlr": 1e-3,
    #     "epochs": 50,
    #     "alpha": 5
    # }

    # ECGCNN_Normal_wgan = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 64,
    #     "percentage": 1,
    #     "g_optim": optim.SGD,
    #     "d_optim": optim.Adam,
    #     "glr": 0.1,
    #     "dlr": 1e-3,
    #     "epochs": 50,
    #     "alpha": None,
    #     "loss": None
    # }

    # ECGCNNAdversary_switch0_alpha2 = {
    #     "explainable": True,
    #     "explanationType": "adversary",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNSaliencyS = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNSaliencyV = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNSaliencyF = {
    #     "explainable": True,
    #     "explanationType": "adversary",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNIg = {
    #     "explainable": True,
    #     "explanationType": "ig",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNSaliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # ECGCNNShap = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": EcgCNNGenerator,
    #     "discriminator": EcgCNNDiscriminator,
    #     "dataset": "ecg",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }

    # FMNIST35Normal = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 0.05,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }
    #
    # # MNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Saliency = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Shap = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # MNIST100Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }
    #
    # FMNIST100Lime = {
    #     "explainable": True,
    #     "explanationType": "lime",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 50
    # }

    # DemoCIFAR = {
    #     "explainable": False,
    #     "explanationType": None,
    #     "generator": GeneratorNetCifar10,
    #     "discriminator": DiscriminatorNetCifar10,
    #     "dataset": "cifar",
    #     "batchSize": 128,
    #     "percentage": 0.5,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }
    #
    # DemoMNIST = {
    #     "explainable": True,
    #     "explanationType": "saliency",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "mnist",
    #     "batchSize": 128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }
    #
    # DemoFMNIST = {
    #     "explainable": True,
    #     "explanationType": "shap",
    #     "generator": GeneratorNet,
    #     "discriminator": DiscriminatorNet,
    #     "dataset": "fmnist",
    #     "batchSize": 128,
    #     "percentage": 0.35,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 5
    # }

    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
