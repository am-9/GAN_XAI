"""
This file contains the experiment class which runs the experiments

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

from get_data import get_loader
from utils.vector_utils import noise, values_target, vectors_to_images, vectors_to_images_cifar, noise_cifar, \
    weights_init
from evaluation.evaluate_generator import calculate_metrics
from evaluation.evaluate_generator_cifar10 import calculate_metrics_cifar
from evaluation.MMD import pairwisedistances, MMDStatistic
from logger import Logger
from utils.explanation_utils import explanation_hook, get_explanation, \
    explanation_hook_cifar, explanation_hook_lstm, get_explanation_IMV
from torch.autograd import Variable
from torch import nn
from torch.optim import lr_scheduler
import torch
import time
import ecg_dataset_pytorch
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance, pearsonr
from fastdtw import fastdtw
import numpy as np
import similaritymeasures
import sys
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.distributions.kl as kl
from models.discriminators import ECGLSTMDiscriminator, EcgCNNDiscriminator
from models.generators import ECGLSTMGenerator, EcgCNNGenerator

class Experiment:
    """ The class that contains the experiment details """
    def __init__(self, experimentType):
        """
        Standard init
        :param experimentType: experiment enum that contains all the data needed to run the experiment
        :type experimentType:
        """
        self.name = experimentType.name
        self.type = experimentType.value
        self.explainable = self.type["explainable"]
        self.explanationType = self.type["explanationType"]

        gen_model_dir = "/Users/alainamahalanabis/Documents/GAN_XAI/src/results/ecg/ECGCNN_Normal/generator.pt"
        self.generator = EcgCNNGenerator()
        self.generator.load_state_dict(torch.load(gen_model_dir), strict=False)
        self.generator.eval()

        disc_model_dir = "/Users/alainamahalanabis/Documents/GAN_XAI/src/results/ecg/ECGCNN_Normal/discriminator.pt"
        self.discriminator = EcgCNNDiscriminator()
        self.discriminator.load_state_dict(torch.load(disc_model_dir), strict=False)
        self.discriminator.eval()

        #self.generator = self.type["generator"]()
        #self.discriminator = self.type["discriminator"]()
        self.g_optim = self.type["g_optim"](self.generator.parameters(), betas=(0.0, 0.9))
        self.d_optim = self.type["d_optim"](self.discriminator.parameters(), betas=(0.0, 0.9))
        #self.g_schedule = lr_scheduler.ExponentialLR(self.g_optim, gamma=0.9)
        #self.d_schedule = lr_scheduler.ExponentialLR(self.d_optim, gamma=0.9)
        #self.loss = self.type["loss"]
        self.epochs = self.type["epochs"]
        self.cuda = False
        self.real_label = 0.9
        self.fake_label = 0.1
        self.samples = 16
        torch.backends.cudnn.benchmark = True
        self.discriminator_iterations = 5
        self.lambda_gp = 10
        self.weight_clip = 0.01
        self.alpha = self.type["alpha"]


    def run(self, logging_frequency=4) -> (list, list):
        """
        This function runs the experiment
        :param logging_frequency: how frequently to log each epoch (default 4)
        :type logging_frequency: int
        :return: None
        :rtype: None
        """

        print ('start experiment')

        start_time = time.time()

        #explanationSwitch = (self.epochs + 1) / 2 if self.epochs % 2 == 1 else self.epochs / 2
        #explanationSwitch=self.epochs - 5
        explanationSwitch=0

        logger = Logger(self.name, self.type["dataset"])

        if self.type["dataset"] == "cifar":
            test_noise = noise_cifar(self.samples, self.cuda)
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)
        else:
            test_noise = noise(self.samples, self.cuda)

        loader = get_loader(self.type["batchSize"], self.type["percentage"], self.type["dataset"])
        num_batches = len(loader)
        print("num batches", num_batches)

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.loss = self.loss.cuda()

        if self.explainable:
            trained_data = next(iter(loader))
            if self.cuda:
                trained_data = trained_data.cuda()
        else:
            trained_data = None

        # track losses
        G_losses = []
        D_losses = []

        DTW = []
        RMSE = []
        WDIST = []
        PEARSON = []
        KL_DIV = []

        local_explainable = False

        # mean = 0.0
        #
        # print ("calculating mean and var")
        # for n_batch, images in enumerate(loader):
        #     images = images['cardiac_cycle'].float()
        #     batch_samples = images.size(0)
        #     images = images.view(batch_samples, images.size(1), -1)
        #     mean += images.mean(2).sum(0)
        # mean = mean / len(loader.dataset)
        #
        # var = 0.0
        # for n_batch, images in enumerate(loader):
        #     images = images['cardiac_cycle'].float()
        #     batch_samples = images.size(0)
        #     images = images.view(batch_samples, images.size(1), -1)
        #     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
        # std = torch.sqrt(var / (len(loader.dataset)*224*224))

        # Start training
        for epoch in range(1, self.epochs + 1):

            if self.explainable and (epoch - 1) == explanationSwitch:
                if self.type["dataset"] == "cifar":
                    self.generator.out.register_backward_hook(explanation_hook_cifar)
                else:
                    if "LSTM" in self.name:
                        self.generator.out.register_backward_hook(explanation_hook_lstm)
                    else:
                        self.generator.out.register_backward_hook(explanation_hook)
                local_explainable = True

            for n_batch, data in enumerate(loader):

                sys.stdout.flush()
                real_batch = data['cardiac_cycle'].float()
                #print (real_batch[1])
                #np.savetxt("./xAIoutput/F_beat_sinput.csv", real_batch[1], delimiter=',')
                #exit(0)
                print ("batch number ", n_batch)
                labels = data['label']
                #print(labels)

                labels_class = torch.max(labels, 1)[1]

                N = real_batch.size(0)
                #print ("N", N)

                # 1. Train Discriminator
                # Generate fake data and detach (so gradients are not calculated for generator)
                if self.type["dataset"] == "cifar":
                    fake_data = self.generator(noise_cifar(N, self.cuda)).detach()
                else:
                    fake_data = self.generator(noise(N, self.cuda)).detach()

                if self.cuda:
                    real_batch = real_batch.cuda()
                    fake_data = fake_data.cuda()


                # Train D
                for _ in range(self.discriminator_iterations):
                    print("train d")
                    fake_data = self.generator(noise(N, self.cuda)).detach()
                    d_error, d_pred_real, d_pred_fake = self._train_discriminator(real_data=real_batch, fake_data=fake_data,
                                                                              local_explainable=local_explainable)
                # 2. Train Generator
                # Generate fake data

                if self.type["dataset"] == "cifar":
                    fake_data = self.generator(noise_cifar(N, self.cuda))
                else:
                    fake_data = self.generator(noise(N, self.cuda))

                if self.cuda:
                    fake_data = fake_data.cuda()

                # Train G
                g_error = self._train_generator(fake_data=fake_data, local_explainable=local_explainable,
                                                trained_data=trained_data)

                #sprint('done generator')

                # Save Losses for plotting later
                G_losses.append(g_error.item())
                D_losses.append(d_error.item())

                logger.log(d_error, g_error, epoch, n_batch, num_batches)

                # Display status Logs
                if n_batch % (num_batches // logging_frequency) == 0:
                    logger.display_status(
                        epoch, self.epochs, n_batch, num_batches,
                        d_error, g_error, d_pred_real, d_pred_fake
                    )

                    #distance, path = fastdtw(real_batch, fake_data, dist=euclidean)

                #for each epoch calculate MMD and DTW
                distance = 0
                rmse = 0
                w_dist = 0
                kl_div = 0
                pearson = 0

                if n_batch == len(loader)-1:

                    for i in range(N):
                        for m in range(N):
                            with torch.no_grad():
                                #print("m ", m, "i ", i)
                                real = real_batch[i, :].numpy()
                                fake = fake_data[m, :].numpy()
                                if "CNN" in self.name:
                                    fake = np.interp(fake, (min(fake), max(fake)), (-0.4, 1.5))
                                distance += fastdtw(real, fake, dist=euclidean, radius = 216)[0]
                                rmse += np.sqrt((real-fake) ** 2).mean()
                                w_dist += wasserstein_distance(real, fake)
                                #print("wdist ", w_dist)
                                #kl_div += np.sum(np.where(fake != 0, fake * np.log(fake / real), 0))
                                pearson += pearsonr(real, fake)[0]
                                #print("pearson ", pearson)
                                #frechet += similaritymeasures.frechet_dist(real, fake)
                        if i == 0:
                            print ("real array ", real)
                            print ("fake array ", fake)

                    DTW.append(distance/(N**2))
                    RMSE.append(rmse/(N**2))
                    PEARSON.append(pearson/(N**2))
                    WDIST.append(w_dist/(N**2))
                    #KL_DIV.append(kl_div/(N**2))
                    print ("dtw ", distance/(N**2))
                    print ("rmse ", rmse/(N**2))
            #mmd_list.append(mmd_eval.item())
            print("end of 1 epoch")
            #self.g_schedule.step()
            #self.d_schedule.step()

        #logger.save_models(generator=self.generator)
        #logger.save_discriminator(discriminator=self.discriminator)
        logger.save_errors(g_loss=G_losses, d_loss=D_losses)
        logger.save_dtw_rmse(DTW, RMSE, epoch, num_batches)
        logger.save_pearson_wdist(PEARSON, WDIST, epoch, num_batches)
        #logger.save_kldiv(KL_DIV, epoch, num_batches)
        timeTaken = time.time() - start_time
        test_images = self.generator(test_noise)

        if self.type["dataset"] == "cifar":
            test_images = vectors_to_images_cifar(test_images).cpu().data
            calculate_metrics_cifar(path=f'{logger.data_subdir}/generator.pt', numberOfSamples=10000)
        elif self.type["dataset"] == "ecg":
            if "CNN" in self.name:
                for i in range(len(test_images)):
                    with torch.no_grad():
                        fake = test_images[i, :].numpy()
                        fake = np.interp(fake, (min(fake), max(fake)), (-0.4, 1.5))
                        test_images[i,:] = torch.tensor(fake)
            else:
                test_images = test_images
        else:
            test_images = vectors_to_images(test_images).cpu().data
            calculate_metrics(path=f'{logger.data_subdir}/generator.pt', numberOfSamples=10000,
                              datasetType=self.type["dataset"])

        logger.log_images(test_images, self.epochs + 1, 0, num_batches)
        logger.save_scores(timeTaken, 0)
        return

    def _train_generator(self, fake_data: torch.Tensor, local_explainable, trained_data=None) -> torch.Tensor:
        """
        This function performs one iteration of training the generator
        :param fake_data: tensor data created by generator
        :return: error of generator on this training step
        """
        N = fake_data.size(0)

        # Reset gradients
        self.g_optim.zero_grad()

        # Sample noise and generate fake data
        if "IMV" in self.name:
            prediction = self.discriminator(fake_data)[0].view(-1)
        else:
            prediction = self.discriminator(fake_data).view(-1)

        if local_explainable and "LSTMIMV" not in self.name:
            print("local explanation true")
            get_explanation(generated_data=fake_data, discriminator=self.discriminator, prediction=prediction,
                            XAItype=self.explanationType, cuda=self.cuda, trained_data=trained_data,
                            data_type=self.type["dataset"], alpha=self.alpha)

        # Calculate error and back-propagate
        #error = self.loss(prediction, values_target(size=(N,), value=self.real_label, cuda=self.cuda))
        print("prediction shape ", prediction.shape)
        error = -torch.mean(prediction)
        error.backward()

        # clip gradients to avoid exploding gradient problem
        nn.utils.clip_grad_norm_(self.generator.parameters(), 10)

        # update parameters
        self.g_optim.step()

        # Return error
        return error

    def _train_discriminator(self, real_data: Variable, fake_data: torch.Tensor, local_explainable):
        """
        This function performs one iteration of training the discriminator
        :param real_data: batch from dataset
        :type real_data: torch.Tensor
        :param fake_data: data from generator
        :type fake_data: torch.Tensor
        :return: tuple of (mean error, predictions on real data, predictions on generated data)
        :rtype: (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        N = real_data.size(0)

        # Reset gradients
        self.d_optim.zero_grad()

        # 1.1 Train on Real Data
        #print ("at discriminator training")
        if "IMV" in self.name:
            #print ("at discrminator IMV")
            prediction_real, alphas, betas = self.discriminator(real_data)
            alphas = alphas.view(alphas.shape[0], alphas.shape[1])
            prediction_real = prediction_real.view(-1)
            #print ("preidction real ", prediction_real.shape)
            #print ("alphas size ", alphas.shape)
            if local_explainable and self.explanationType=="IMV":
                get_explanation_IMV(alphas)
                #print("got alphas ", alphas.shape)
        else:
            prediction_real = self.discriminator(real_data).view(-1)
            ##print (prediction_real)

        # Calculate error
        #error_real = self.loss(prediction_real, values_target(size=(N,), value=self.real_label, cuda=self.cuda))

        # 1.2 Train on Fake Data
        if "IMV" in self.name:
            prediction_fake, a, b = self.discriminator(fake_data)
        else:
            prediction_fake = self.discriminator(fake_data)
        prediction_fake = prediction_fake.view(-1)

        # Calculate error
        #error_fake = self.loss(prediction_fake, values_target(size=(N,), value=self.fake_label, cuda=self.cuda))


        #Calculate gradient penalty
        epsilon = torch.rand(N, 1, requires_grad = True).repeat(1, real_data.shape[1])
        interpolated_data = real_data * epsilon + fake_data * (1-epsilon)
        mixed_scores = self.discriminator(interpolated_data)
        gradient = torch.autograd.grad(
            inputs = interpolated_data,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)

        gradient_norm = gradient.norm(2, dim=1)
        gp = torch.mean((gradient_norm -1)**2)

        # Sum up error and backpropagate
        #error = error_real + error_fake
        error = (
            -(torch.mean(prediction_real) - torch.mean(prediction_fake)) + self.lambda_gp*gp
        )
        error.requires_grad_()

        # print ("error shape ", error.shape)
        # print ("error ", error)
        # print("real data shape ", real_data.shape)
        # print("fake data shape ", fake_data.shape)
        # print ("mean real data ", torch.mean(real_data))
        # print ("mean real data ", torch.mean(fake_data))

        error.backward()
        #sprint ("D BACKWARD DONE ", error.shape)

        # 1.3 Update weights with gradients
        self.d_optim.step()

        # for p in self.discriminator.parameters():
        #     p.data.clamp_(-self.weight_clip, self.weight_clip)
        #Weight clipping for Wasserstein GAN

        # Return error and predictions for real and fake inputs
        return error, prediction_real, prediction_fake
