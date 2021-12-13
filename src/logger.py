"""
This file logs experiment runs. Using code by Diego Gomez from: https://github.com/diegoalejogm/gans/ with slight
modifications

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import torch
import errno
import os
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
import pandas
import time


class Logger:
    """
    Logs information during the experiment
    """
    def __init__(self, experiment_name, datasetName):
        """
        Standard init
        :param experiment_name: name of the experiment enum
        :type experiment_name: str
        :param datasetName: name of the dataset
        :type datasetName: str
        """
        self.model_name = experiment_name
        self.data_subdir = f'./results/{datasetName}/{experiment_name}'
        Logger._make_dir(self.data_subdir)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.data_subdir, write_to_disk=False)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.data_subdir), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.data_subdir), g_error, step)

    def save_errors(self, g_loss, d_loss):
        np.save(self.data_subdir + "/g_loss.npy", np.array(g_loss))
        np.save(self.data_subdir + "/d_loss.npy", np.array(d_loss))

        plt.plot(g_loss, color="blue", label="generator")
        plt.plot(d_loss, color="orange", label="discriminator")
        plt.legend()
        plt.savefig(self.data_subdir + "/plotLoss.png")

    def log_images(self, images, epoch, n_batch, num_batches, i_format='NCHW', normalize=True):
        """ input images are expected in format (NCHW) """


        images = images.detach().numpy()
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        print (type(images))

        print (images.shape)
        #test plots
        self.save_ecg(images, epoch, n_batch)

        print (images.shape)

        if i_format == 'NHWC':
            images = images.transpose(1, 3)

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.data_subdir, '')

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True )

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, epoch, n_batch):
        # Plot and save horizontal
        fig = plt.figure(figsize=(32,32))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        fig.savefig('{}/epoch_{}_batch_{}.png'.format(self.data_subdir, epoch, n_batch))
        plt.close()

    def save_ecg(self, images, epoch, n_batch):

        print ("SAVING ECG IMAGE")
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12),
              (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4)

        ax1.plot(images[0])
        ax2.plot(images[1])
        ax3.plot(images[2])
        ax4.plot(images[3])
        ax5.plot(images[4])
        ax6.plot(images[5])
        ax7.plot(images[6])
        ax8.plot(images[7])
        ax9.plot(images[8])
        ax10.plot(images[9])
        ax11.plot(images[10])
        ax12.plot(images[11])
        ax13.plot(images[12])
        ax14.plot(images[13])
        ax15.plot(images[14])
        ax16.plot(images[15])

        for ax in fig.get_axes():
            ax.label_outer()
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth('1')

        plt.axis('off')
        fig.savefig('{}/epoch_{}_batch_{}_{}.png'.format(self.data_subdir, epoch, n_batch, time.time()))
        #plt.show()
        plt.close()

        np.savetxt('{}/epoch_{}_batch_{}_heartbeat_time{}.csv'.format(self.data_subdir, epoch, n_batch, time.time()), images[0], delimiter=",")

    def save_dtw_fd(self, dtw, fd, epoch, n_batch):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        print ("DTW ", dtw)
        print ("FD ", fd)

        x_range = range(1, len(dtw)+1)
        ax1.scatter(x_range, dtw)
        ax2.scatter(x_range, fd)
        ax1.set_title("epoch vs dtw")
        ax2.set_title("epoch vs fd")
        #ax1.xticks(fontsize=30)
        #ax1.yticks(fontsize=30)
        fig.savefig('{}/epoch_{}_batch_{}_dtw_fd_time{}.png'.format(self.data_subdir, epoch, n_batch, time.time()))
        plt.close()
        np.savetxt('{}/epoch_{}_batch_{}_dtw_time{}.csv'.format(self.data_subdir, epoch, n_batch, time.time()), dtw, delimiter=",")
        np.savetxt('{}/epoch_{}_batch_{}_fd_time{}.csv'.format(self.data_subdir, epoch, n_batch, time.time()), fd, delimiter=",")

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator):
        torch.save(generator.state_dict(), f'{self.data_subdir}/generator.pt')

    def close(self):
        self.writer.close()

    # Private Functions
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_scores(self, time, fid):
        with open(f'{self.data_subdir}/results.txt', 'w') as file:
            file.write(f'time taken: {round(time, 4)}\n')
            file.write(f'fid score: {round(fid, 4)}')
