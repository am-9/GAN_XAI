"""
This file contains the explainable AI utils needed for xAI-GAN to work

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
from captum.attr import DeepLiftShap, Saliency, IntegratedGradients, DeepLift
from utils.vector_utils import values_target, images_to_vectors
import pdb
from utils.vector_utils import values_target
#from lime import lime_image

# defining global variables
global values
global discriminatorLime


def get_explanation(generated_data, discriminator, prediction, XAItype, cuda=True, trained_data=None,
                    data_type="mnist") -> None:
    """
    This function calculates the explanation for given generated images using the desired xAI systems and the
    :param generated_data: data created by the generator
    :type generated_data: torch.Tensor
    :param discriminator: the discriminator model
    :type discriminator: torch.nn.Module
    :param prediction: tensor of predictions by the discriminator on the generated data
    :type prediction: torch.Tensor
    :param XAItype: the type of xAI system to use. One of ("shap", "lime", "saliency")
    :type XAItype: str
    :param cuda: whether to use gpu
    :type cuda: bool
    :param trained_data: a batch from the dataset
    :type trained_data: torch.Tensor
    :param data_type: the type of the dataset used. One of ("cifar", "mnist", "fmnist")
    :type data_type: str
    :return:
    :rtype:
    """
    # initialize temp values to all 1s
    temp = values_target(size=generated_data.size(), value=1.0, cuda=cuda)

    # mask values with low prediction
    mask = (prediction < 0.5).view(-1)
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()
    #print ("LENGTH INDICES ", indices)

    data = generated_data[mask, :]
    # print ("trained data size ", trained_data["cardiac_cycle"].size())
    # print ("data size ", data.unsqueeze(0).size())
    # print ("indices size ", len(indices))
    #exit(0)

    if len(indices) > 1:

        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0))

        elif XAItype == "ig":
            for i in range(len(indices)):
                explainer = IntegratedGradients(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0),  target=0)

        elif XAItype == "deeplift":
            print ("shap xAI")
            for i in range(len(indices)):
                explainer = DeepLift(discriminator)
                #pdb.set_trace()
                #print ("one training data ", trained_data['cardiac_cycle'][1, :])
                print ("one training data size ", trained_data['cardiac_cycle'][1, :].size())
                print ("data ", data[1, :].detach())
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0), baselines=torch.zeros(size=(1, 216)), target=0)
                print ("first temp indices done")
                exit(0)


        elif XAItype == "shap":
            print ("shap xAI")
            for i in range(len(indices)):
                explainer = DeepLiftShap(discriminator.double())
                #pdb.set_trace()
                #print ("one training data ", trained_data['cardiac_cycle'][1, :])
                print ("one training data size ", trained_data['cardiac_cycle'][1, :].size())
                print ("data ", data[1, :].detach())
                #temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0), trained_data['cardiac_cycle'], target=0)
                attributions, delta = explainer.attribute(data[i, :].unsqueeze(0).detach(), baselines=torch.zeros(size=(216)), target=0)
                #print('DeepLiftSHAP Attributions:', attributions)
                print ("first temp indices done")
                exit(0)


        elif XAItype == "lime":
            explainer = lime_image.LimeImageExplainer()
            global discriminatorLime
            discriminatorLime = deepcopy(discriminator)
            discriminatorLime.cpu()
            discriminatorLime.eval()
            for i in range(len(indices)):
                if data_type == "cifar":
                    tmp = data[i, :].detach().cpu().numpy()
                    tmp = np.reshape(tmp, (32, 32, 3)).astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict_cifar, num_samples=100)
                else:
                    tmp = data[i, :].squeeze().detach().cpu().numpy().astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict, num_samples=100)
                _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
                temp[indices[i], :] = torch.tensor(mask.astype(np.float))
            del discriminatorLime
        else:
            raise Exception("wrong xAI type given")

    if cuda:
        temp = temp.cuda()
    #print ("DONE SETTING VALUES")
    set_values(normalize_vector(temp))

def explanation_hook(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """

    print ("AT EXPLANATION HOOK")
    # get stored mask
    temp = get_values()
    temp = temp.unsqueeze(1)
    temp = temp.unsqueeze(1)

    print ("TEMP SHAPE")
    print (temp.shape)

    #print ("grad input shape ", grad_input[0].shape)
    #print ("grad output shape ", grad_output[0].shape)

    # multiply with mask to generate values in range [1x, 1.2x] where x is the original calculated gradient
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * temp)

    print ("DONE COMPUTING NEW GRAD")
    print ("new grad input shape ", new_grad.shape)

def explanation_hook_lstm(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """

    #print ("AT EXPLANATION HOOK")
    # get stored mask
    temp = get_values()
    #temp = temp.unsqueeze(1)
    #temp = temp.unsqueeze(1)

    print ("TEMP SHAPE")
    print (temp.shape)

    temp2 = torch.mean(temp, dim=0)
    # print ("TEMP2 SHAPE")
    # print (temp2.shape)

    #grad_input[0] = grad_input[0].unsqueeze(0)

    # grad input 0 shape torch.Size([216])
    # grad input 1 shape torch.Size([128, 100])
    # grad input 2 shape torch.Size([100, 216])

    # print ("length of grad input ", len(grad_input))
    # print ("grad input 0 shape ", grad_input[0].shape)
    # print ("grad input 1 shape ", grad_input[1].shape)
    # print ("grad input 2 shape ", grad_input[2].shape)
    #
    # print ("length of grad output ", len(grad_output))
    # print ("grad ouput 0 shape ", grad_output[0].shape)


    # multiply with mask to generate values in range [1x, 1.2x] where x is the original calculated gradient
    new_grad_0 = grad_input[0] + 0.2 * (grad_input[0] * temp2)
    tmp = torch.mm(grad_input[1], grad_input[2])
    print ("size of tmp ", tmp.shape)
    new_grad_1 = grad_input[1]
    new_grad_2 = grad_input[2] + 0.2 * (grad_input[2] * temp2)

    new_grad = (new_grad_0, new_grad_1, new_grad_2)

    print ("DONE COMPUTING NEW GRAD")
    print ("new_grad[0].shape ", new_grad[0].shape)
    print ("new_grad[2].shape ", new_grad[2].shape)

    return new_grad


def explanation_hook_cifar(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    temp = get_values()

    # multiply with mask
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * temp)

    return (new_grad, )


def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values() -> np.array:
    """ get global values """
    global values
    return values

def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x


def batch_predict(images):
    """ function to use in lime xAI system for MNIST and FashionMNIST"""
    # convert images to greyscale
    images = np.mean(images, axis=3)
    # stack up all images
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    logits = discriminatorLime(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().numpy()


def batch_predict_cifar(images):
    """ function to use in lime xAI system for CIFAR10"""
    # stack up all images
    images = np.transpose(images, (0, 3, 1, 2))
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    logits = discriminatorLime(batch)
    probs = F.softmax(logits, dim=1).view(-1).unsqueeze(1)
    return probs.detach().numpy()
