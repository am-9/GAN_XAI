# XAI-GAN

## Table of Contents
* Introduction
* Usage
* Report

## Introduction
Generative Adversarial Networks (GANs) are a revolutionary class of Deep Neural Networks (DNNs) that have been successfully used to generate realistic images, music, text, and other data. 
However, GAN training presents many challenges, notably it can be very resource-intensive. A potential weakness in GANs is that it requires a lot of data for successful training and data collection can be an expensive process. Typically, discriminator DNNs provide only one value (loss) of corrective feedback to generator DNNs (namely, the discriminator's assessment of the generated example). By contrast, we propose a new class of GAN we refer to as xAI-GAN that leverages recent advances in explainable AI (xAI) systems to provide a "richer" form of corrective feedback from discriminators to generators. 

### Source Code
The code of the project is found in the `/src/` directory and run using the main.py file. 
