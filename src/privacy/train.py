import pickle
#from tensorboardX import Summary#writer
import os
import logging
import shutil
from LSTM_classifier import ECGLSTM
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append("..")
import patient, ecg_mit_bih, dataset_configs, ecg_dataset_pytorch
from torchvision import transforms, datasets
from models.generators import EcgCNNGenerator, ECGLSTMGenerator

#best_auc_scores = train_classifier(net, model_dir=model_dir, train_config=train_config)

def main():

    net = ECGLSTM(5, 512, 2, 2)
    model_dir = "/Users/alainamahalanabis/Documents/GAN_XAI/src/results/ecg/ECGLSTMIMVNormal/generator.pt"
    train_configs = dataset_configs.DatasetConfigs('train', one_vs_all=True, lstm_setting=False,
                                        over_sample_minority_class=False,
                                        under_sample_majority_class=False, only_take_heartbeat_of_type='N',
                                        classified_heartbeat='N')
    best_auc_scores = train_classifier(net=net, model_dir=model_dir, train_config=train_configs)

def train_classifier(net, train_config, model_dir, n=10000, ch_path=None, batch_size=128, beat_type='N', gan_type=None):
    """s
    :param network:
    :return:
    """
    best_auc_scores = [0, 0, 0, 0]

    gNet = ECGLSTMGenerator()
    gNet.load_state_dict(torch.load(model_dir))
    gNet.eval()

    with torch.no_grad():
        #change n to be number of heartbeats needed
        input_noise = torch.Tensor(np.random.normal(0, 1, (n, 100)))
        output_g = gNet(input_noise)
        #print ("output g ", output_g)
        output_g = output_g.numpy()
        print("length of output g", len(output_g))
        output_g = np.array(
            [{'cardiac_cycle': torch.from_numpy(x), 'beat_type': beat_type,
              'label': torch.tensor([1,0])} for x in output_g])
        print ("output g shape ", output_g[0])

    composed = transforms.Compose([ecg_dataset_pytorch.ToTensor()])

    dataset = output_g


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=1, shuffle=True)

    print("DONE GETTING GENERATOR DATALOADER", len(dataloader))

    testset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(transform=composed, configs=train_config)
    testdataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    print("DONE GETTING REAL DATALOADER", len(testdataloader))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    #writer = Summary#writer("/Users/alainamahalanabis/Documents/GAN_XAI/src/privacy/")
    total_iters = 0

    for epoch in range(4):  # loop over the dataset multiple times

        for i, data in enumerate(dataloader):
            total_iters += 1
            print("1 ENUMERATE DONE")

            # get the inputs
            # ecg_batch = data['cardiac_cycle'].view(-1, 1, 216).float()
            ecg_batch = data['cardiac_cycle'].float()

            b_size = ecg_batch.shape[0]
            print ("b_size ", b_size)
            labels = data['label']
            print(labels)
            labels_class = torch.max(labels, 1)[1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(ecg_batch)
            print("OUTPUTS ", outputs)
            exit(0)
            outputs_class = torch.max(outputs, 1)[1]
            print(outputs_class)

            accuracy = (outputs_class == labels_class).sum().float() / b_size
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            #writer.add_scalars('Cross_entropy_loss', {'Train batches loss': loss.item()}, total_iters)
            #writer.add_scalars('Accuracy', {'Train batches accuracy': accuracy.item()}, total_iters)
            # print statistics
            print("Epoch {}. Iteration {}.\t Batch loss = {:.2f}. Accuracy = {:.2f}".format(epoch + 1, i, loss.item(),
                                                                                            accuracy.item()))
            if total_iters % 50 == 0:
                fig, _ = plot_confusion_matrix(labels_class.numpy(), outputs_class.numpy(),
                                               np.array(['N', 'S', 'V', 'F', 'Q']))
                #writer.add_figure('train/confusion_matrix', fig, total_iters)

            if total_iters % 200 == 0:
                with torch.no_grad():
                    labels_total_one_hot = np.array([]).reshape((0, 5))
                    outputs_preds = np.array([]).reshape((0, 5))
                    labels_total = np.array([])
                    outputs_total = np.array([])
                    loss_hist = []
                    for _, test_data in enumerate(testdataloader):
                        # ecg_batch = test_data['cardiac_cycle'].view(-1, 1, 216).float()
                        ecg_batch = test_data['cardiac_cycle'].float()
                        labels = test_data['label']
                        labels_class = torch.max(labels, 1)[1]
                        outputs = net(ecg_batch)
                        loss = criterion(outputs, torch.max(labels, 1)[1])
                        loss_hist.append(loss.item())
                        outputs_class = torch.max(outputs, 1)[1]

                        labels_total_one_hot = np.concatenate((labels_total_one_hot, labels.numpy()))
                        labels_total = np.concatenate((labels_total, labels_class.numpy()))
                        outputs_total = np.concatenate((outputs_total, outputs_class.numpy()))
                        outputs_preds = np.concatenate((outputs_preds, outputs.numpy()))

                    outputs_total = outputs_total.astype(int)
                    labels_total = labels_total.astype(int)
                    fig, _ = plot_confusion_matrix(labels_total, outputs_total,
                                                   np.array(['N', 'S', 'V', 'F', 'Q']))
                    # Accuracy and Loss:
                    accuracy = sum((outputs_total == labels_total)) / len(outputs_total)
                    #writer.add_scalars('Accuracy', {'Test set accuracy': accuracy}, global_step=total_iters)
                    #writer.add_figure('test/confusion_matrix', fig, total_iters)
                    loss = sum(loss_hist) / len(loss_hist)
                    #writer.add_scalars('Cross_entropy_loss', {'Test set loss': loss}, total_iters)
                    #auc_roc = plt_roc_curve(labels_total_one_hot, outputs_preds, np.array(['N', 'S', 'V', 'F', 'Q']),
                                            #writer, total_iters)

                    for i_auc in range(4):
                        if auc_roc[i_auc] > best_auc_scores[i_auc]:
                            best_auc_scores[i_auc] = auc_roc[i_auc]

    #writer.close()
    return best_auc_scores

if __name__ == "__main__":
    main()
