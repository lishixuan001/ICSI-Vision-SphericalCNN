import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import gzip
import pickle
import os
import argparse
from helper_methods import *
from logger import *
from torch.utils.data import DataLoader

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class S2ConvNet(nn.Module):

    def __init__(self, para_dict):
        super(S2ConvNet, self).__init__()

        self.para_dict = para_dict
        self.batch_size = self.para_dict['batchsize']
        self.num_points = self.para_dict['num_points']

        self.f1 = self.para_dict['f1']
        self.f2 = self.para_dict['f2']
        self.f_output = self.para_dict['f_output']

        self.b_in = self.para_dict['b_in']
        self.b_l1 = self.para_dict['b_l1']
        self.b_l2 = self.para_dict['b_l2']

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=self.f1,
            b_in=self.b_in,
            b_out=self.b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=self.f1,
            nfeature_out=self.f2,
            b_in=self.b_l1,
            b_out=self.b_l2,
            grid=grid_so3)

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(num_features=10)
        self.out_layer = nn.Linear(10 * (self.num_points * self.f2 - 8 + 1), self.f_output)

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = F.relu(conv2) ###
        in_data = relu2[:, :, 0, 0, 0]
        in_reshape = in_data.reshape(self.batch_size, 1, self.num_points * self.f2)  # B * C * L
        conv3 = self.conv3(in_reshape)  # (B, 1, L) -> (B, 10, L'), L' = num_points * f2 - kernel_size + 1
        relu3 = F.relu(conv3)
        bn3 = self.bn3(relu3)
        bn3_reshape = bn3.reshape((self.batch_size, -1))  # (B, 10 * L')
        output = self.out_layer(bn3_reshape)

        # x = so3_integrate(x)
        # x = self.out_layer(x)
        # return x
        return output


def main():
    logger = setup_logger('data_generate')
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=10,  ###
                        required=True)
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=2, ###
                        required=True)
    parser.add_argument("--validsize",
                        help="percentage that training set split into validation set",
                        type=float,
                        default=1/6,
                        required=False)
    parser.add_argument("--signal-type",
                        help="Gaussian or Potential",
                        type=str,
                        default="Gaussian",
                        required=False)
    parser.add_argument("--input-prefix",
                        help="file for saving the data (.gz file)",
                        type=str,
                        default="mnistPC",
                        required=True)
    parser.add_argument("--num-epochs",
                        help="number of epochs",
                        type=int,
                        default=20,
                        required=False)
    parser.add_argument("--learning-rate",
                        help="learning rate of the model",
                        type=float,
                        default=5e-3,
                        required=False)
    args = parser.parse_args()
    logger.info("call with args: \n{}".format(args))
    logger.info("getting MNIST data")

    """load dataset and generate dataloader"""
    with gzip.open(os.path.join(args.input_prefix, 'train_mnist.gz'), 'rb') as f:
        train_dataset = pickle.load(f)
#    with gzip.open(os.path.join(args.input_prefix, 'test_mnist.gz'), 'rb') as f:
#        test_dataset = pickle.load(f)

    min_data = train_dataset.points.min()
    max_data = train_dataset.points.max()

    train_dataset.points = (train_dataset.points - min_data) / (max_data - min_data)
#    test_dataset.points = (test_dataset.points - min_data) / (max_data - min_data)

    train_loader, valid_loader = split_train_and_valid(trainset=train_dataset,
                                                       batch_size=args.batchsize,
                                                       valid_size=args.validsize)

#    test_loader = DataLoader(dataset=test_dataset,
#                             batch_size=args.batchsize,
#                             shuffle=False)

    parameter_dict = {
        'batchsize': args.batchsize,
        'num_points': 512,
        'f1': 2,
        'f2': 4,
        'f_output': 10,  # should be the number of classes
        'b_in': args.bandwidth,
        'b_l1': 8,
        'b_l2': 4
    }
    classifier = S2ConvNet(parameter_dict)
    classifier.cuda()

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        i = 0
        for tl in train_loader:
            images = tl['data'].reshape((-1, 1, 2 * args.bandwidth, 2 * args.bandwidth)) # (b * 512, 1, 2b, 2b)
            labels = tl['label']  # shape [1]

            images = images.cuda().float()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            logger.info("Epoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                epoch + 1, args.num_epochs, i + 1, len(train_dataset) * (1 - args.validsize) // args.batchsize,
                loss.item()
            ))
            # print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            #     epoch + 1, args.num_epochs, i + 1, len(train_dataset) // args.batchsize,
            #     loss.item()), end="")
            i = i + 1
        correct = 0
        total = 0
        for vl in valid_loader:
            images = vl['data'].reshape((-1, 1, 2 * args.bandwidth, 2 * args.bandwidth))
            labels = vl['label']
            classifier.eval()

            with torch.no_grad():
                images = images.cuda().float()
                labels = labels.cuda()

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()
        logger.info("TEST ACC: {0}".format(100 * correct / total))
        # print('\nTest Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main()
