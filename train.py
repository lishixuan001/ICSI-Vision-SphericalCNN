import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class S2ConvNet(nn.Module):

    def __init__(self):
        super(S2ConvNet, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = so3_integrate(x)

        x = self.out_layer(x)

        return x


def main():
    logger = setup_logger('data_generate')
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=30,
                        required=False)
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--validsize",
                        help="percentage that training set split into validation set",
                        type=float,
                        default=0.5,
                        required=False)
    parser.add_argument("--demo",
                        help="if demo is true, then only load 10 image",
                        type=bool,
                        default=True,
                        required=False)
    parser.add_argument("--signal-type",
                        help="Gaussian or Potential",
                        type=str,
                        default="Gaussian",
                        required=False)
    parser.add_argument("--input-prefix",
                        help="file for saving the data (.gz file)",
                        type=str,
                        default="../mnistPC",
                        required=False)
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
    with gzip.open(os.path.join(args.input_prefix, 'train_mnist.gz'), 'rb') as f:
        test_dataset = pickle.load(f)

    min_data = train_dataset.points.min()
    max_data = train_dataset.points.max()

    train_dataset.points = (train_dataset.points - min_data) / (max_data - min_data)
    test_dataset.points = (test_dataset.points - min_data) / (max_data - min_data)

    train_loader, valid_loader = split_train_and_valid(trainset=train_dataset,
                                                       batch_size=args.batchsize,
                                                       valid_size=args.validsize)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batchsize,
                             shuffle=True)

    classifier = S2ConvNet()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        i = 0
        for tl in train_loader:
            images = tl['data']
            labels = tl['label']

            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch + 1, args.num_epochs, i + 1, len(train_dataset) // args.batchsize,
                loss.item()), end="")
            i = i + 1
        correct = 0
        total = 0
        for images, labels in valid_loader:
            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        print('Test Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main()
