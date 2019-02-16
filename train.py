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

# The device assigne to torch.Tensor
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class S2ConvNet(nn.Module):

    def __init__(self, para_dict):
        """
        :param para_dict: A dictionary containing all pairs of parameters
        """
        super(S2ConvNet, self).__init__()

        self.para_dict = para_dict
        self.batch_size = self.para_dict['batchsize']
        self.num_points = self.para_dict['num_points']

        self.l1_num_output_features = self.para_dict['l1_num_output_features']
        self.l2_num_output_features = self.para_dict['l2_num_output_features']
        self.f_output = self.para_dict['f_output']

        self.l1_input_bandwidth = self.para_dict['l1_input_bandwidth']
        self.l1_output_bandwidth = self.para_dict['l1_output_bandwidth']
        self.l2_output_bandwidth = self.para_dict['l2_output_bandwidth']

        grid_s2 = s2_near_identity_grid() # len() -> 24=3*8 -> ((alpha0, beta0), (alpha1, beta1), ...)
        grid_so3 = so3_near_identity_grid() # len() -> 72=3*8*3 -> ((alpha0, beta0, gamma0), ...)

        self.conv1 = S2Convolution(
            nfeature_in=1, # number of input fearures (default=1)
            nfeature_out=self.l1_num_output_features, # number of output features  (default=2)
            b_in=self.l1_input_bandwidth, # input bandwidth (precision of the input SOFT grid) (default=10)
            b_out=self.l1_output_bandwidth, # output bandwidth (default=8)
            grid=grid_s2 # points of the sphere defining the kernel, tuple of (alpha, beta)'s
        )

        self.conv2 = SO3Convolution(
            nfeature_in=self.l1_num_output_features, # (default=2)
            nfeature_out=self.l2_num_output_features, # (default=4)
            b_in=self.l1_output_bandwidth, # (default=8)
            b_out=self.l2_output_bandwidth, # (default=4)
            grid=grid_so3
        )

        self.conv3 = nn.Conv1d(
            in_channels=1,
            out_channels=10,
            kernel_size=8
        )
        self.bn3 = nn.BatchNorm1d(num_features=10)
        self.out_layer = nn.Linear(10 * (self.num_points * self.l2_num_output_features - 8 + 1), self.f_output)

    def forward(self, x):
        """
        :param x: [batch_size, num_points, 1, 2b, 2b]
        :return:
        """
        conv1 = self.conv1(x) # [batch_size * num_points, feature_out_1, 2b1, 2b1]
        relu1 = F.relu(conv1) # [batch_size * num_points, feature_out_1, 2b1, 2b1]
        conv2 = self.conv2(relu1) # [batch_size * num_points, feature_out_2, 2b2, 2b2]
        relu2 = F.relu(conv2) # [batch_size * num_points, feature_out_2, 2b2, 2b2]

        in_data = relu2[:, :, 0, 0, 0] # [batch_size * num_points, feature_out_2]
        in_reshape = in_data.reshape(self.batch_size, 1, self.num_points * self.l2_num_output_features)  # [batch_size, 1, num_points * feature_out_2]
        conv3 = self.conv3(in_reshape)  # [batch_size, out_channels_3, feature_out_3] ==> feature_out_3 = num_points * f2 - kernel_size + 1
        relu3 = F.relu(conv3) # [batch_size, out_channels_3, feature_out_3] ==> feature_out_3 = num_points * f2 - kernel_size + 1

        bn3 = self.bn3(relu3) # [batch_size, out_channels_3, feature_out_3]
        bn3_reshape = bn3.reshape((self.batch_size, -1)) # [batch_size, out_channels_3 * feature_out_3]
        output = self.out_layer(bn3_reshape) # [batch_size, features_classes]

        return output


def main():
    logger = setup_logger('data_generate')
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=10,  ###
                        required=False)
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=2, ###
                        required=False)
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
    with gzip.open(os.path.join(args.input_prefix, 'test_mnist.gz'), 'rb') as f:
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
                             shuffle=False)

    parameter_dict = {
        'batchsize': args.batchsize,
        'num_points': 512,
        'l1_num_output_features': 2,
        'l2_num_output_features': 4,
        'f_output': 10,  # should be the number of classes
        'l1_input_bandwidth': args.bandwidth,
        'l1_output_bandwidth': 8,
        'l2_output_bandwidth': 4
    }
    classifier = S2ConvNet(parameter_dict)
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
            print("tl[point]: {}".format(tl["point"].shape))
            images = tl['point'].reshape((-1, 1, 2 * args.bandwidth, 2 * args.bandwidth)) # (b * 512, 1, 2b, 2b)
            print("images: {}".format(images.shape))
            labels = tl['label']  # shape [1]

            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            logger.info("Epoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                epoch + 1, args.num_epochs, i + 1, len(train_dataset) // args.batchsize,
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
                images = images.to(DEVICE).float()
                labels = labels.to(DEVICE)
                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()
        logger.info("TEST ACC: {0}".format(100 * correct / total))
        # print('\nTest Accuracy: {0}'.format(100 * correct / total))


if __name__ == '__main__':
    main()
