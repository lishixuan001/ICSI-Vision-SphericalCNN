import torch.nn as nn
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
import torch.nn.functional as F
import h5py
import argparse
from utils import *
from logger import *

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
    """path to store input/ output file"""
    parser.add_argument("--train-file-path",
                        help="the path of train file",
                        type=str,
                        default='../mnistPC/train.hdf5',
                        required=False)
    parser.add_argument("--test-file-path",
                        help="the path of test file",
                        type=str,
                        default='../mnistPC/test.hdf5',
                        required=False)
    parser.add_argument("--output-prefix",
                        help="file for saving data output (.gz file)",
                        type=str,
                        default="../mnistPC",  # mind the correctness when loading file
                        required=False)
    parser.add_argument("--demo",
                        help="if demo is true, then only load 10 image",
                        type=bool,
                        default=False,
                        required=False)
    """args about translation"""
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=10,  ###
                        required=True)
    parser.add_argument("--signal-type",
                        help="Gaussian or Potential",
                        type=str,
                        default="Gaussian",
                        required=False)
    """args for training"""
    parser.add_argument("--validsize",
                        help="percentage that training set split into validation set",
                        type=float,
                        default=1/6,
                        required=False)
    parser.add_argument("--batchsize",
                        help="the batch size of the dataloader",
                        type=int,
                        default=2, ###
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

    ####################################################################################################################
    """load MNIST point set"""
    logger.info("getting MNIST data and basic configuration")
    f_train = h5py.File(args.train_file_path)  # the point is ranged from -1 to 1
    f_test = h5py.File(args.test_file_path)
    if args.demo:
        train_size = 12
        test_size = 1
    else:
        train_size = f_train['data'].shape[0]
        test_size = f_test['data'].shape[0]
    num_points = f_train['data'].shape[1]

    if args.signal_type == "Gaussian":
        utility_type = UtilityTypes.Gaussian
    elif args.signal_type == "Potential":
        utility_type = UtilityTypes.Potential
    else:
        raise UtilityError("invalid utility type, should be chosen from 'Gaussian' or 'Potential'.")
    logger.info("finish loading MNIST data and basic configuration")

    ####################################################################################################################
    """load train and test set, and calculate radius"""
    logger.info("start loading train and test set")
    # add [()] can read h5py file as numpy.ndarray
    train_dataset = f_train['data'][()][0:train_size]  # [train_size, 512, 2]
    train_dataset = torch.from_numpy(train_dataset)  # convert from numpy.ndarray to torch.Tensor

    tensor_label_train = f_train['labels'][()][0:train_size]

    # test_np_dataset = f_test['data'][()][0:test_size]
    # test_torch_dataset = torch.from_numpy(test_np_dataset)  # convert from numpy.ndarray to torch.Tensor

    radius = get_radius(train_dataset)

    """add z dimension if 2D point set"""
    if train_dataset.shape[-1] == 2:
        """if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (train_size, num_points, 3) -> z-dimension additionally padded by 0 -> (x, y, 0)
        zero_padding = torch.zeros((train_size, num_points, 1), dtype=train_dataset.dtype)
        train_dataset = torch.cat((train_dataset, zero_padding), -1) # -> [train_size, 512, 3]
    logger.info("finish loading train dataset")

    ####################################################################################################################
    """data loader and net loader"""
    logger.info("start initialize the dataloader, and network")
    # train_loader [batch, 512, 3], same as valid_loader
    train_loader, valid_loader = split_train_and_valid(trainset=train_dataset,
                                                       labelset=tensor_label_train,
                                                       batch_size=args.batchsize,
                                                       valid_size=args.validsize)
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
    logger.info("finish loading the network")

    ####################################################################################################################
    """start iteration"""
    logger.info("start training")
    for epoch in range(args.num_epochs):
        i = 0
        for tl in train_loader:
            images = tl['point'] # (b, 512, 3)
            labels = tl['label']  # shape [1]
            """hold translation from [B, 512, 3] -> [B * 512, 1, 2b, 2b]"""
            images = translation(
                images=images,
                bandwidth=args.bandwidth,
                radius=radius,
                utility_type=utility_type
            ) # -> [B * 512, 1, 2b, 2b]

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

            i = i + 1

        correct = 0
        total = 0
        for vl in valid_loader:
            images = vl['point']
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


if __name__ == '__main__':
    main()
