import numpy as np
import h5py
import lie_learn.spaces.S2 as S2
import argparse
from logger import *
from torch.utils.data import Dataset, DataLoader
import torch
from helper_methods import *
from scipy.spatial import distance as spdist

def main():
    logger = setup_logger('data_generate')
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--demo",
                        help="if demo is true, then only load 10 image",
                        type=bool,
                        default=True,
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="s2_mnist.gz",
                        required=False)
    args = parser.parse_args()
    logger.info("call with args: \n{}".format(args))
    logger.info("getting MNIST data")

    f_train = h5py.File(args.train_file_path)  # the point is ranged from -1 to 1
    f_test = h5py.File(args.test_file_path)
    if args.demo:
        train_size = 1 # num of images to input
    else:
        train_size = f_train['data'].shape[0]
    num_points = f_train['data'].shape[1]
    logger.info("finish loading MNIST data and basic configuration")

    # TODO: LOAD TEST DATA SET
    logger.warning("haven't loaded test data set")
    logger.warning("also, haven't split the validation set")

    # add [()] can read h5py file as numpy.ndarray
    train_np_dataset = f_train['data'][()][0:train_size] # train_size * 512 * 2
    train_torch_dataset = torch.from_numpy(train_np_dataset)  # this convert from numpy to torch

    # have to calculate the min distance at first, otherwise cannot distinguish which image it is in
    radius = get_radius(train_torch_dataset)


    if train_torch_dataset.shape[-1] == 2:
        """if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (train_size * num_points, 3) -> z-dimenson additionally padded by 0 -> (x, y, 0)
        zero_padding = torch.zeros((train_size, num_points, 1), dtype=train_torch_dataset.dtype)
        train_torch_dataset = torch.cat((train_torch_dataset, zero_padding), -1)

    #  (train_size, 512, 3) ==> (train_size * 512, 3)
    train_torch_dataset = train_torch_dataset.reshape((-1, 3))

    # get_projection_grid returns tensor (train_size * num_points, 4 * b * b, 3)
    grid = get_projection_grid(b=args.bandwidth,
                               point_cloud=train_torch_dataset,
                               radius=radius,
                               grid_type="Driscoll-Healy")

    # pairwise_distance returns tensor (train_size * num_points, 2b * 2b)
    tensor_data_train = pairwise_distance(grid=grid,
                                          point_cloud=train_torch_dataset,
                                          logger=logger,
                                          ctype=UtilityTypes.Gaussian)

    # TODO: Check reshape before/after details
    # (train_size * num_points, 2b * 2b) -> (train_size, num_points, 2b * 2b)
    tensor_data_train = tensor_data_train.reshape(train_size, num_points, 2 * args.bandwidth, 2 * args.bandwidth)

    tensor_label_train = f_train['labels'][()][0:train_size]

    trainset = DataLoader(dataset=MNIST(tensor_data_train, tensor_label_train),
                          batch_size=args.batchsize,
                          shuffle=True)

    with open("output.txt", "w+") as output_file:
        for ts in trainset:
            output_file.write("{}\n".format(ts))
    # question: how to transform test set? How should the radius set, and the


if __name__ == '__main__':
    main()
