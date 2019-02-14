import h5py
import argparse
import gzip
import pickle
import os
from logger import *
from helper_methods import *


def main():
    logger = setup_logger('data_generate')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path",
                        help="the path of train file",
                        type=str,
                        default='./mnistPC/train.hdf5',
                        required=False)
    parser.add_argument("--test-file-path",
                        help="the path of test file",
                        type=str,
                        default='./mnistPC/test.hdf5',
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
    parser.add_argument("--signal-type",
                        help="Gaussian or Potential",
                        type=str,
                        default="Gaussian",
                        required=False)
    parser.add_argument("--output-prefix",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="mnistPC",
                        required=False)
    args = parser.parse_args()
    logger.info("call with args: \n{}".format(args))
    logger.info("getting MNIST data")

    f_train = h5py.File(args.train_file_path)  # the point is ranged from -1 to 1
    f_test = h5py.File(args.test_file_path)
    if args.demo:
        train_size = 2
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

    # add [()] can read h5py file as numpy.ndarray
    """load train set"""
    train_np_dataset = f_train['data'][()][0:train_size]  # train_size * 512 * 2
    train_torch_dataset = torch.from_numpy(train_np_dataset)  # convert from numpy.ndarray to torch.Tensor

    """load test set"""
    test_np_dataset = f_test['data'][()][0:test_size]
    test_torch_dataset = torch.from_numpy(test_np_dataset)  # convert from numpy.ndarray to torch.Tensor

    """calculate the radius"""
    # have to calculate the min distance at first, otherwise cannot distinguish which image it is in
    radius = get_radius(train_torch_dataset)

    """transform train data"""
    if train_torch_dataset.shape[-1] == 2:
        """if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (train_size * num_points, 3) -> z-dimension additionally padded by 0 -> (x, y, 0)
        zero_padding = torch.zeros((train_size, num_points, 1), dtype=train_torch_dataset.dtype)
        train_torch_dataset = torch.cat((train_torch_dataset, zero_padding), -1)

    #  (train_size, 512, 3) ==> (train_size * 512, 3)
    train_torch_dataset = train_torch_dataset.reshape((-1, 3))

    # get_projection_grid returns tensor (train_size * num_points, 4 * b * b, 3)
    grid_train = get_projection_grid(b=args.bandwidth,
                                     point_cloud=train_torch_dataset,
                                     radius=radius,
                                     grid_type="Driscoll-Healy")

    logger.info("start calculate pairwise distance for train set")
    # pairwise_distance returns tensor (train_size * num_points, 2b * 2b)
    tensor_data_train = pairwise_distance(grid=grid_train,
                                          point_cloud=train_torch_dataset,
                                          ctype=utility_type)

    # TODO: Check reshape before/after details
    # (train_size * num_points, 2b * 2b) -> (train_size * num_points, 1, 2b * 2b)
    # !important change: should give points from different images
    tensor_data_train = tensor_data_train.reshape(train_size * num_points, 1, 2 * args.bandwidth, 2 * args.bandwidth)
    logger.info("finish!")

    """transform test data"""
    if test_torch_dataset.shape[-1] == 2:
        """if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        # (test_size * num_points, 3) -> z-dimension additionally padded by 0 -> (x, y, 0)
        zero_padding = torch.zeros((test_torch_dataset.shape[0], test_torch_dataset.shape[1], 1),
                                   dtype=test_torch_dataset.dtype)
        test_torch_dataset = torch.cat((test_torch_dataset, zero_padding), -1)

    #  (test_size, 512, 3) ==> (test_size * 512, 3)
    test_torch_dataset = test_torch_dataset.reshape((-1, 3))

    # get_projection_grid returns tensor (test_size * num_points, 4 * b * b, 3)
    grid_test = get_projection_grid(b=args.bandwidth,
                                    point_cloud=test_torch_dataset,
                                    radius=radius,
                                    grid_type="Driscoll-Healy")

    logger.info("start calculate pairwise distance for test set")
    # pairwise_distance returns tensor (test_size * num_points, 1, 2b * 2b)
    tensor_data_test = pairwise_distance(grid=grid_test,
                                         point_cloud=test_torch_dataset,
                                         ctype=utility_type)
    logger.info("finish!")

    # (test_size * num_points, 2b * 2b) -> (test_size * num_points, 1, 2b * 2b)
    tensor_data_test = tensor_data_test.reshape(test_size * num_points, 1, 2 * args.bandwidth, 2 * args.bandwidth)

    """load label"""
    tensor_label_train = f_train['labels'][()][0:train_size].repeat(num_points)
    tensor_label_test = f_test['labels'][()][0:test_size].repeat(num_points)
    assert tensor_label_train.shape[0] == tensor_data_train.shape[0]

    """generate train set & test set"""

    logger.info("finish loading the data set")

    logger.info("start saveing dataset")
    with gzip.open(os.path.join(args.output_prefix, 'train_mnist.gz'), 'wb') as f:
        pickle.dump(MNIST(tensor_data_train, tensor_label_train), f)
    with gzip.open(os.path.join(args.output_prefix, 'test_mnist.gz'), 'wb') as f:
        pickle.dump(MNIST(tensor_data_test, tensor_label_test), f)
    logger.info("finish saving dataset")


if __name__ == '__main__':
    main()
