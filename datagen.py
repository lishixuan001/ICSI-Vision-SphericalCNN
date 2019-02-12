import numpy as np
import h5py
import lie_learn.spaces.S2 as S2
import argparse
from logger import *
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.spatial import distance as spdist


class MNIST(Dataset):  # 60000 * 512 * 2
    """2D point dataset for MNIST"""

    def __init__(self, tensor_train, tensor_label):
        self.data = tensor_train # (60000, 512, 2b, 2b)
        self.label = tensor_label # (60000, )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        point = self.data[idx] # (512, 2b, 2b)
        label = self.label[idx]
        sample = {'data': point, 'label': label}
        return sample


def get_radius(point_cloud):
    """
    calculate the minimal distance of the given point cloud
    :param point_cloud: tensor (train_size, num_points, num_dims)
    :return: half of the minimal distance between 2 points, which is assigned as the radius of sphere
    """
    num_points_in_cloud = point_cloud.shape[0]
    min_distance = np.inf
    for idx in range(num_points_in_cloud):
        points_coords = point_cloud[idx]
        pairwise_point_distances = spdist.pdist(points_coords)
        min_distance = min(pairwise_point_distances.min(), min_distance)
    return min_distance / 2.0


def get_projection_grid(b, point_cloud, radius, grid_type="Driscoll-Healy"):
    """
    returns the spherical grid in euclidean coordinates, which, to be specify,
    for each image in range(train_size):
        for each point in range(num_points):
            generate the 2b * 2b S2 points , each is (x, y, z)
    therefore returns tensor (train_size * num_points, 2b * 2b, 3)
    :param b: the number of grids on the sphere
    :param point_cloud: tensor (train_size * num_points, num_dim)
    :param grid_type: "Driscoll-Healy"
    :return: tensor (train_size *num_points, 4 * b * b, 3)
    """
    assert type(point_cloud) == torch.Tensor
    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[-1] == 3

    # theta in shape (2b, 2b), range [0, pi]; phi range [0, 2 * pi]
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    theta = torch.from_numpy(theta)
    phi = torch.from_numpy(phi)

    x_ = radius * torch.sin(theta) * torch.cos(phi)
    """x will be reshaped to have one dimension of 1, then can broadcast
    look this link for more information: https://pytorch.org/docs/stable/notes/broadcasting.html
    """
    x = x_.reshape((1, 4 * b * b)) # tensor (1, 4 * b * b)
    px = point_cloud[:, 0].reshape((-1, 1)) # tensor (train_size * num_points, 1)
    x = x + px  # (train_size * num_points, 4 * b * b)

    # same for y and z
    y_ = radius * torch.sin(theta) * torch.sin(phi)
    y = y_.reshape((1, 4 * b * b))
    py = point_cloud[:, 1].reshape((-1, 1))
    y = y + py

    z_ = radius * torch.cos(theta)
    z = z_.reshape((1, 4 * b * b))
    pz = point_cloud[:, 2].reshape((-1, 1))
    z = z + pz

    # give x, y, z extra dimension, so that it can concat by that dimension
    x = torch.unsqueeze(x, 2)  # (train_size * num_points, 4 * b * b, 1)
    y = torch.unsqueeze(y, 2)
    z = torch.unsqueeze(z, 2)
    grid = torch.cat((x, y, z), 2) # (train_size * num_points, 4 * b * b, 3)
    return grid


def pairwise_distance(grid, point_cloud, logger, ctype="Gaussian"):
    """Compute the distance between a point cloud and grid
    :param ctype: "Gaussian" or "Potential"
    :param grid: tensor (train_size * num_points, 2b * 2b, 3)
    :param point_cloud: tensor (train_size * num_points, num_dims)
    :return: pairwise distance: (train_size * num_points, 2b * 2b)
    """
    assert type(point_cloud) == torch.Tensor
    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[-1] == 3 # num_dims = 3

    dim0 = point_cloud.shape[0] # dim0 = train_size * num_points
    dim1 = grid.shape[1] # dim1 = 2b * 2b

    point_cloud_transpose = torch.t(point_cloud)  # (3, dim0)

    """point_cloud_transpose_square_sum will first do element-wise square operation,
    and then sum up the (3, dim0) function along the 0-axis,
    which, intuitively, will get x^2 + y^2 + z^2 for each point in point_cloud
    """
    point_cloud_transpose_square_sum = torch.sum(point_cloud_transpose ** 2, dim=0, keepdim=True) # (1, dim0)

    result = np.zeros((dim0, dim1))  # initialize the result tensor
    for i in range(dim0):  # again, dim0 = train_size * num_points
        """mask_point is the tensor (dim1, dim0), which, will change the mask value of the i-th point
         (column) to zero, so that when sum up, will not count the distance between the i-th point
        and sphere point origin in i-th point
        """
        mask_point = torch.ones((dim1, dim0), dtype=point_cloud.dtype)
        mask_point[:, i] = 0
        """for each point in range(dim0), get a sphere around it, each_point_grid [2b * 2b, 3] 
        """
        each_point_grid = grid[i]  # (dim1, 3)
        grid_square_sum = torch.sum(each_point_grid ** 2, dim=-1, keepdim=True) # (dim1, 1)

        gr_product_point = torch.matmul(each_point_grid, point_cloud_transpose) # matrix product
        gr_product_point = -2 * gr_product_point  # (dim1, dim0)

        sum_up = gr_product_point + grid_square_sum + point_cloud_transpose_square_sum
        sum_up = sum_up * mask_point # (dim1, dim0)
        sum_up = torch.sum(sum_up, dim=-1)  # (dim1)
        assert len(sum_up.shape) == 1

        if ctype == "Gaussian":
            """the value of sum_up is around 1024, so I add the constant to make sure transform is not NaN
            """
            transform = torch.exp(- sum_up / 1000.0)
        if ctype == "Potential": # borrow from molecules charge
            """ the value of sum_up ** (-1) is around 0.0010
            """
            transform = sum_up ** (-1)

        result[i] = transform

        logger.info(i)

    return result  # (train_size * num_points, 2b * 2b)


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
        train_size = 2
    else:
        train_size = f_train['data'].shape[0]
    num_points = f_train['data'].shape[1]
    logger.info("finish loading MNIST data and basic configuration")
    logger.warning("haven't loaded test data set")
    logger.warning("also, haven't split the validation set")

    # add [()] can read numpy.ndarray file from h5py file
    train_np_dataset = f_train['data'][()][0:train_size]
    train_torch_dataset = torch.from_numpy(train_np_dataset)  # this convert from numpy to torch

    # have to calculate the min distance at first, otherwise cannot distinguish which image it is in
    radius = get_radius(train_torch_dataset)

    if train_torch_dataset.shape[-1] == 2:
        """if deal with 2D point set, have to add one dimension as z dimension
        z dimension should be padded with 0, since point is ranged from -1 to 1, 0 is the average value
        """
        zero_padding = torch.zeros((train_size, num_points, 1), dtype=train_torch_dataset.dtype)
        train_torch_dataset = torch.cat((train_torch_dataset, zero_padding), -1)
    train_torch_dataset = train_torch_dataset.reshape((-1, 3))  # (train_size * num_points, 3)

    # get_projection_grid returns tensor (train_size * num_points, 4 * b * b, 3)
    grid = get_projection_grid(b=args.bandwidth,
                               point_cloud=train_torch_dataset,
                               radius=radius,
                               grid_type="Driscoll-Healy")

    # pairwise_distance returns tensor (train_size * num_points, 2b * 2b)
    tensor_data_train = pairwise_distance(grid=grid,
                                          point_cloud=train_torch_dataset,
                                          logger=logger,
                                          ctype="Gaussian")
    tensor_data_train = tensor_data_train.reshape(train_size, num_points, 2 * args.bandwidth, 2 * args.bandwidth)
    tensor_label_train = f_train['labels'][()]

    trainset = DataLoader(dataset=MNIST(tensor_data_train, tensor_label_train),
                          batch_size=args.batchsize,
                          shuffle=True)

    with open("output.txt", "w+") as output_file:
        for ts in trainset:
            output_file.write("{}\n".format(ts))
    # question: how to transform test set? How should the radius set, and the


if __name__ == '__main__':
    main()
