import numpy as np
import h5py
import lie_learn.spaces.S2 as S2
import argparse
from logger import *
from torch.utils.data import Dataset, DataLoader
import torch

train_path = "mnistPC/train.hdf5"
test_path = "mnistPC/test.hdf5"
# the point is ranged from -1 to 1
f_train = h5py.File(train_path)
f_test = h5py.File(test_path)


class MNIST(Dataset): # 60000 * 512 * 2
    """2D point dataset for MNIST"""
    def __init__(self, h5py_file):
        self.file = h5py_file

    def __len__(self):
        return self.file['data'].shape[0]

    def __getitem__(self, idx):
        point = self.file['data'][idx]
        label = self.file['labels'][idx]
        sample = {'data': point, 'label': label}
        return sample


def get_projection_grid(b, point_cloud, radius, grid_type="Driscoll-Healy"):
    """
    this function is inspired from:
    https://github.com/jonas-koehler/s2cnn/blob/07ad63441730811dcda2ccf9ac4027f406f5b605/examples/mnist/gendata.py#L65
    returns the spherical grid in euclidean coordinates
    :param b: the number of grids on the sphere
    :param point_cloud: tensor (train_size, num_point, num_dim)
    :param grid_type: "Driscoll-Healy"
    :return: tensor (train_size, num_point, 3)
    """
    train_size = point_cloud.shape[0]
    num_point = point_cloud.shape[1]
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type) # theta in range[0, pi], phi in range[0, 2pi]
    x_ = radius * np.sin(theta) * np.cos(phi) + point_cloud[:, :, 0]
    y_ = radius * np.sin(theta) * np.sin(phi) + point_cloud[:, :, 1]
    z_ = radius * np.cos(theta) + point_cloud[:, :, 2]
    x = torch.unsqueeze(x_, 2)
    y = torch.unsqueeze(y_, 2)
    z = torch.unsqueeze(z_, 2)
    grid = torch.cat((x, y, z), 2)
    assert grid.shape == torch.Size([train_size, 4 * b * b, 3])
    return grid


def pairwise_distance(grid, point_cloud, type="Gaussian"):
    """Compute pairwise distance of a point cloud.
    :param grid: tensor (train_size, 2b * 2b, 3)
    :param point_cloud: tensor (train_size, num_points, num_dims)
    :return: pairwise distance: (train_size, 2b, 2b)
    """
    point_cloud = torch.from_numpy(point_cloud)
    train_size = point_cloud.shape[0]  # point_cloud.get_shape().as_list()[0]
    num_points = point_cloud.shape[1]
    if point_cloud.shape[-1] == 2:
        zero_padding = torch.zeros((train_size, num_points, 1), dtype=point_cloud.dtype)
        point_cloud = torch.cat((point_cloud, zero_padding), 2)

    assert point_cloud.shape[-1] == 3
    # point_cloud = torch.squeeze(point_cloud)
    # if train_size == 1:
    #     point_cloud = point_cloud.unsqueeze(0)  # torch.expand_dims(point_cloud, 0)

    point_cloud_transpose = point_cloud.permute(0, 2, 1) # (train_size, num_dims, num_points)
    # torch.transpose(point_cloud, perm=[0, 2, 1])
    # point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = torch.matmul(grid, point_cloud_transpose) # (train_size, 2b * 2b, num_points)
    point_cloud_inner = -2 * point_cloud_inner # (train_size, 2b * 2b, num_points)
    point_cloud_square = torch.sum(point_cloud ** 2, dim=-1, keepdim=True) # (train_size, num_points, 1)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1) # (train_size, 1, num_points)
    grid_square = torch.sum(grid ** 2, dim=-1, keepdim=True) # (train_size, 2b * 2b, 1)
    sum_up = grid_square + point_cloud_inner + point_cloud_square_tranpose # (train_size, 2b * 2b, num_points)

    if type == "Gaussian":
        transform = torch.exp(sum_up)

    if type == "Potential":
        transform = sum_up ** (-1)

    transform = torch.sum(transform, dim=-1, keepdim=True)  # (train_size, 2b * 2b, 1) ## 没有算j != i
    return torch.squeeze(transform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=30,
                        required=False)
    parser.add_argument("--noise",
                        help="the rotational noise applied on the sphere",
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument("--chunk_size",
                        help="size of image chunk with same rotation",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="s2_mnist.gz",
                        required=False)
    parser.add_argument("--no_rotate_train",
                        help="do not rotate train set",
                        dest='no_rotate_train', action='store_true')
    parser.add_argument("--no_rotate_test",
                        help="do not rotate test set",
                        dest='no_rotate_test', action='store_true')

    args = parser.parse_args()

    logging.info("getting MNIST data")
    trainset = DataLoader(MNIST(f_train), shuffle = True)
    testset = DataLoader(MNIST(f_test), shuffle = True)

    grid = get_projection_grid(b=args.bandwidth)



if __name__ == '__main__':
    main()