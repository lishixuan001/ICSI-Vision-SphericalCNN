import numpy as np
import lie_learn.spaces.S2 as S2
from torch.utils.data import Dataset
import torch
from scipy.spatial import distance as spdist
from enum import Enum

class MNIST(Dataset):  # 60000 * 512 * 2
    """2D point dataset for MNIST"""

    def __init__(self, tensor_train, tensor_label):
        self.points = tensor_train # (60000, 512, 2b, 2b)
        self.labels = tensor_label # (60000, )

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        point = self.points[idx] # (512, 2b, 2b)
        label = self.labels[idx]
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

    point_cloud_transpose = torch.t(point_cloud)  # (dim0, 3) ==> (3, dim0)

    """point_cloud_transpose_square_sum will first do element-wise square operation,
    and then sum up the (3, dim0) function along the 0-axis,
    which, intuitively, will get x^2 + y^2 + z^2 for each point in point_cloud
    """
    point_cloud_transpose_square_sum = torch.sum(torch.pow(point_cloud_transpose, 2), dim=(0,), keepdim=True) # (1, dim0) ==> x^2 (self)

    result = np.zeros((dim0, dim1))  # initialize the result tensor
    for i in range(dim0):  # again, dim0 = train_size * num_points
        """mask_point is the tensor (dim1, dim0), which, will change the mask value of the i-th point
         (column) to zero, so that when sum up, will not count the distance between the i-th point
        and sphere point origin in i-th point
        """
        mask_point = torch.ones((dim1, dim0), dtype=point_cloud.dtype)
        mask_point[:, i] = 0 # TODO: Check Mask_Point
        """for each point in range(dim0), get a sphere around it, each_point_grid [2b * 2b, 3]
        """
        each_point_grid = grid[i]  # (dim1, 3)
        grid_square_sum = torch.sum(each_point_grid ** 2, dim=(-1,), keepdim=True) # (dim1, 1) ==> y^2 (other)

        gr_product_point = torch.matmul(each_point_grid, point_cloud_transpose) # matrix product -> (dim1, dim0)
        gr_product_point = torch.mul(gr_product_point, -2) # element-wise multiplication ==> -2xy

        dist_sum_up = torch.add( torch.add(gr_product_point, grid_square_sum), point_cloud_transpose_square_sum )
        assert ctype in UtilityTypes
        # signal_matrix ==> The matrix of elements to be summed for S2_Distance_Map (<<Spherical CNN>> 5.4.2 -> line5)
        if ctype == UtilityTypes.Gaussian:
            signal_matrix = torch.exp(torch.mul(dist_sum_up, -1)) # FIXME: Check Result
        else:
            # ctype == UtilityTypes.Potential
            signal_matrix = torch.pow(dist_sum_up, -1)
        masked_signal_matrix = torch.mul(signal_matrix, mask_point) # (dim1, dim0) => 2b*2b distance map for every point (n)
        summed_masked_signal_matrix = torch.sum(masked_signal_matrix, dim=(-1,))  # (dim1)
        assert summed_masked_signal_matrix.size() == torch.Size([dim1]) # TODO: Mind The Correctness

        result[i] = summed_masked_signal_matrix

        logger.info(i)

    return result  # (train_size * num_points, 2b * 2b) => (dim0, dim1)

##########################################################
#                       Enum Types                       #
##########################################################
class UtilityTypes(Enum):
    Gaussian = "Gaussian"
    Potential = "Potential"