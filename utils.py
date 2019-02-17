import numpy as np
import lie_learn.spaces.S2 as S2
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from scipy.spatial import distance as spdist
from enum import Enum
from torch.utils.data.sampler import SubsetRandomSampler


class MNIST(Dataset):  # 60000 * 512 * 2
    """2D point dataset for MNIST"""

    def __init__(self, tensor_train, tensor_label):
        self.points = tensor_train  # (60000, 512, 2b, 2b)
        self.labels = tensor_label  # (60000, )

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        point = self.points[idx]  # (512, 2b, 2b)
        label = self.labels[idx]
        sample = {'point': point, 'label': label}
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


def get_projection_grid(b, images, radius, grid_type="Driscoll-Healy"):
    """
    returns the spherical grid in euclidean coordinates, which, to be specify,
    for each image in range(train_size):
        for each point in range(num_points):
            generate the 2b * 2b S2 points , each is (x, y, z)
    therefore returns tensor (train_size * num_points, 2b * 2b, 3)
    :param b: the number of grids on the sphere
    :param images: tensor (batch_size, num_points, 3)
    :param radius: the radius of each sphere
    :param grid_type: "Driscoll-Healy"
    :return: tensor (batch_size, num_points, 4 * b * b, 3)
    """
    assert type(images) == torch.Tensor
    assert len(images.shape) == 3
    assert images.shape[-1] == 3
    batch_size = images.shape[0]
    num_points = images.shape[1]
    images = images.reshape((-1, 3))  # -> (B * 512, 3)

    # theta in shape (2b, 2b), range [0, pi]; phi range [0, 2 * pi]
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    theta = torch.from_numpy(theta).cuda()
    phi = torch.from_numpy(phi).cuda()

    x_ = radius * torch.sin(theta) * torch.cos(phi)
    """x will be reshaped to have one dimension of 1, then can broadcast
    look this link for more information: https://pytorch.org/docs/stable/notes/broadcasting.html
    """
    x = x_.reshape((1, 4 * b * b))  # tensor (1, 4 * b * b)
    px = images[:, 0].reshape((-1, 1))  # tensor (batch_size * 512, 1)
    x = x + px  # (batch_size * num_points, 4 * b * b)

    # same for y and z
    y_ = radius * torch.sin(theta) * torch.sin(phi)
    y = y_.reshape((1, 4 * b * b))
    py = images[:, 1].reshape((-1, 1))
    y = y + py

    z_ = radius * torch.cos(theta)
    z = z_.reshape((1, 4 * b * b))
    pz = images[:, 2].reshape((-1, 1))
    z = z + pz

    # give x, y, z extra dimension, so that it can concat by that dimension
    x = torch.unsqueeze(x, 2)  # (B * 512, 4 * b * b, 1)
    y = torch.unsqueeze(y, 2)
    z = torch.unsqueeze(z, 2)
    grid = torch.cat((x, y, z), 2)  # (B * 512, 4 * b * b, 3)
    grid = grid.reshape((batch_size, num_points, 4 * b * b, 3))
    return grid


def pairwise_distance(grid, images, ctype="Gaussian"):
    """Compute the distance between a point cloud and grid
    :param ctype: "Gaussian" or "Potential"
    :param grid: tensor (B, num_points, 2b * 2b, 3)
    :param images: tensor (B, num_points, 3)
    :return: pairwise distance: (B, num_points, 2b * 2b)
    """
    assert type(images) == torch.Tensor
    assert len(images.shape) == 3
    assert images.shape[-1] == 3  # num_dims = 3
    batch_size = images.shape[0]
    num_points = images.shape[1]
    dim1 = grid.shape[2]  # dim1 = 2b * 2b
    grid = grid.cuda()
    result = torch.zeros((batch_size, num_points, dim1)).cuda()  # initialize the result tensor

    total = batch_size * num_points  # add for progress bar
    for each in range(batch_size):
        """basic distance is calculated in this way: (x - y)^2 = x ^ 2 - 2 * x * y + y ^ 2
        """
        image = images[each]
        image_t = torch.t(image)  # (num_points, 3) -> (3, num_points)
        """image_ts_sum (image transpose square sum) will first do element-wise square operation,
        and then sum up the (3, num_points) function along the 0-axis,
        which, intuitively, will get x^2 + y^2 + z^2 for each point in image
        (x ^ 2)
        """
        image_ts_sum = torch.sum(torch.pow(image_t, 2), dim=(0,), keepdim=True)  # (1, num_points) which is x^2 (self)
        image_result = torch.zeros((num_points, dim1)).cuda()  # initialize the result tensor
        for point in range(num_points):
            """mask_point is the tensor (dim1, dim0), which, will change the mask value of the i-th point
            (column) to zero, so that when sum up, will not count the distance between the i-th point
            and sphere point origin in i-th point
            """
            mask_points = torch.ones((dim1, num_points), dtype=image.dtype).cuda()
            mask_points[:, point] = 0

            """for each point in the image, get a sphere around it, each_point_grid [2b * 2b, 3], and calculate the 
            element-wise square of the point_grid -> grid_square_sum 
            (y ^ 2)
            """
            each_point_grid = grid[each, point, :, :].reshape((dim1, 3))  # (dim1, 3)
            grid_square_sum = torch.sum(each_point_grid ** 2, dim=(-1,), keepdim=True)
            # -> (dim1, 1) which is y^2 (other)

            """Also we have to calculate the matrix product of each_point_grid (2b * 2b, 3) and image_t (2, num_points),
            which is a preparation of the distance between each point 2b * 2b sphere grid and other points in the image 
            (- 2 * x * y)
            """
            matrix_product = torch.matmul(each_point_grid, image_t)  # matrix product -> (dim1, num_points) which is xy
            matrix_product = torch.mul(matrix_product, -2)  # element-wise multiplication -> -2xy

            """Finally we sum up the distance
            """
            dist_sum_up = torch.add(torch.add(matrix_product, grid_square_sum), image_ts_sum)  # (dim1, num_points)

            """get distance translated into spherical signal"""
            assert ctype in UtilityTypes
            # signal_matrix ==> The matrix of elements to be summed for S2_Distance_Map
            # (<<Spherical CNN>> 5.4.2 -> line5)
            if ctype == UtilityTypes.Gaussian:
                signal_matrix = torch.exp(torch.mul(dist_sum_up, -1))  # FIXME: Check Result
            else:
                # ctype == UtilityTypes.Potential
                signal_matrix = torch.pow(dist_sum_up, -1)
            # (dim1, num_points) => 2b*2b distance map for every point in range(num_points)
            masked_signal_matrix = torch.mul(signal_matrix, mask_points)
            summed_masked_signal_matrix = torch.sum(masked_signal_matrix, dim=(-1,))  # (dim1)
            assert summed_masked_signal_matrix.size() == torch.Size([dim1])  # TODO: Mind The Correctness

            image_result[point] = summed_masked_signal_matrix

            progress(i, total)

        image_min = image_result.min()
        image_max = image_result.max()
        image_result = (image_result - image_min) / (image_max - image_min)
        result[each] = image_result

    return result  # (B, num_points, 2b * 2b)


def translation(images, bandwidth, radius, utility_type):
    """
    do the translation of the images points
    :param images: [B, 512, 3]
    :param bandwidth: b when generate spherical grid
    :param utility_type: "Gaussian" or "Potential"
    :return: [B * 512, 1, 2b, 2b]
    """
    images = images.cuda()
    grid_images = get_projection_grid(
        b=bandwidth,
        images=images,
        radius=radius,
        grid_type="Driscoll-Healy"
    )  # -> (B, 512, 4 * b * b, 3)
    data_train = pairwise_distance(
        grid=grid_images,
        images=images,
        ctype=utility_type
    )  # -> (B, 512, 4 * b * b)
    data_train = data_train.reshape(-1, 1, 2 * bandwidth, 2 * bandwidth)
    return data_train


def split_train_and_valid(trainset, batch_size, valid_size=0.1):
    """
    utility function for loading and returning train and valid multi-process iterators
    :param trainset: train set [train_size, num_points, 3]
    :param batch_size: the batch size of the dataloader
    :param valid_size: percentage that training set split into validation set
    :return:
    - train_loader: training set iterator
    - valid_loader: validation set iterator
    """
    err_msg = "[!] valid_size should range in (0, 1)"
    assert ((valid_size >= 0) and (valid_size <= 1)), err_msg

    train_size = len(trainset)  # (train_size, 512, 3)
    indices = list(range(train_size))
    split = int(np.floor(valid_size * train_size))
    err_msg = "[!] train set is too small for split"
    assert split != 0, err_msg

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader


def progress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('\r[%s] %s%s' % (bar, percents, '%'), end='')
    # sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()


class UtilityError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


##########################################################
#                       Enum Types                       #
##########################################################
class UtilityTypes(Enum):
    Gaussian = "Gaussian"
    Potential = "Potential"
