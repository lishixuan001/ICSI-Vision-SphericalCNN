import numpy as np
import lie_learn.spaces.S2 as S2
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import os
import gc
# import psutil
from scipy.spatial import distance as spdist
from enum import Enum
from torch.utils.data.sampler import SubsetRandomSampler
import time


class MNIST(Dataset):  # 60000 * 512 * 2
    """2D point dataset for MNIST"""

    def __init__(self, tensor_train, tensor_label):
        self.points = tensor_train  # (60000, 512, 3)
        self.labels = tensor_label  # (60000, )

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        point = self.points[idx]  # (512, 3)
        label = self.labels[idx]
        sample = {'point': point, 'label': label}
        return sample


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{}: {} sec'.format(method.__name__, te-ts))
        return result
    return timed


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def debug_memory():
    import collections, gc
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))
  

def get_radius(point_cloud):
    """
    calculate the minimal distance of the given point cloud
    :param point_cloud: tensor (train_size, num_points, num_dims)
    :return: half of the minimal distance between 2 points, which is assigned as the radius of sphere
    """
    num_points_in_cloud = point_cloud.shape[0]
    min_distance = 1000.0
    for idx in range(num_points_in_cloud):
        points_coords = point_cloud[idx]
        pairwise_point_distances = spdist.pdist(points_coords)
        min_distance = min(pairwise_point_distances.min(), min_distance)
        if min_distance == 0:
            print(idx)
    return min_distance / 2.0


@timeit
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


@timeit
def get_grid(b, radius, grid_type="Driscoll-Healy"):
    """
    returns the spherical grid in euclidean coordinates, which, to be specify,
    for each image in range(train_size):
        for each point in range(num_points):
            generate the 2b * 2b S2 points , each is (x, y, z)
    therefore returns tensor (train_size * num_points, 2b * 2b, 3)
    :param b: the number of grids on the sphere
    :param radius: the radius of each sphere
    :param grid_type: "Driscoll-Healy"
    :return: tensor (batch_size, num_points, 4 * b * b, 3)
    """
    # theta in shape (2b, 2b), range [0, pi]; phi range [0, 2 * pi]
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    theta = torch.from_numpy(theta).cuda()
    phi = torch.from_numpy(phi).cuda()

    x_ = radius * torch.sin(theta) * torch.cos(phi)
    """x will be reshaped to have one dimension of 1, then can broadcast
    look this link for more information: https://pytorch.org/docs/stable/notes/broadcasting.html
    """
    x = x_.reshape((1, 4 * b * b))  # tensor (1, 4 * b * b)

    # same for y and z
    y_ = radius * torch.sin(theta) * torch.sin(phi)
    y = y_.reshape((1, 4 * b * b))

    z_ = radius * torch.cos(theta)
    z = z_.reshape((1, 4 * b * b))

    grid = torch.cat((x, y, z), dim=0)  # (3, 4 * b * b)
    assert grid.shape == torch.Size([3, 4 * b * b])
    # grid = grid.reshape((1, 4 * b * b, 3))
    return grid


@timeit
def pairwise_distance(grid, images, ctype="Gaussian"):
    """Compute the distance between a point cloud and grid
    :param ctype: "Gaussian" or "Potential"
    :param grid: tensor (3, 2b * 2b)
    :param images: tensor (B, num_points, 3)
    :return: pairwise distance: (B, num_points, 2b * 2b)
    """
    assert type(images) == torch.Tensor
    assert len(images.shape) == 3
    assert images.shape[-1] == 3  # num_dims = 3
    batch_size = images.shape[0]
    num_points = images.shape[1]
    # images = images.cuda()
    grid = grid.cuda()

    """first compute the inner product of x0 in each images
    images_origin is the coordinate of x0, repeat with num_points time on dim 1
    images_diff is the difference of all other points - images_origin
    """
    images_origin = images[:, 0, :].unsqueeze(1).repeat(1, num_points, 1)  # B, 1, 3 -> B, 512, 3
    images_diff = images - images_origin  # -> B, 512, 3
    #print("images", images_diff)
    """grid_batch is the grid on the sphere, originated at 0, 0, 0 """
    grid_batch = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # -> B, 3, 4 * b * b
    #print("grid", grid_batch)
    """bmm takes (b, m, n) * (b, n, p) -> (b, m, p), it do batch matrix mul
    and then I sum up the product along the num_points dimension,
    in this way I get the first point's sphere value 4 * b * b
    """
    product = torch.bmm(images_diff, grid_batch)  # -> B, 512, 4 * b * b
    product_sum = torch.sum(product, dim=1)  # -> B, 4 * b * b
    #print("product sum", product_sum)
    """then I use the product computed above to do the later calculation
    I should, for each point, add up the product of <x0, grid>, 
    and also the N * <x0-xi, grid>, which is (1, 4 * b * b), **note there is a minus
    then do it along the hum_points: just use the product above to times -512 to get the result
    """
    left = product_sum.unsqueeze(1).repeat(1, num_points, 1)  # -> B, 512, 4 * b * b
    right = torch.mul(product, -num_points)
    images_result = torch.add(left, right)

    image_min = images_result.min()
    #print("images min", image_min)
    image_max = images_result.max()
    #print("images max", image_max)
    images_result = (images_result - image_min) / (image_max - image_min)

   # debug_memory()
    return images_result


def translation(images, bandwidth, radius, utility_type):
    """
    do the translation of the images points
    :param images: [B, 512, 3]
    :param bandwidth: b when generate spherical grid
    :param utility_type: "Gaussian" or "Potential"
    :return: [B * 512, 1, 2b, 2b]
    """
    images = images.cuda()
    grid_images = get_grid(
        b=bandwidth,
       # images=images,
        radius=radius,
        grid_type="Driscoll-Healy"
    )  # -> (B, 4 * b * b, 3)
    data_train = pairwise_distance(
        grid=grid_images,
        images=images,
        ctype=utility_type
    )  # -> (B, 512, 4 * b * b)
    data_train = data_train.reshape(-1, 1, 2 * bandwidth, 2 * bandwidth)
    return data_train


def split_train_and_valid(trainset, labelset, batch_size, valid_size=0.1):
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

    train_loader = DataLoader(MNIST(trainset, labelset), batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(MNIST(trainset, labelset), batch_size=batch_size, sampler=valid_sampler)
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
