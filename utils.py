import numpy as np
import torch
from pdb import set_trace as st


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0] #point_cloud.get_shape().as_list()[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0) #torch.expand_dims(point_cloud, 0)
    
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    #torch.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = torch.sum( point_cloud**2, dim=-1, keepdim = True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1) #torch.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    nn_idx = np.argsort(adj_matrix, axis=-1)[:,:,1:k+1] #torch.nn.top_k(neg_adj, k=k)
    return nn_idx


def sample_subset(idx_input, num_output):
    return np.random.choice(idx_input, num_output ,replace = False)


def sdt(x, grid = 20, sigma = 1):
    dim = x.shape[2]
    num_point = x.shape[1]
    out = np.zeros((x.shape[0],x.shape[1],grid**dim,1))
    linspace = np.linspace(0,1,grid)
    mesh = linspace
    for i in range(dim-1):
        mesh = np.meshgrid(mesh, linspace)
    mesh = np.array(mesh)
    mesh = mesh.reshape(mesh.shape[0], -1)
    for batch_id in range(x.shape[0]):
        for id_, var in enumerate(mesh.T):
            var = var.reshape((1, -1))
            core_dis = np.sum( (np.squeeze(x[batch_id, ...]) -  np.repeat(var, num_point, axis = 0) ) **2, axis =1) *1.0 /(2*sigma)
            out[batch_id, :, id_,0] = np.exp( -core_dis)
    return out

def main():
    point_cloud = torch.arange(4).reshape((1, 2, 2))
    print(pairwise_distance(point_cloud))

if __name__ == '__main__':
    main()