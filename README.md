# PointCloud Classification & Segmentation With Spherical CNNs
> Implementation of Sphereical CNNs on PointCloud datasets

## Training & Running [Notes]
### Data Generation
* Logger messages are stored in **/log**, recording maintainable results
* Outputs (for **ts** in **trainset**) are stored in output.txt

## important notes
### explanation on same distance array
the same distance array comes from the Driscoll-Heayley algorithm: 
```python
theta, phi = S2.meshgrid(b=b, grid_type="Driscoll-Heayley")
```
where, the output of theta and phi is shown as follows:
```markdown
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988
 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988 0.05235988]
 ...
 [3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278
 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278 3.08923278]
```
theta share same value along the row.
As for phi:
```markdown
[0.         0.10471976 0.20943951 0.31415927 0.41887902 0.52359878
 0.62831853 0.73303829 0.83775804 0.9424778  1.04719755 1.15191731
 1.25663706 1.36135682 1.46607657 1.57079633 1.67551608 1.78023584
 1.88495559 1.98967535 2.0943951  2.19911486 2.30383461 2.40855437
 2.51327412 2.61799388 2.72271363 2.82743339 2.93215314 3.0368729
 3.14159265 3.24631241 3.35103216 3.45575192 3.56047167 3.66519143
 3.76991118 3.87463094 3.97935069 4.08407045 4.1887902  4.29350996
 4.39822972 4.50294947 4.60766923 4.71238898 4.81710874 4.92182849
 5.02654825 5.131268   5.23598776 5.34070751 5.44542727 5.55014702
 5.65486678 5.75958653 5.86430629 5.96902604 6.0737458  6.17846555]
 ...
 [0.         0.10471976 0.20943951 0.31415927 0.41887902 0.52359878
 0.62831853 0.73303829 0.83775804 0.9424778  1.04719755 1.15191731
 1.25663706 1.36135682 1.46607657 1.57079633 1.67551608 1.78023584
 1.88495559 1.98967535 2.0943951  2.19911486 2.30383461 2.40855437
 2.51327412 2.61799388 2.72271363 2.82743339 2.93215314 3.0368729
 3.14159265 3.24631241 3.35103216 3.45575192 3.56047167 3.66519143
 3.76991118 3.87463094 3.97935069 4.08407045 4.1887902  4.29350996
 4.39822972 4.50294947 4.60766923 4.71238898 4.81710874 4.92182849
 5.02654825 5.131268   5.23598776 5.34070751 5.44542727 5.55014702
 5.65486678 5.75958653 5.86430629 5.96902604 6.0737458  6.17846555]
```
phi share the same value along column.

Function which project from S2 space to R3 space is: (assume the sphere origin at (0, 0, 0))
$$
\begin{array}
x_ = radius * torch.sin(theta) * torch.cos(phi) \\
y_ = radius * torch.sin(theta) * torch.sin(phi) \\
z_ = radius * torch.cos(theta) \\
\end{array}
$$

Then, for x_ and y_, the first row is same for each grid in (2b * 2b) grids, since the value of theta is always 0,
and sin(0) = 0

As for z_, since the cos(0)=1, the first row of each grid is always 1

So basically, the first 2b points share the same (x, y, z), and therefore when calculating the distance with other points,
they should remain the same value.

### explanation on training batch
Firstly, I thought each image (which contains 512 points), should be the input to the network.
 However, Rudra tells me that we should shuffle the points, and in each batch (i.e. batch size 8),
 will get random points from different image (i.e. one from "2", another from "4")
 
 Explanation on it should be: the kernel size (2b * 2b), shared across whole points. It doesn't matter which label it is
 
 Also, the channel of input feature should be 1
 Therefore, I do some change on the shape of the input tensor:
 > datagen.py line 110: 
 
 ```python
reshape (train_size, num_points, 2 * args.bandwidth, 2 * args.bandwidth) --> (train_size * num_points, 1, 2 * args.bandwidth, 2 * args.bandwidth)
```
 > datagen.py line 140
 ```python
reshape (test_size, num_points, 2 * args.bandwidth, 2 * args.bandwidth) --> (test_size * num_points, 1, 2 * args.bandwidth, 2 * args.bandwidth)
```