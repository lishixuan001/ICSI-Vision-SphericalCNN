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
```
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
```
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

![img](http://latex.codecogs.com/svg.latex?x%3Dradius%2Atorch.sin%28%5Ctheta%29%2Atorch.cos%28%5Cphi%29)

![img](http://latex.codecogs.com/svg.latex?y%3Dradius%2Atorch.sin%28%5Ctheta%29%2Atorch.sin%28%5Cphi%29)

![img](http://latex.codecogs.com/svg.latex?z%3Dradius%2Atorch.cos%28theta%29)

Then, for x and y, the first row is same for each grid in (2b * 2b) grids, since the value of theta is always 0,
and sin(0) = 0

As for z, since the cos(0)=1, the first row of each grid is always 1

So basically, the first 2b points share the same (x, y, z), and therefore when calculating the distance with other points,
they should remain the same value.

### explanation on training batch
The shape of train dataset is:
```python
# datagen.py line 105 & 135
reshape(train_size, num_points, 1, 2 * b, 2 * b)
```
When it is loaded into dataloader, batch size comes from the train_size
 (for a whole train size, should be 60000)

Then, when it is iterate from the dataloader, do the reshape:
```python
# train.py line 166
for tl in train_loader:
    images = tl['data'].reshape((-1, 1, 2 * b, 2 * b))
```
As shown, the data input to the S2CNN should be 
(batch_size * num_points, 1, 2 * b, 2 * b), so that the channel into 
to S2CNN is 1 (not num_points), which means, each individual 
points is put into the S2CNN for forwarding.

Then, after the S2CNN part, it should go into the standard 
CNN as a whole image (512 points). Do the following transformation:
```python
    # train.py line 56
    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = F.relu(conv2) # (B * 512, f2, 2 * b_l2, 2 * b_l2, 2 * b_l2)
        in_data = relu2[:, :, 0, 0, 0] # -> (B * 512, f2)
        in_reshape = in_data.reshape(self.batch_size, 1, self.num_points * self.f2)  # -> (B, 1, 512 * f2)
        conv3 = self.conv3(in_reshape)  # -> (B, 10, L'), L' = 512 * f2 - kernel_size + 1
        relu3 = F.relu(conv3)
        bn3 = self.bn3(relu3)
        bn3_reshape = bn3.reshape((self.batch_size, -1))  # -> (B, 10 * L')
        output = self.out_layer(bn3_reshape) # -> (B, 10)
    
    return output
```

and then we can feed them into the criterion