# by yuuri 2024.3

import os
import numpy as np
import time
from MPS import mps
from superquadrics import superquadric
from mayavi import mlab

if __name__ == "__main__":

    file_path = r"data/chair6_normalized.csv"
    if not file_path:
        raise ValueError("No file selected.")

    # 读取CSV文件
    sdf = np.genfromtxt(file_path, delimiter=',').T
    voxelGrid = {}

    # 设置体素网格参数
    voxelGrid['size'] = np.ones(3, dtype=int) * int(sdf[0])
    voxelGrid['range'] = sdf[1:7]
    sdf = sdf[7:]
    # 创建线性空间
    voxelGrid['x'] = np.linspace(voxelGrid['range'][0], voxelGrid['range'][1], int(voxelGrid['size'][0]))
    voxelGrid['y'] = np.linspace(voxelGrid['range'][2], voxelGrid['range'][3], int(voxelGrid['size'][1]))
    voxelGrid['z'] = np.linspace(voxelGrid['range'][4], voxelGrid['range'][5], int(voxelGrid['size'][2]))

    # 创建网格
    x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'], indexing='ij')
    points = np.stack((x,y,z),axis=3)
    voxelGrid['points'] = points.reshape((-1,3),order='F').T 

    # 计算间隔和截断
    voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
    voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
    voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
    voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

    # 截断SDF
    sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])
    print('sdf.shape: ', sdf.shape)
    print('voxelGrid[\'points\'].shape: ', voxelGrid['points'].shape)

    # %% marching-primitives
    # Python中的时间测量
    start_time = time.time()
    x = mps(sdf, voxelGrid) 
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    for quadric in x:
        sq = superquadric(quadric[0 : 2], quadric[2 : 5], quadric[5 : 8], quadric[8 : 11])
        sq.showSuperquadric(arclength=0.2)
    mlab.show()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
