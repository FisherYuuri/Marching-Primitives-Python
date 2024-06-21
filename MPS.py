# by yuuri 2024.3


import numpy as np
from skimage.measure import label, regionprops
from scipy.optimize import least_squares

def parse_input_args(voxel_grid, **kwargs):

    defaults = {
        'verbose': True,
        'padding_size': int(np.ceil(12 * voxel_grid['truncation'] / voxel_grid['interval'])),
        'min_area': int(np.ceil(voxel_grid['size'][0] / 20)),
        'max_division': 50,
        'scaleInitRatio': 0.1,
        'nanRange': 0.5 * voxel_grid['interval'],
        'w': 0.99,
        'tolerance': 1e-6,
        'relative_tolerance': 1e-4,
        'switch_tolerance': 1e-1,
        'maxSwitch': 2,
        'iter_min': 2,
        'maxOptiIter': 3,
        'maxIter': 15,
        'activeMultiplier': 3
    }


    for key, value in kwargs.items():
        if key in defaults:
            defaults[key] = value

    return defaults

def idx3d_flatten(indices, grid):
    return indices[0, :] + grid['size'][0] * (indices[1, :] - 1) + grid['size'][0] * grid['size'][1] * (indices[2, :] - 1)

def idx2Coordinate(indices, grid):
    idx_floor = np.floor(indices).astype(int)
    idx_floor[idx_floor == 0] = 1

    x = grid['x'][idx_floor[0, :] - 1] + (indices[0, :] - idx_floor[0, :]) * grid['interval']
    y = grid['y'][idx_floor[1, :] - 1] + (indices[1, :] - idx_floor[1, :]) * grid['interval']
    z = grid['z'][idx_floor[2, :] - 1] + (indices[2, :] - idx_floor[2, :]) * grid['interval']

    return np.vstack([x, y, z])

def eul2rotm(euler):
    # from euler angles to rotation matrix (ZYX_intrinsic)

    RotZ = np.array(
        [[np.cos(euler[0]), -np.sin(euler[0]), 0.0],
         [np.sin(euler[0]), np.cos(euler[0]), 0.0],
         [0.0, 0.0, 1.0]]
    )

    RotY = np.array(
        [[np.cos(euler[1]), 0.0, np.sin(euler[1])],
         [0.0, 1.0, 0.0],
         [-np.sin(euler[1]), 0.0, np.cos(euler[1])]]
    )

    RotX = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(euler[2]), -np.sin(euler[2])],
         [0.0, np.sin(euler[2]), np.cos(euler[2])]]
    )

    return RotZ @ RotY @ RotX

def rotm2eul(R):

    s = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = s < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], s)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], s)
        z = 0

    return np.array([z, y, x])

def difference_sqsdf(params, sdf, points, truncation, weight):
    R = eul2rotm(params[5:8])
    t = params[8:11]
    X = R.T @ points - R.T @ t[:, np.newaxis]

    r0 = np.linalg.norm(X, axis=0)

    scale = ((((X[0, :] / params[2]) ** 2) ** (1 / params[1]) + 
              ((X[1, :] / params[3]) ** 2) ** (1 / params[1])) ** (params[1] / params[0]) + 
             ((X[2, :] / params[4]) ** 2) ** (1 / params[0])) ** (-params[0] / 2)

    sdf_para = r0 * (1 - scale)

    if truncation != 0:
        sdf_para = np.clip(sdf_para, -truncation, truncation)

    dist = (sdf_para - sdf) * np.sqrt(weight)
    
    return dist

def cost_switched(params, sdf, points, truncation, weight):
    value = np.zeros(params.shape[0])

    for i in range(params.shape[0]):
        diff = difference_sqsdf(params[i, :], sdf, points, truncation, weight)
        value[i] = np.sum(diff ** 2)

    return value

def sdf_superquadric(params, points, truncation):
    R = eul2rotm(params[5:8])
    t = params[8:11]
    X = R.T @ points - R.T @ t[:, np.newaxis]

    r0 = np.linalg.norm(X, axis=0)
    scale = ((((X[0, :] / params[2]) ** 2) ** (1 / params[1]) + 
              ((X[1, :] / params[3]) ** 2) ** (1 / params[1])) ** (params[1] / params[0]) + 
             ((X[2, :] / params[4]) ** 2) ** (1 / params[0])) ** (-params[0] / 2)

    sdf = r0 * (1 - scale)

    if truncation != 0:
        sdf = np.clip(sdf, -truncation, truncation)

    return sdf

def inlier_weight(sdf_active, active_idx, sdf_current, sigma2, w, truncation):
    in_idx = sdf_active < 0.0 * truncation
    sdf_current = sdf_current[active_idx]

    const = w / ((1 - w) * (2 * np.pi * sigma2) ** (-1 / 2) * 1 * truncation)
    dist_current = np.clip(sdf_current[in_idx], -truncation, truncation) - sdf_active[in_idx]

    weight = np.ones(sdf_active.shape)
    p = np.exp(-0.5 / sigma2 * dist_current ** 2)
    p = p / (const + p)
    weight[in_idx] = p

    return weight

def fit_superquadric_tsdf(sdf, x_init, truncation, points, roi_idx, bounding_points, para):
    valid = np.zeros(6)
    t_lb = bounding_points[:, 0]
    t_ub = bounding_points[:, 7]

    lb = np.array([0.0, 0.0, truncation, truncation, truncation, -2 * np.pi, -2 * np.pi, -2 * np.pi] + t_lb.tolist())
    ub = np.array([2, 2, 1, 1, 1, 2 * np.pi, 2 * np.pi, 2 * np.pi] + t_ub.tolist())

    x = np.array(x_init.copy())
    cost = 0
    switched = 0
    nan_idx = ~np.isnan(sdf)
    sigma2 = np.exp(truncation) ** 2

    for iter in range(para['maxIter']):
        # print(iter)
        Rot = eul2rotm(x[5:8])
        check_points = np.array([
            x[8:11] - Rot[:, 0].T * x[2],
            x[8:11] + Rot[:, 0].T * x[2],
            x[8:11] - Rot[:, 1].T * x[3],
            x[8:11] + Rot[:, 1].T * x[3],
            x[8:11] - Rot[:, 2].T * x[4],
            x[8:11] + Rot[:, 2].T * x[4]]
        )
        valid[:3] = np.min(check_points, axis=0) >= t_lb - truncation
        valid[3:6] = np.max(check_points, axis=0) <= t_ub + truncation

        if not np.all(valid):
            break

        sdf_current = sdf_superquadric(x, points, 0)
        active_idx = (sdf_current < para['activeMultiplier'] * truncation) & \
                     (sdf_current > -para['activeMultiplier'] * truncation) & \
                     nan_idx

        points_active = points[:, active_idx]
        sdf_active = sdf[active_idx]

        weight = inlier_weight(sdf_active, active_idx, sdf_current, sigma2, para['w'], truncation)

        Rot = eul2rotm(x[5:8])
        bP_body = Rot.T @ (bounding_points - x[8:11][:, np.newaxis])
        scale_limit = np.mean(np.abs(bP_body), axis=1)
        ub[2:5] = scale_limit


        x = np.minimum(x,ub)
        x = np.maximum(x,lb)
        result = least_squares(difference_sqsdf, x, bounds=(lb, ub), method='trf',max_nfev=3,args=(sdf_active, points_active, truncation, weight))
        x_n = result.x
        cost_n = result.cost * 2

        x_n = np.array(x_n)
        sigma2_n = cost_n / np.sum(weight)
        cost_n /= len(sdf_active)

        relative_cost = abs(cost - cost_n) / cost_n

        if (cost_n < para['tolerance'] and iter > 1) or \
           (relative_cost < para['relative_tolerance'] and switched >= para['maxSwitch'] and iter > para['iter_min']):
            x = x_n
            break
        if relative_cost < para['switch_tolerance'] and iter != 1 and switched < para['maxSwitch']:
            switch_success = False
            axis_0 = eul2rotm(x[5:8])
            axis_1 = axis_0[:, np.array([1, 2, 0])]
            axis_2 = axis_0[:, np.array([2, 0, 1])]
            eul_1 = rotm2eul(axis_1)
            eul_2 = rotm2eul(axis_2)
            x_axis = np.array(
                [[x[1], x[0], x[3], x[4], x[2], eul_1[0], eul_1[1], eul_1[2], x[8], x[9], x[10]],
                 [x[1], x[0], x[4], x[2], x[3], eul_2[0], eul_2[1], eul_2[2], x[8], x[9], x[10]]]
            )

            scale_ratio = x[np.array([3, 4, 2])] / x[2:5]
            scale_idx = np.argwhere(np.logical_and(scale_ratio > 0.8, scale_ratio < 1.2))
            x_rot = np.zeros((scale_idx.shape[0], 11))

            for idx in range(scale_idx.shape[0]):
                if scale_idx[idx, 0] == 0:
                    eul_rot = rotm2eul(axis_0 @ eul2rotm(np.array([np.pi / 4, 0.0, 0.0])))
                    if x[1] <= 1:
                        x_rot[idx, :] = np.array(
                            [x[0], 2 - x[1],
                             ((1 - np.sqrt(2)) * x[1] + np.sqrt(2)) * min(x[2], x[3]),
                             ((1 - np.sqrt(2)) * x[1] + np.sqrt(2)) * min(x[2], x[3]),
                             x[4], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )
                    else:
                        x_rot[idx, :] = np.array(
                            [x[0], 2 - x[1],
                             ((np.sqrt(2) / 2 - 1) * x[1] + 2 - np.sqrt(2) / 2) * min(x[2], x[3]),
                             ((np.sqrt(2) / 2 - 1) * x[1] + 2 - np.sqrt(2) / 2) * min(x[2], x[3]),
                             x[4], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )

                elif scale_idx[idx, 0] == 1:
                    eul_rot = rotm2eul(axis_1 @ eul2rotm(np.array([np.pi / 4, 0.0, 0.0])))
                    if x[0] <= 1:
                        x_rot[idx, :] = np.array(
                            [x[1], 2 - x[0],
                             ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[3], x[4]),
                             ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[3], x[4]),
                             x[2], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )
                    else:
                        x_rot[idx, :] = np.array(
                            [x[1], 2 - x[0],
                             ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[3], x[4]),
                             ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[3], x[4]),
                             x[2], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )

                elif scale_idx[idx, 0] == 2:
                    eul_rot = rotm2eul(axis_2 @ eul2rotm(np.array([np.pi / 4, 0.0, 0.0])))
                    if x[0] <= 1:
                        x_rot[idx, :] = np.array(
                            [x[1], 2 - x[0],
                             ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[4], x[2]),
                             ((1 - np.sqrt(2)) * x[0] + np.sqrt(2)) * min(x[4], x[2]),
                             x[2], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )
                    else:
                        x_rot[idx, :] = np.array(
                            [x[1], 2 - x[0],
                             ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[4], x[2]),
                             ((np.sqrt(2) / 2 - 1) * x[0] + 2 - np.sqrt(2) / 2) * min(x[4], x[2]),
                             x[2], eul_rot[0], eul_rot[1], eul_rot[2],
                             x[8], x[9], x[10]]
                        )

            x_candidate = np.zeros((2 + x_rot.shape[0], 11))
            x_candidate[0: 2] = x_axis
            if scale_idx.shape[0] > 0:
                x_candidate[2: 2 + scale_idx.shape[0]] = x_rot

            cost_candidate = cost_switched(x_candidate, sdf_active, points_active, truncation, weight) 
            idx_nan = np.argwhere(
                np.logical_and(~np.isnan(cost_candidate), ~np.isinf(cost_candidate))
            ).reshape(1, -1)[0]

            cost_candidate = cost_candidate[idx_nan]
            idx = np.argsort(cost_candidate)

            for i_candidate in idx:
                Rot = eul2rotm(x_candidate[i_candidate, 5:8])
                bP_body = Rot.T @ (bounding_points - x_candidate[i_candidate, 8:11][:, np.newaxis])
                scale_limit = np.mean(np.abs(bP_body), axis=1)
                ub[2:5] = scale_limit

                x_candidate[i_candidate] = np.minimum(x_candidate[i_candidate], ub)
                x_candidate[i_candidate] = np.maximum(x_candidate[i_candidate], lb)

                result = least_squares(difference_sqsdf, x_candidate[i_candidate], bounds=(lb, ub),max_nfev=3,args = (sdf_active, points_active, truncation, weight))
                x_switch = result.x
                cost_switch = result.cost * 2

                if cost_switch / len(sdf_active) < min(cost_n, cost):
                    x = x_switch
                    cost = cost_switch / len(sdf_active)
                    sigma2 = cost_switch / np.sum(weight)
                    switch_success = True
                    break

            if not switch_success:
                cost = cost_n
                x = x_n
                sigma2 = sigma2_n
            switched += 1
        else:
            cost = cost_n
            sigma2 = sigma2_n
            x = x_n

    sdf_occ = sdf_superquadric(x, points, 0)

    occ = sdf_occ < para['nanRange']
    occ_idx = roi_idx[occ]

    occ_in = sdf_occ <= 0

    num_idx = np.zeros(3)
    num_idx[0] = np.sum(np.logical_or(sdf[occ_in] <= 0, np.isnan(sdf[occ_in])))
    num_idx[1] = np.sum(sdf[occ_in] > 0)
    num_idx[2] = np.sum(sdf[occ_in] <= 0)

    Rot = eul2rotm(x[5:8])
    check_points = np.array([
        x[8:11] - Rot[:, 0].T * x[2],
        x[8:11] + Rot[:, 0].T * x[2],
        x[8:11] - Rot[:, 1].T * x[3],
        x[8:11] + Rot[:, 1].T * x[3],
        x[8:11] - Rot[:, 2].T * x[4],
        x[8:11] + Rot[:, 2].T * x[4]]
    )

    valid[:3] = np.min(check_points, axis=0) >= t_lb - truncation
    valid[3:6] = np.max(check_points, axis=0) <= t_ub + truncation

    return x, occ_idx, valid, num_idx


def mps(sdf, voxel_grid, **kargs):
    parameters = parse_input_args(voxel_grid, **kargs)
    num_division = 1
    x = []
    dratio = 3/5
    conn_ratio = [1, dratio, dratio**2, dratio**3, dratio**4,
                  dratio**5, dratio**6, dratio**7, dratio**8]

    conn_pointer = 1
    num_region = 1

    while num_division < parameters['max_division']:
        if conn_pointer != 1 and num_region != 0:
            conn_pointer = 1

        conn_threshold = conn_ratio[conn_pointer - 1] * np.nanmin(sdf)
        if conn_threshold > -voxel_grid['truncation'] * 0.3:
            break

        sdf3d_region = sdf.reshape((voxel_grid['size'][0], voxel_grid['size'][1], voxel_grid['size'][2]),order='F')
        
        labeled_region = label(sdf3d_region <= conn_threshold)
        regions = regionprops(labeled_region)

        regions = [region for region in regions if region.area >= parameters['min_area']]
        num_region = len(regions)

        if parameters['verbose']:
            print(f"Number of regions: {num_region}")

        if num_region == 0:
            if conn_pointer < len(conn_ratio):
                conn_pointer += 1
            else:
                break

        num_region = len(regions)
        x_temp = np.zeros((num_region, 11))
        del_idx = np.zeros(num_region, dtype=int)
        occ_idx_in = []
        num_idx = np.zeros((num_region, 3))
        for i in range(num_region):
            occ_idx = []
            # 获取并调整边界框
            bbox = regions[i].bbox
            nbbox = np.zeros(len(bbox)) 
            nbbox[0] = bbox[1] + 1
            nbbox[1] = bbox[0] + 1
            nbbox[2] = bbox[2] + 1
            nbbox[3] = bbox[4] - bbox[1]
            nbbox[4] = bbox[3] - bbox[0]
            nbbox[5] = bbox[5] - bbox[2]

            nbbox[3:] = np.minimum(nbbox[:3] + nbbox[3:] + parameters['padding_size'],
                                [voxel_grid['size'][1], voxel_grid['size'][0], voxel_grid['size'][2]])
            nbbox[:3] = np.maximum(nbbox[:3] - parameters['padding_size'], 1) 
            regions[i].nbbox = nbbox
            # 计算激活的体素索引
            idx_x, idx_y, idx_z = np.mgrid[nbbox[1]:nbbox[4]+1, nbbox[0]:nbbox[3]+1, nbbox[2]:nbbox[5]+1]

            indices = np.vstack([idx_x.T.ravel(), idx_y.T.ravel(), idx_z.T.ravel()])
            roi_idx = idx3d_flatten(indices, voxel_grid) 
            regions[i].idx = roi_idx

            # 计算边界点坐标
            bounding_points = idx2Coordinate(np.array([
                [nbbox[1], nbbox[1], nbbox[4], nbbox[4], nbbox[1], nbbox[1], nbbox[4], nbbox[4]],
                [nbbox[0], nbbox[0], nbbox[0], nbbox[0], nbbox[3], nbbox[3], nbbox[3], nbbox[3]],
                [nbbox[2], nbbox[5], nbbox[2], nbbox[5], nbbox[2], nbbox[5], nbbox[2], nbbox[5]]
            ]), voxel_grid)
            regions[i].bounding_points = bounding_points
            # 确定中心点并向下取整
            centroid = np.maximum(np.floor(regions[i].centroid), 1).astype(int)
            # 将中心点坐标转换为一维索引
            centroid_flatten = idx3d_flatten(np.array([[centroid[0] + 1, centroid[1] + 1, centroid[2] + 1]]).T, voxel_grid)
            # 获取区域的三维坐标
            coords = regions[i].coords.T
            # 将三维坐标转换为一维线性索引
            pixel_idx_list = idx3d_flatten(np.array([coords[0]+1, coords[1]+1, coords[2]+1]), voxel_grid)
            pixel_idx_list = np.sort(pixel_idx_list)
            regions[i].pixel_idx_list = pixel_idx_list
            print('pixel_idx_list:', pixel_idx_list.shape, pixel_idx_list[0])
            # 检查中心点一维索引是否在像素索引列表中
            if centroid_flatten[0] in pixel_idx_list:
                centroid = voxel_grid['points'][:,centroid_flatten - 1]
                regions[i].ncentroid = centroid
            else:
                def dsearchn(x, y):
                    IDX = []
                    for line in range(y.shape[0]):
                        distances = np.sqrt(np.sum(np.power(x - y[line, :], 2), axis=1))
                        found_min_dist_ind = (np.min(distances, axis=0) == distances)
                        length = found_min_dist_ind.shape[0]
                        IDX.append(np.array(range(length))[found_min_dist_ind][0])
                    return np.array(IDX)
                pixel_coords = voxel_grid['points'][:,pixel_idx_list-1].T.round(6)
                query_point = voxel_grid['points'][:,centroid_flatten-1].T.round(6)
                k = dsearchn(pixel_coords,query_point)
                # 更新区域中心点坐标
                regions[i].ncentroid = voxel_grid['points'][:,pixel_idx_list[k]-1]


            valid = np.zeros(6, dtype=int)
            while not np.all(valid):
                scale_init = parameters['scaleInitRatio'] * (regions[i]['nbbox'][3:] - regions[i]['nbbox'][:3]) * voxel_grid['interval']


                # 初始化超四面体参数
                ctmp = np.squeeze(regions[i].ncentroid.T)
                x_init = np.hstack(([1, 1], scale_init[[1, 0, 2]], [0, 0, 0], ctmp))

                # 为每个区域找到最佳的超四面体表示
                x_temp[i, :], occ_idx, valid, num_idx[i, :] = fit_superquadric_tsdf(
                    sdf[regions[i].idx.astype(int)-1],
                    x_init,
                    voxel_grid['truncation'],
                    voxel_grid['points'][:, regions[i].idx.astype(int)-1],
                    regions[i]['idx'],
                    regions[i]['bounding_points'],
                    parameters
                )

                if not np.all(valid):
                    extense = np.logical_not(valid)

                    tbbox = nbbox[:3]
                    extense[[0, 1]] = extense[[1, 0]]
                    extense[[3, 4]] = extense[[4, 3]]
                    if any([any(tbbox[extense[:3]] == 1),
                            nbbox[3] == voxel_grid['size'][1] and extense[3],
                            nbbox[4] == voxel_grid['size'][0] and extense[4],
                            nbbox[5] == voxel_grid['size'][2] and extense[5]]):
                        break
                    idx_extend = np.logical_not(valid) * parameters['padding_size']
                    nbbox[3:] = np.minimum(nbbox[3:] + idx_extend[[4, 3, 5]], [voxel_grid['size'][1],voxel_grid['size'][0],voxel_grid['size'][2]])
                    nbbox[:3] = np.maximum(nbbox[:3] - idx_extend[[1, 0, 2]], 1)
                    idx_x,idx_y,idx_z = np.mgrid[nbbox[1]:nbbox[4]+1,nbbox[0]:nbbox[3]+1,nbbox[2]:nbbox[5]+1]
                    indices = np.vstack([idx_x.T.ravel(), idx_y.T.ravel(), idx_z.T.ravel()])
                    regions[i].idx = idx3d_flatten(indices,voxel_grid)
                    regions[i].bounding_points = idx2Coordinate(np.array([
                                                    [nbbox[1], nbbox[1], nbbox[4], nbbox[4], nbbox[1], nbbox[1], nbbox[4], nbbox[4]],
                                                    [nbbox[0], nbbox[0], nbbox[0], nbbox[0], nbbox[3], nbbox[3], nbbox[3], nbbox[3]],
                                                    [nbbox[2], nbbox[5], nbbox[2], nbbox[5], nbbox[2], nbbox[5], nbbox[2], nbbox[5]]
                                                ]), voxel_grid)
            occ_idx_in.append(occ_idx[sdf[occ_idx.astype(int)-1] <= 0])

        for i, region in enumerate(regions):
            if num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1]) > 0.3 or num_idx[i, 0] < parameters['min_area'] or num_idx[i, 2] <= 1:
                del_idx[i] = 1
                sdf[region['pixel_idx_list']-1] = np.nan
                if parameters['verbose'] == 1:
                    print('region', i, '/', num_region, 'outPercentage:',
                          num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1]), 'inNumber:', num_idx[i, 2],
                          '...REJECTED')
            else:
                sdf[occ_idx_in[i].astype(int)-1] = np.nan
                if parameters['verbose'] == 1:
                    print('region', i, '/', num_region, 'outPercentage:',
                          num_idx[i, 1] / (num_idx[i, 0] + num_idx[i, 1]), 'inNumber:', num_idx[i, 2],
                          '...ACCEPTED')

        x_temp = x_temp[del_idx == 0]
        x.extend(x_temp)
        num_division += 1

    return x
