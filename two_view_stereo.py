import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""

    H_Ki = np.dot(K_i_corr, np.dot(rect_R_i, np.linalg.inv(K_i)))
    H_Kj = np.dot(K_j_corr, np.dot(rect_R_j, np.linalg.inv(K_j)))


    rgb_i_rect = cv2.warpPerspective(rgb_i, H_Ki, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_Kj, (w_max, h_max))
        
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w
    Returns
    -------
    [3,3], [3,1], float
        p_i = i_R_j @ p_j + i_T_j, B is the baseline
    """



    i_Trans_w = np.zeros((4, 4))
    i_Trans_w[:3, :3] = i_R_w
    i_Trans_w[:3, -1] = np.ravel(i_T_w)
    i_Trans_w[-1, -1] = 1

    j_Trans_w = np.zeros((4, 4))
    j_Trans_w[:3, :3] = j_R_w
    j_Trans_w[:3, -1] = np.ravel(j_T_w)
    j_Trans_w[-1, -1] = 1    

    i_Trans_j = i_Trans_w @ np.linalg.inv(j_Trans_w)
    # i_Trans_j = i_Trans_w @ j_Trans_w.T




    i_R_j = i_Trans_j[:3, :3]
    i_T_j = i_Trans_j[:3, -1].reshape((3, 1))


    B = np.linalg.norm(i_T_j)

    return i_R_j, i_T_j, B


def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    [3,3]
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)


    r2 = e_i
    r2 = r2/np.linalg.norm(r2) 
    # print('r2 norm: ', np.linalg.norm(r2))
    r1 = np.cross(r2, np.array([0, 0, 1]))
    r1 = r1/np.linalg.norm(r1)
    # print('r1 norm: ', np.linalg.norm(r1))
    r3 = np.cross(r1, r2)
    r3 = r3/np.linalg.norm(r3)
    # print('r3 norm: ', np.linalg.norm(r3))
    rect_R_i = np.row_stack((r1, r2, r3))
    # print('rect_R_i norm: ', np.linalg.norm(rect_R_i))


    return rect_R_i


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src_height, src_width, src_depth = np.shape(src)
    dst_height, dst_width, dst_depth = np.shape(dst)

    src.reshape((src_height, src_width, 3))
    dst.reshape((dst_height, src_width, 3))
    ssd = np.ones((src_height, dst_height))

    for i in range(src_height):

        for j in range(dst_height):

            channel_diff = src[i] - dst[j]
            channel_errors = np.sum(np.square(channel_diff), axis=1)
            ssd[i, j] = np.sum(channel_errors)
    
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""

    src_height, src_width, src_depth = np.shape(src)
    dst_height, dst_width, dst_depth = np.shape(dst)

    src.reshape((src_height, src_width, 3))
    dst.reshape((dst_height, src_width, 3))
    sad = np.ones((src_height, dst_height))

    for i in range(src_height):
        
        for j in range(dst_height):

            channel_diff = src[i] - dst[j]
            channel_errors = np.sum(np.abs(channel_diff), axis=1)
            sad[i, j] = np.sum(channel_errors)

    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""

    src_height, src_width, src_depth = np.shape(src)
    dst_height, dst_width, dst_depth = np.shape(dst)

    zncc = np.zeros((src_height, dst_height))
    EPS = 1e-8

    for spatch in range(src_height):
        for dpatch in range(dst_height):
            for channel in range(3):
                patchsrc, patchdst = src[spatch, :, channel], dst[dpatch, :, channel]
                patchsrcmean, patchdstmean = np.mean(patchsrc), np.mean(patchdst)
                patchsrcdiff = patchsrc - patchsrcmean
                patchdstdiff = patchdst - patchdstmean
                covariance = np.sum((patchsrcdiff * patchdstdiff))
                patchsrcstd = EPS + np.std(patchsrc)
                patchdststd = EPS + np.std(patchdst)
                zncc[spatch, dpatch] = zncc[spatch, dpatch]  + covariance / (patchdststd * patchsrcstd)

    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    patch_buffer = np.zeros((image.shape[0], image.shape[1], k_size ** 2, 3))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            i_start = max(0, i - k_size//2)
            i_end = min(image.shape[0], i + k_size//2 + 1)
            j_start = max(0, j - k_size//2)
            j_end = min(image.shape[1], j + k_size//2 + 1)

            temp = image[i_start:i_end, j_start:j_end]

            temp = np.pad(temp, ((max(0, k_size//2 - i), max(0, i + k_size//2 + 1 - image.shape[0])), (max(0, k_size//2 - j), max(0, j + k_size//2 + 1 - image.shape[1])), (0, 0)), mode='constant')


            patch_buffer[i, j] = temp.reshape(-1, 3)


    
    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel,  img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    
    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    # DO NOT DIRECTLY USE: patch_buffer = image2patch(image, k_size), as it may cause errors in the autograder

    """Student Code Starts"""



    rgb_i_normalized = rgb_i.astype(float) / 255.0
    rgb_j_normalized = rgb_j.astype(float) / 255.0

    H, W, _ = rgb_i.shape

    index_i = np.arange(H)
    index_j = np.arange(H)
    disparity_indices = index_i[:, None] - index_j[None, :] + d0  
    valid_disparity = disparity_indices > 0.0

    img2patch_i = img2patch_func(rgb_i_normalized, k_size) 
    img2patch_j = img2patch_func(rgb_j_normalized, k_size)  

    disp_map = np.zeros((H, W), dtype=np.float64)
    lr_consistency_mask = np.zeros((H, W), dtype=np.float64)


    for i in range(W):
        # print(i, end = '\r')
        kernel = kernel_func(img2patch_i[:, i], img2patch_j[:, i])

        mask_invalid = 1.0 + kernel.max() 
        kernel[~valid_disparity] = mask_invalid

        least= np.argmin(kernel, axis = 1)

        disp_map[:, i] = disparity_indices[index_i, least]

        consistency_mask = np.argmin(kernel[:, least], axis=0) == index_i
        lr_consistency_mask[:, i] = consistency_mask

    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""

    dep_map = K[1, 1] * B / disp_map

    H, W = disp_map.shape
    columns = np.arange(W)
    rows = np.arange(H)

    col_indices, row_indices = np.meshgrid(*[indices for indices in (columns, rows)])

    xyz_cam = np.dstack(((col_indices - K[0, -1]) * dep_map / K[0, 0], 
                     (row_indices - K[1, -1]) * dep_map / K[1, 1],
                     dep_map))

    
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""

    pcl_world = ((c_R_w.T @ pcl_cam.T) - (c_R_w.T @ c_T_w)).T

    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
