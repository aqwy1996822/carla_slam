import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def inv_T(T):
    R = np.array([[T[0][0], T[0][1], T[0][2]],
                  [T[1][0], T[1][1], T[1][2]],
                  [T[2][0], T[2][1], T[2][2]]])
    t = np.array([[T[0][3]], [T[1][3]], [T[2][3]]])
    RT = R.T
    t_ = -RT @ t
    T_ = np.array([[RT[0][0], RT[0][1], RT[0][2], t_[0][0]],
                   [RT[1][0], RT[1][1], RT[1][2], t_[1][0]],
                   [RT[2][0], RT[2][1], RT[2][2], t_[2][0]],
                   [0, 0, 0, 1]])
    return T_


def Rm_t2T(R_m, t):
    T = []
    T.append(list(np.append(R_m[0], t[0])))
    T.append(list(np.append(R_m[1], t[1])))
    T.append(list(np.append(R_m[2], t[2])))
    T.append(list(np.array([0.0, 0.0, 0.0, 1.0])))
    T = np.array(T)
    return T

def get_camera_top_T():
    T0 = Rm_t2T(eulerAnglesToRotationMatrix([0, 0, 0]), [0, 0, 10])
    T1 = Rm_t2T(eulerAnglesToRotationMatrix([0, -90, 0]), [0, 0, 0])
    T_fix1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    T_fix2 = np.array([[0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
    T = T_fix2 @ T1 @ T_fix1 @ T0
    return T

def get_camera_left_T():
    T0 = Rm_t2T(eulerAnglesToRotationMatrix([0, 0, 0]), [0, -1, 2.4])
    T1 = Rm_t2T(eulerAnglesToRotationMatrix([0, -30, 90]), [0, 0, 0])
    T_fix1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    T_fix2 = np.array([[0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
    T = T_fix2 @ T1 @ T_fix1 @ T0
    return T

def get_camera_right_T():
    T0 = Rm_t2T(eulerAnglesToRotationMatrix([0, 0, 0]), [0, -1, 2.4])
    T1 = Rm_t2T(eulerAnglesToRotationMatrix([0, -30, -90]), [0, 0, 0])
    T_fix1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    T_fix2 = np.array([[0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
    T = T_fix2 @ T1 @ T_fix1 @ T0
    return T

def get_camera_mid_T():
    T0 = Rm_t2T(eulerAnglesToRotationMatrix([0, 0, 0]), [1.5, 0, 2.4])
    T1 = Rm_t2T(eulerAnglesToRotationMatrix([0, -30, 0]), [0, 0, 0])
    T_fix1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
    T_fix2 = np.array([[0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
    T = T_fix2 @ T1 @ T_fix1 @ T0
    return T

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(math.pi / 180 * theta[0]), -math.sin(math.pi / 180 * theta[0])],
                    [0, math.sin(math.pi / 180 * theta[0]), math.cos(math.pi / 180 * theta[0])]
                    ])

    R_y = np.array([[math.cos(math.pi / 180 * theta[1]), 0, math.sin(math.pi / 180 * theta[1])],
                    [0, 1, 0],
                    [-math.sin(math.pi / 180 * theta[1]), 0, math.cos(math.pi / 180 * theta[1])]
                    ])

    R_z = np.array([[math.cos(math.pi / 180 * theta[2]), -math.sin(math.pi / 180 * theta[2]), 0],
                    [math.sin(math.pi / 180 * theta[2]), math.cos(math.pi / 180 * theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def get_mid_H():
    # 原始相机参数
    width = 800
    height = 600
    cx = width / 2
    cy = height / 2
    fov = 90
    fx = width / (2.0 * math.tan(fov * math.pi / 360.0))
    fy = fx
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    T_mid = get_camera_mid_T()
    T_top = get_camera_top_T()
    T_top_mid = T_top@inv_T(T_mid)
    R_top_mid = T_top_mid[:3, :3]
    t_top_mid = T_top_mid[:3, 3].reshape((3,1))

    d = 2.4
    n= T_mid[:3,:3]@np.array([[0,0,-1]]).T

    H = K @ (R_top_mid - t_top_mid @ n.T / d) @ K_inv
    return H

def get_left_H():
    # 原始相机参数
    width = 800
    height = 600
    cx = width / 2
    cy = height / 2
    fov = 90
    fx = width / (2.0 * math.tan(fov * math.pi / 360.0))
    fy = fx
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    T_mid = get_camera_left_T()
    T_top = get_camera_top_T()
    T_top_mid = T_top@inv_T(T_mid)
    R_top_mid = T_top_mid[:3, :3]
    t_top_mid = T_top_mid[:3, 3].reshape((3,1))

    d = 2.4
    n= T_mid[:3,:3]@np.array([[0,0,-1]]).T

    H = K @ (R_top_mid - t_top_mid @ n.T / d) @ K_inv
    return H

def get_right_H():
    # 原始相机参数
    width = 800
    height = 600
    cx = width / 2
    cy = height / 2
    fov = 90
    fx = width / (2.0 * math.tan(fov * math.pi / 360.0))
    fy = fx
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    T_mid = get_camera_right_T()
    T_top = get_camera_top_T()
    T_top_mid = T_top@inv_T(T_mid)
    R_top_mid = T_top_mid[:3, :3]
    t_top_mid = T_top_mid[:3, 3].reshape((3,1))

    d = 2.4
    n= T_mid[:3,:3]@np.array([[0,0,-1]]).T

    H = K @ (R_top_mid - t_top_mid @ n.T / d) @ K_inv
    return H

if __name__ == "__main__":
    H_t_m = get_mid_H()
    H_t_l = get_left_H()
    H_t_r = get_right_H()
    img_m = cv2.imread("_outM/M013006.png")
    img_l = cv2.imread("_outL/L013006.png")
    img_r = cv2.imread("_outR/R013006.png")
    img_t = cv2.imread("_outT/T013006.png")
    res_m = cv2.warpPerspective(img_m, H_t_m, (800, 600))
    res_l = cv2.warpPerspective(img_l, H_t_l, (800, 600))
    res_r = cv2.warpPerspective(img_r, H_t_r, (800, 600))

    # add_img = cv2.add(img_t, res_m)
    add_img = cv2.add(res_m, res_l)
    add_img = cv2.add(add_img, res_r)
    cv2.imshow("res_m", res_m)
    cv2.imshow("res_l", res_l)
    cv2.imshow("res_r", res_r)
    cv2.imshow("add", add_img)
    cv2.imshow("img_t", img_t)
    cv2.waitKey()
    # result1 = cv2.warpPerspective(img_m, H1, (800, 600))