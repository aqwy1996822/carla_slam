import cv2
import numpy as np
import glob
import math


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


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, (corner), tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


# 标定图像
def calibration_photo(photo_path, x_nums=9, y_nums=6, win_size=(11, 11), cell_size=26.3, axis_rate=0.5):
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)
    world_point[:, :2] = cell_size * np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)
    # print('world point:', world_point)

    axis = 10 * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image = cv2.imread(photo_path)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
    if ok:
        exact_corners = cv2.cornerSubPix(gray, corners, win_size, (-1, -1), criteria)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
        rotation_m, _ = cv2.Rodrigues(rvec)
        print(rotation_m)
        print('欧拉角是：\n', rvec * 180 / math.pi)
        print('平移矩阵是:\n', tvec)
        rotation_t = np.hstack([rotation_m, tvec])
        rotation_t_Homogeneous_matrix = np.vstack([rotation_t, np.array([[0, 0, 0, 1]])])
        print("平移与旋转矩阵", rotation_t_Homogeneous_matrix)
        imgpts, djac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        # 可视化角点
        img = cv2.drawChessboardCorners(image, (x_nums, y_nums), exact_corners, ok)
        # img = draw(img, corners, imgpts)
        # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        # cv2.imshow('img',  cv2.resize(img,(0,0),fx=0.8,fy=0.8))
        photo_save_path = photo_path.replace(".jpg", "_rvec=" + str(rvec * 180 / math.pi) + "_tvec=" + str(
            tvec) + ".jpg").replace("checkerboard", "output").replace("input", "output").replace("\n", "").replace(
            "] [", "_").replace(" ", "").replace("[", "").replace("]", "")
        # print(photo_save_path)
        # cv2.imwrite(photo_save_path,img)
        # cv2.waitKey(100)
        return rotation_t_Homogeneous_matrix
    else:
        print("失败")
        return None


if __name__ == '__main__':
    with np.load('checkerboard_mate30_1_720.npz') as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
        # print(mtx, '\n', dist)
    img1 = cv2.imread("./00001.jpg")
    img2 = cv2.imread("./00066.jpg")
    T_c1 = calibration_photo("00001.jpg", x_nums=9, y_nums=6, win_size=(11, 11), cell_size=0.0263, axis_rate=0.5)
    T_c2 = calibration_photo("00066.jpg", x_nums=9, y_nums=6, win_size=(11, 11), cell_size=0.0263, axis_rate=0.5)

    R_c1 = T_c1[:3, :3]
    R_c2 = T_c2[:3, :3]
    t_c1 = T_c1[:3, 3].reshape((3, 1))
    t_c2 = T_c2[:3, 3].reshape((3, 1))

    T_c1c2 = T_c1 @ inv_T(T_c2)
    T_c2c1 = T_c2 @ inv_T(T_c1)

    R_c1c2 = T_c1c2[:3, :3]
    t_c1c2 = T_c1c2[:3, 3].reshape((3, 1))
    print("R_c1c2", R_c1c2)
    print("t_c1c2", t_c1c2)

    R_c2c1 = T_c2c1[:3, :3]
    t_c2c1 = T_c2c1[:3, 3].reshape((3, 1))
    print("R_c2c1", R_c2c1)
    print("t_c2c1", t_c2c1)

    print("mtx", mtx)
    n = R_c1@np.array([[0, 0, -1]]).T

    d1 = T_c1[2, 3]


    H3 = mtx @ (R_c2c1 - t_c2c1 @ n.T / d1) @ np.linalg.inv(mtx)

    res3 = cv2.warpPerspective(img1, H3, (1280, 720))

    add_img3 = cv2.add(img2, res3)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("res3", res3)
    cv2.imshow("add_img3", add_img3)
    cv2.waitKey()
