import cv2
import numpy as np
import glob

def calibration_photo(photo_path,x_nums,y_nums):
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)
    world_point[:, :2] = 26.3 * np.mgrid[:x_nums, :y_nums].T.reshape(-1,2)
    print(world_point)
    world_position = []
    image_position = []

    '''
    下面就是查找图片中角点的像素坐标存入image_position了
    '''
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 获取所有标定图
    images = glob.glob(photo_path + '\\*.jpg')
    # print(images)
    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 查找角点
        ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)

        if ok:
            # 把每一幅图像的世界坐标放到world_position中
            world_position.append(world_point)
            # 获取更精确的角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 把获取的角点坐标放到image_position中
            image_position.append(exact_corners)
            # 可视化角点
            image = cv2.drawChessboardCorners(image, (x_nums, y_nums), exact_corners, ok)
            cv2.imshow('image_corner', cv2.resize(image,(0,0),fx=0.2,fy=0.2))
            cv2.waitKey(1)

    """
    点对应好了，开始计算内参，畸变矩阵，外参
    """
    print("正在计算内外参数")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_position, image_position, gray.shape[::-1], None, None)
    np.savez('checkerboard_mate30_1_720', mtx=mtx, dist=dist)

    print('内参是：\n', mtx, '\n畸变参数是：\n', dist,
          '\n外参：旋转向量（要得到矩阵还要进行罗德里格斯变换，下章讲）是：\n', rvecs, '\n外参：平移矩阵是：\n', tvecs)

    # 计算偏差
    mean_error = 0
    for i in range(len(world_position)):
        image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_position[i], image_position2, cv2.NORM_L2) / len(image_position2)
        mean_error += error
    print("total error: ", mean_error / len(image_position))


def main():
    # 标定图像保存路径
    photo_path = "."
    x_nums = 9  # x方向上的角点个数
    y_nums = 6
    calibration_photo(photo_path,x_nums,y_nums)


if __name__ == '__main__':
    main()