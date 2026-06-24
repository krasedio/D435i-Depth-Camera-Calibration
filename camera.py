import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 初始化设置与参数定义
# 标定板参数：请根据你的实际情况修改
CHECKERBOARD = (8, 11)  # 棋盘格内部角点数目 (行, 列)
square_size = 30        # 单位：mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备世界坐标系中的物体点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
used_images = []

# 修改为你本地的图像路径
image_path = r'C:\Users\Admin\Desktop\d435ifigure\figure*_Color.png'
images = glob.glob(image_path)

if not images:
    print(f"错误: 未在路径 {image_path} 下找到图像，请检查路径。")
    exit()

print(f"开始处理 {len(images)} 张图像...")

# 图像处理与角点提取
gray_shape = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        used_images.append(fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        

cv2.destroyAllWindows()

# 执行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

print(f"\n标定完成！有效图像: {len(used_images)}")
print(f"总平均重投影误差: {ret:.4f} pixels")
print(f"内参矩阵:\n{mtx}")
print(f"畸变系数:\n{dist}")

# 计算误差数据用于绘图 (仿 Kalibr 逻辑)
all_errors_mag = []    # 误差模长
all_polar_angles = []  # 极角 (相对于图像中心)
all_res_x = []         # X轴误差
all_res_y = []         # Y轴误差
image_indices = []     # 图像索引

cx, cy = mtx[0, 2], mtx[1, 2]

for i in range(len(objpoints)):
    # 投影 3D 点到 2D 得到估计值
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    
    # 计算残差 (观测点 - 投影点)
    res = (imgpoints[i] - imgpoints2).squeeze() 
    
    # 计算每个角点相对于图像中心的极角 (用于图a)
    pts_centered = imgpoints[i].squeeze() - np.array([cx, cy])
    angles = np.arctan2(pts_centered[:, 1], pts_centered[:, 0]) * 180 / np.pi
    angles = np.mod(angles, 360) 
    
    # 记录数据
    err_mag = np.linalg.norm(res, axis=1)
    all_errors_mag.extend(err_mag)
    all_polar_angles.extend(angles)
    all_res_x.extend(res[:, 0])
    all_res_y.extend(res[:, 1])
    image_indices.extend([i] * len(err_mag))

# 绘图
plt.style.use('default')
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 4, figure=fig)

# --- 图 (a) 极线误差分析 ---
ax_a1 = fig.add_subplot(gs[0, 0])
# 绘制折线散点图 (reprojection error vs polar angle)
sorted_idx = np.argsort(all_polar_angles)
ax_a1.plot(np.array(all_polar_angles)[sorted_idx], np.array(all_errors_mag)[sorted_idx], 
           color='blue', marker='x', markersize=3, linewidth=0.5, alpha=0.7)
ax_a1.set_xlabel('polar angle (deg)')
ax_a1.set_ylabel('reprojection error (pixels)')
ax_a1.set_ylim(0, max(all_errors_mag) * 1.1)
ax_a1.grid(True, linestyle='-', alpha=0.5)

ax_a2 = fig.add_subplot(gs[0, 1])
# 绘制角度分布直方图
ax_a2.hist(all_polar_angles, bins=15, color='#348ABD', rwidth=0.85, alpha=0.9)
ax_a2.set_xlabel('polar angle (deg)')
ax_a2.set_ylabel('count')
ax_a2.grid(True, linestyle='-', alpha=0.5)

fig.text(0.25, 0.48, "(a)", fontsize=14, fontweight='bold')

# --- 图 (b) 投影误差分析 ---
ax_b1 = fig.add_subplot(gs[0, 2])
# 绘制角点在图像上的覆盖情况（色彩代表误差大小）
sc1 = ax_b1.scatter(np.array(imgpoints).reshape(-1,2)[:,0], 
                    np.array(imgpoints).reshape(-1,2)[:,1], 
                    c=all_errors_mag, cmap='jet', s=5, alpha=0.6)
ax_b1.set_xlim(0, gray_shape[0])
ax_b1.set_ylim(gray_shape[1], 0) # 图像坐标系 Y 反向
ax_b1.set_aspect('equal')
ax_b1.grid(True, linestyle='-', alpha=0.3)

ax_b2 = fig.add_subplot(gs[0, 3])
# 绘制误差分布散点图 (Error X vs Error Y)
sc2 = ax_b2.scatter(all_res_x, all_res_y, c=image_indices, cmap='jet', 
                    s=20, marker='x', alpha=0.7)
ax_b2.set_xlabel('error x (pix)')
ax_b2.set_ylabel('error y (pix)')
ax_b2.set_xlim([-1, 1])
ax_b2.set_ylim([-1, 1])
ax_b2.set_aspect('equal')
ax_b2.grid(True, linestyle='-', alpha=0.5)

# 添加颜色条，指示图像索引
cbar = plt.colorbar(sc2, ax=ax_b2, fraction=0.046, pad=0.04)
cbar.set_label('image index')

fig.text(0.75, 0.48, "(b)", fontsize=14, fontweight='bold')

plt.suptitle("Figure 5-6 Calibration Error Analysis", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
