import cv2     # 导入open cv库

img = cv2.imread("buble.jpg") # 读取图像。讲光信号转化为计算机内存中的矩阵数据

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将彩色图像转化为灰度图（黑白单色）
# 颜色信息对计算气泡数量来说是噪声干扰，灰度化减少计算量

_, binary = cv2.threshold(gray,120,255,cv2.THRESH_BINARY) # 图像二值化。cv.threshold() 阈值处理
# 通过二值化处理来区分气泡（前景）和基底（背景）
contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # 寻找轮廓
# 在二值图上找寻所有白色斑块的边界线。RETR_EXTERNAL表示只招最外围的轮廓

print("bubble count:",len(contours)) # 统计气泡个数