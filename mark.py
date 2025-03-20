import cv2

# 读取图片
image = cv2.imread('../build/axle_result.jpg')

# 获取图片的高度和宽度
height, width = image.shape[:2]

# 计算图片的中心点
center_x, center_y = width // 2, height // 2

# 定义十字的长度
cross_length = 20

# 在中心点画十字
color = (0, 0, 255)  # 红色
thickness = 2       # 线条粗细

# 画横线
cv2.line(image, (center_x - cross_length, center_y), (center_x + cross_length, center_y), color, thickness)
# 画竖线
cv2.line(image, (center_x, center_y - cross_length), (center_x, center_y + cross_length), color, thickness)

# 显示图片
cv2.imshow('Image with Center Cross', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

