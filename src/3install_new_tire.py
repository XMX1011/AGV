# 点击“校准”按钮，在距离车轴1000mm的位置，系统首先启动粗定位程序，对车轴进行初步的定位校准。
# 完成粗定位后，继续平稳前行至距车轴900mm的位置。
# 随后，备胎检测单元迅速响应，伸出其检测杆直至距离车轴600mm处，进行精定位操作；
# 然后自动检测备胎螺栓孔与车轴螺栓的相对位姿，
# 系统自动调整备胎位姿，将备胎螺栓孔对准车轴螺栓，发出完成校准提醒；
# 否则，发出校准错误报警，检查无误后，重新校准；

# 第三步，安装新轮胎
# 重头戏来了，按照下面的步骤
# 首先定位备胎螺栓孔位置，即检测备胎螺栓孔与过备胎圆心垂线夹角λ，由相机采集图像
# y为过备胎圆心垂线、RL为已知螺栓孔圆心与备胎圆心距离、x为螺栓孔圆心与过备胎圆心垂线距离

# 然后定位车轴位置
# 由相机采集图像，系统对图像预处理后通过模板匹配算法快速定位车轴ROI，
# 用Hough变换对车轴外沿圆形边缘以及10个圆形螺栓轮廓边缘进行提取，
# 使用目标定位算法定位车轴相对相机位置以及螺栓相对垂直过车轴圆心的直线的夹角

# 考虑是否分两步，远距离时候使用粗对齐，近距离时候使用精对齐

from rough_alignment import align_to_vehicle
from precise_alignment import traditionalCV,dl_yolov8

distance_to_vehicle = 1000 # mm
# 使用激光测距判断摄像头与车轴的距离
# 实时变化
def distance_to_camera(distance_to_vehicle):
    pass

def main():
    if distance_to_camera(distance_to_vehicle) < 900:
        align_to_vehicle()
    elif distance_to_camera(distance_to_vehicle) < 600:
        # traditionalCV(distance_to_vehicle)
        dl_yolov8()