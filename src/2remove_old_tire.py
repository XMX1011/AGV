# 10）点击“校准”按钮，在距离旧轮胎1000mm的位置，
# 系统首先启动粗定位程序，对旧胎进行初步的定位校准。
# 完成粗定位后，继续平稳前行至距旧胎900mm的位置。
# 随后，旧胎检测单元迅速响应，伸出其检测杆直至距离旧胎600mm处，进行精定位操作；
# 然后对螺母进行安全检测，若有螺母，发出安全报警；
# 否则，自动检测并调整旧胎门架与旧胎的相对位姿，
# 使旧胎门架自动对准车轴，发出完成校准提醒；
# 否则，发出校准错误报警；


# 第二步，拆除旧轮胎
# 首先看螺母是否拆卸完全，可以考虑机器学习或者模式匹配
# 对齐过程精度可以不用太高，大致对齐，角度和第一步一样，差不多就行
# 不过重点在于俯仰角度，一定要是平行的

from rough_alignment import align_to_vehicle
from nut_detection import ND_mobilenetv3,ND_trad
from utils import get_distance, get_pitch_angle, get_yaw_angle, get_roll_angle,alert


def pitch_adjustment():
    # 根据激光测距调整俯仰角度
    pitch_angel = get_pitch_angle()
    pass

if __name__ == "__main__":
    if not ND_mobilenetv3():
        alert()
    if align_to_vehicle():
        print("对齐成功")
    else:
        print("未对齐，需要重新对齐")