# 备胎门架朝向备胎并进入检测范围，检测备胎是否完全放气；
# 点击“校准”按钮，开始在距离备胎1000mm时自动检测备胎门架与备胎的相对位姿，
# 全向智能车体和门架自动调整位姿，将备胎门架自动对准备胎，发出完成校准提醒；
# 否则，发出校准错误报警；


# 第一步，从轮胎架子上取出新轮胎，
# 大致对齐中轴，俯仰角度不考虑，滚转角度不考虑
# 实际上考虑一下吧，就看一眼角度参数是否过于离谱，主要还是要看中轴
# 我看到有说“备胎完全放气是否需要视觉检测”
# 上面的内容可以附加一个功能，不一定实际采用，有空的话就实现一下
# 对齐后后给出一个信号，让机器爪在抓到轮胎后略微抬起



import cv2
from rough_alignment import align_to_rack


# 同样需要很多的图像数据
def tire_is_flat():
    pass
def alert():
    pass
def signal_grapple_up():
    pass

if __name__ == '__main__':
    if not tire_is_flat():
        alert()
    aligned = align_to_rack()
    # # 爪抬起信号
    # if aligned :
    #     signal_grapple_up()