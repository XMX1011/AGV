# 点击“校准”按钮，在距离旧胎架1000mm时，自动检测旧胎门架与备胎架的相对位姿，
# 全向智能车体和门架自动调整位姿，
# 将旧胎门架自动对准备胎架，发出完成校准提醒；
# 否则，发出校准错误报警
# 最后一步
# 和第一步差不多，中轴对齐即可，对齐之后和距离接近到能放下去之前给出信号抬起一定距离

# ! 具体像素和距离的对应关系还需要进行计算修改

from rough_alignment import align_to_rack
def signal_grapple_up():
    # 距离轮胎架大概胎宽距离的时候给出一个向上抬极短距离的信号，方便放下轮胎
    if distance_to_rack() < 10:
        pass
    pass
def signal_grapple_down():
    # 距离轮胎架距离接近到能放下去时给出信号
    if distance_to_rack() < 5:
        pass
    pass

def distance_to_rack():
    # 调用激光测距模块，返回距离胎架的距离
    pass

if __name__ == '__main__':
    if align_to_rack() and distance_to_rack() < 5:
        signal_grapple_down()