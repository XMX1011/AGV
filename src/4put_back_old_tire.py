# 点击“校准”按钮，在距离旧胎架1000mm时，自动检测旧胎门架与备胎架的相对位姿，
# 全向智能车体和门架自动调整位姿，
# 将旧胎门架自动对准备胎架，发出完成校准提醒；
# 否则，发出校准错误报警
# 最后一步
# 和第一步差不多，中轴对齐即可，对齐之后和距离接近到能放下去之前给出信号抬起一定距离

from rough_alignment import align_to_rack