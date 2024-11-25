#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>
#include <thread>
#include <chrono>
#include <mutex>

#include "vulkan_utils.h"
#include "camera_capture.h"
#include "mechArmCtrl.h"
#include "laser_ranging.h"

using namespace std::this_thread;
using namespace std::chrono_literals;

bool checkNutsRemoved();
bool alignRack();
bool alignAxle();

int main()
{
    // 任务是使用vulkan加速，
    // 使用OpenCV或yolo等成熟方法
    // 进行四个主要任务
    return 0;
}
bool checkNutsRemoved(){
    // 使用yolov8或者传统目标检测算法
    // 检测在拆卸轮胎前，螺母是否已经拆完
    bool fullyRemoved = false;
    // TODO: 此处占位，之后实现
    return fullyRemoved;
}

bool alignRack(){
    // 判断机械爪的中轴是否和轮胎的中轴对齐
    // 若不对齐，则需要调整机械爪的位置
    // 包括给出建议的移动方向和距离
    // （像素，之后根据距离等再添加转换函数）
    // TODO: 此处占位，之后实现
    bool aligned = false;
    return aligned;
}

bool alignAxle(){
    // 判断机械爪的中心是否和车轴中心对齐
    // 若不对齐，则需要调整机械爪的位置
    // 使用机器学习或者传统视觉方案
    // 需求精度最高，1mm，可能更应该用yolo
    // 包括给出建议的移动方向和距离
    // TODO: 此处占位，之后实现
    bool aligned = false;
    return aligned;
}