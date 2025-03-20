#include "config.h"

Eigen::Vector2d computeGripperCommand(const cv::Point2d &pixel,
                                     const FineConfig &cfg)
{
    // 使用统一坐标转换流程
    Eigen::Vector3d gripper_point = pixelTo3D(pixel, cfg.cam, 0.0); // 假设固定距离
    
    // 应用抓手偏移补偿
    gripper_point.head<2>() -= cfg.gripper_offset;
    
    // 返回抓手指令部分
    return {gripper_point.x(), gripper_point.y()};
}