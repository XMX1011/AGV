#include "config.h"
#include <opencv2/imgproc.hpp>

Eigen::Vector2d computeVehicleCommand(const cv::Point2d &pixel,
                                     const LaserData &laser_data,
                                     const CoarseConfig &cfg,
                                     double current_vehicle_x)
{
    // 处理激光数据获取有效距离和俯仰角
    auto [pitch, distance] = processLaserData(laser_data);
    
    // 坐标转换流程
    Eigen::Vector3d cam_pos = pixelTo3D(pixel, cfg.cam, distance);
    Eigen::Vector3d world_pos = applyPitchCompensation(cam_pos, pitch);
    
    // 分解车体指令
    MotionCommand cmd = decomposeMotion(world_pos, current_vehicle_x);
    
    // 返回车体指令部分
    return {cmd.vehicle.x, cmd.vehicle.z};
}