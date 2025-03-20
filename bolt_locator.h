#ifndef BOLT_LOCATOR_H
#define BOLT_LOCATOR_H

#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief 根据可能得需要，定位螺栓孔位置
 * @details 具体螺栓孔数量暂时默认是10，排列依照正十边形的十个角，车轴中心坐标由外部传入，来源是axle_locator的结果
 * @details 实验用轮毂车轴孔直径281mm，分布圆直径335mm，辐板10孔，螺栓孔径26mm
 *
 * @return std::vector<cv::Point2f> 螺栓孔坐标,共计bolt_num个
 */
namespace BoltLocator
{
    struct BoltInfo
    {
        std::vector<cv::Point2f> bolt_pos;
    };
    struct LocatorConfig
    {
        int bolt_num = 10;

        double max_distance_from_center = 150; // 螺栓孔中心到车轴中心的最大距离
        double min_distance_from_center = 30;  // 螺栓孔边缘到车轴中心的最大距离
        double bolt_radius = 10;               // 螺栓孔径
        double axle_hole_radius = 50;          // 车轴孔径
        // 输出标记参数
        int markerSize = 3;   // 标记大小
        bool markBolt = true; // 是否绘制坐标文本
    };

    cv::Mat preprocess(cv::Mat img, const LocatorConfig &config = LocatorConfig());
    BoltInfo locate_bolt(cv::Mat img, cv::Point2f center, const LocatorConfig &config = LocatorConfig());
    void mark_bolt(cv::Mat img, BoltInfo bolt_info);
}

#endif // BOLT_LOCATOR_H