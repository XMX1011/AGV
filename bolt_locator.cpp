#include "bolt_locator.h"
#include <iostream>

namespace BoltLocator
{

    // 使用辅助摄像机进行螺栓孔定位
    // 故需要进行完全不一样的图像预处理
    // 考虑使用模板匹配还是直接使用形状等其他特征定位

    /**
     * @brief 图像预处理
     * @param img 输入图像
     * @param config 配置参数
     * @return 预处理后的图像
     */

    cv::Mat preprocess(cv::Mat img, const LocatorConfig &config)
    {
        cv::Mat gray, blurred, binary, result;

        // 灰度化
        if (img.channels() > 1)
        {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = img.clone();
        }

        // 去噪（高斯模糊）
        cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

        // 自适应阈值二值化
        cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

        // 形态学操作（开运算）
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, result, cv::MORPH_OPEN, kernel);

        return result;
    }

    /**
     * @brief 定位螺栓孔位置
     * @param img 输入图像
     * @param center 车轴中心坐标
     * @param config 配置参数
     * @return BoltInfo 螺栓孔坐标,共计bolt_num个
     */
    BoltInfo locate_bolt(cv::Mat img, cv::Point2f center, const LocatorConfig &config)
    {
        BoltInfo bolt_info;
        // 检查图像是否为空
        if (img.empty())
        {
            std::cerr << "Error: Input image is empty in locate_bolt." << std::endl;
            return bolt_info;
        }

        // 确保图像是单通道灰度图像
        cv::Mat gray;
        if (img.channels() > 1)
        {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = img.clone();
        }

        // 确保图像类型为 CV_8UC1
        if (gray.type() != CV_8UC1)
        {
            std::cerr << "Error: Image type is not CV_8UC1 in locate_bolt." << std::endl;
            return bolt_info;
        }
        std::vector<cv::Vec3f> circles;

        // 使用霍夫圆变换检测圆形
        cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1, img.rows / 16, 100, 30, 10, 50);

        // 筛选符合条件的螺栓孔
        for (size_t i = 0; i < circles.size() && bolt_info.bolt_pos.size() < config.bolt_num; ++i)
        {
            cv::Point2f circle_center(circles[i][0], circles[i][1]);
            float radius = circles[i][2];

            // 判断是否在车轴中心附近
            if ((cv::norm(circle_center - center) < config.max_distance_from_center) && cv::norm(circle_center - center) > config.min_distance_from_center)
            {
                bolt_info.bolt_pos.push_back(circle_center);
            }
        }

        return bolt_info;
    }

    /**
     * @brief 标记螺栓孔位置
     * @param img 输入图像
     * @param BoltInfo.bolt_pos 螺栓孔坐标
     * @return 标记后的图像
     */
    void mark_bolt(cv::Mat img, BoltInfo bolt_info)
    {
        for (const auto &pos : bolt_info.bolt_pos)
        {
            // 绘制圆圈标记螺栓孔
            cv::circle(img, pos, 5, cv::Scalar(0, 255, 0), 2);
        }
    }
}
