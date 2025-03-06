#ifndef AXLE_LOCATOR_H
#define AXLE_LOCATOR_H

#include <opencv2/opencv.hpp>
#include <string>

namespace AxleLocator
{

    /**
     * @brief 车轴定位结果结构体
     */
    struct AxleResult
    {
        cv::Point2f center;  // 轴心坐标
        bool success;        // 是否成功定位
        std::string message; // 状态消息
        cv::Mat resultImage; // 结果图像

        std::map<std::string, cv::Vec3f> debug; // 调试信息
    };

    /**
     * @brief 车轴定位器参数配置
     */
    struct LocatorConfig
    {
        // 图像预处理参数
        int blurKernelSize = 5;      // 高斯模糊内核大小
        bool enhanceContrast = true; // 是否增强对比度

        // 边缘检测参数
        int cannyThreshold1 = 50;  // Canny低阈值
        int cannyThreshold2 = 150; // Canny高阈值

        // 圆形检测参数
        int minRadius = 50;               // 最小半径
        int maxRadius = 200;              // 最大半径
        double circleAccumThreshold = 40; // 圆形累加器阈值
        double dp = 1.2;                  // 霍夫变换分辨率参数

        bool detectHubSpecifically = true;  // 识别含有轮胎的情况下的识别轮毂中心
        int hubRadiusRatio = 3;         // 轮毂半径比例

        // 轮廓处理参数
        double minContourArea = 100; // 最小轮廓面积

        // 输出标记参数
        int markerSize = 20;         // 标记大小
        bool drawCoordinates = true; // 是否绘制坐标文本
    };

    /**
     * @brief 检测有轮胎情况下的车轴中心
     * @param image 输入图像
     * @param config 配置参数
     * @return 车轴定位结果
     */
    AxleResult locateAxleWithTire(const cv::Mat &image, const LocatorConfig &config = LocatorConfig());

    /**
     * @brief 检测无轮胎情况下的车轴中心
     * @param image 输入图像
     * @param config 配置参数
     * @return 车轴定位结果
     */
    AxleResult locateAxleWithoutTire(const cv::Mat &image, const LocatorConfig &config = LocatorConfig());

    /**
     * @brief 统一车轴定位接口
     * @param image 输入图像
     * @param hasTire 是否有轮胎
     * @param config 配置参数
     * @return 车轴定位结果
     */
    AxleResult locateAxle(const cv::Mat &image, bool hasTire, const LocatorConfig &config = LocatorConfig());
    AxleResult locateAxleMultiStage(const cv::Mat &image, bool hasTire, const LocatorConfig &config);

    /**
     * @brief 保存结果图像到文件
     * @param result 车轴定位结果
     * @param filename 文件名
     * @return 是否保存成功
     */
    bool saveResultImage(const AxleResult &result, const std::string &filename);

} // namespace AxleLocator

#endif // AXLE_LOCATOR_H