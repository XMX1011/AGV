#include "axle_locator.h"
#include <iostream>

namespace AxleLocator
{

    namespace
    {

        /**
         * @brief 图像预处理
         * @param image 输入图像
         * @param hasTire 是否有轮胎
         * @param config 配置参数
         * @return 预处理后的图像
         */
        cv::Mat preprocessImage(const cv::Mat &image, bool hasTire, const LocatorConfig &config)
        {
            cv::Mat result;

            // 转为灰度图
            if (image.channels() > 1)
            {
                cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
            }
            else
            {
                image.copyTo(result);
            }

            // 高斯模糊去噪
            if (config.blurKernelSize > 0)
            {
                cv::GaussianBlur(result, result,
                                 cv::Size(config.blurKernelSize, config.blurKernelSize), 0);
            }
            // 基础中值滤波（窗口大小5x5）
            cv::Mat median_blur;
            cv::medianBlur(result, median_blur, 5);

            // 混合处理：对中值滤波结果再应用均值滤波（窗口3x3）
            cv::Mat mixed_median;
            cv::blur(median_blur, result, cv::Size(3, 3));
            // 应用拉普拉斯算子
            cv::Mat laplacian;
            cv::Laplacian(result, laplacian, CV_16S, 3);

            // 将结果转换为8UC1并叠加原图
            cv::convertScaleAbs(laplacian, laplacian);
            cv::addWeighted(result, 1.5, laplacian, -0.5, 0, result);

            // 根据是否有轮胎调整预处理参数
            if (hasTire)
            {
                if (config.enhanceContrast)
                {
                    // 对比度增强
                    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
                }
            }
            else
            {
                // 局部自适应阈值处理，处理光照不均
                cv::adaptiveThreshold(result, result, 255,
                                      cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv::THRESH_BINARY_INV, 15, 2);

                // 形态学操作去除小噪点
                int morphSize = 2;
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                            cv::Size(2 * morphSize + 1, 2 * morphSize + 1));
                cv::morphologyEx(result, result, cv::MORPH_OPEN, element);
            }

            return result;
        }

        /**
         * @brief 检测圆形
         * @param image 输入图像
         * @param config 配置参数
         * @return 检测到的圆形向量
         */
        std::vector<cv::Vec3f> detectCircles(const cv::Mat &image, const LocatorConfig &config)
        {
            std::vector<cv::Vec3f> circles;

            cv::Mat enhancedImage = image.clone();
            if (config.detectHubSpecifically)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                clahe->apply(enhancedImage, enhancedImage);
            }
            // cv::HoughCircles(enhancedImage, circles, cv::HOUGH_GRADIENT, config.dp, image.rows / 10, config.cannyThreshold1, config.cannyThreshold2, config.minRadius, config.maxRadius);
            cv::HoughCircles(enhancedImage, circles, cv::HOUGH_GRADIENT_ALT, config.dp, image.rows / 10, 300, 0.75, config.minRadius, config.maxRadius);
            return circles;
        }

        /**
         * @brief 检测椭圆
         * @param image 输入图像
         * @param config 配置参数
         * @return 检测到的椭圆向量
         */
        std::vector<cv::RotatedRect> detectEllipses(const cv::Mat &image, const LocatorConfig &config)
        {
            std::vector<cv::RotatedRect> ellipses;

            // 查找轮廓
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(image.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            // 对每个轮廓拟合椭圆
            for (const auto &contour : contours)
            {
                // 过滤面积过小的轮廓
                if (cv::contourArea(contour) < config.minContourArea)
                {
                    continue;
                }

                // 轮廓点数必须大于等于5才能拟合椭圆
                if (contour.size() >= 5)
                {
                    cv::RotatedRect ellipse = cv::fitEllipse(contour);

                    // 过滤掉过小或过大的椭圆
                    float radius = (ellipse.size.width + ellipse.size.height) / 4.0;
                    if (radius >= config.minRadius && radius <= config.maxRadius)
                    {
                        ellipses.push_back(ellipse);
                    }
                }
            }

            return ellipses;
        }

        /**
         * @brief 标记轴心位置
         * @param image 结果图像
         * @param center 轴心坐标
         * @param config 配置参数
         */
        void markAxleCenter(cv::Mat &image, const cv::Point2f &center, const LocatorConfig &config)
        {
            // 画十字标记
            cv::line(image,
                     cv::Point(center.x - config.markerSize, center.y),
                     cv::Point(center.x + config.markerSize, center.y),
                     cv::Scalar(0, 0, 255), 2);
            cv::line(image,
                     cv::Point(center.x, center.y - config.markerSize),
                     cv::Point(center.x, center.y + config.markerSize),
                     cv::Scalar(0, 0, 255), 2);

            // 画圆形标记
            cv::circle(image, center, 3, cv::Scalar(0, 0, 255), -1);

            // 标注坐标
            if (config.drawCoordinates)
            {
                std::string text = "(" + std::to_string(int(center.x)) + ", " +
                                   std::to_string(int(center.y)) + ")";
                cv::putText(image, text, cv::Point(center.x + 10, center.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }

    }

    AxleResult locateAxleWithTire(const cv::Mat &image, const LocatorConfig &config)
    {
        AxleResult result;
        result.success = false;

        // 复制输入图像作为输出基础
        if (image.empty())
        {
            result.message = "Empty input image";
            return result;
        }
        image.copyTo(result.resultImage);

        // 预处理图像
        cv::Mat processedImage = preprocessImage(image, true, config);

        // 边缘检测
        cv::Mat edges;
        cv::Canny(processedImage, edges, config.cannyThreshold1, config.cannyThreshold2);

        // 寻找圆形
        std::vector<cv::Vec3f> circles = detectCircles(processedImage, config);

        if (!circles.empty())
        {
            // 计算半径并且输出
            float radius = circles[0][2];
            std::cout << "Radius: " << radius << std::endl;
            result.debug["radius"] = radius;
            // 找到了圆形，可能有多个，选择最可能的轮胎圆（通常是最大的）
            cv::Vec3f tireCircle = circles[0];
            cv::Vec3f hubCircle = circles[0];
            for (const auto &circle : circles)
            {
                if (circle[2] > hubCircle[2])
                {
                    hubCircle = circle;
                }
            }
            if (config.detectHubSpecifically && circles.size() > 1)
            {
                for (const auto &circle : circles)
                {
                    if (circle[2] < tireCircle[2] / config.hubRadiusRatio * 1.5 && circle[2] > tireCircle[2] / config.hubRadiusRatio * 0.5)
                    {
                        float dx = circle[0] - tireCircle[0];
                        float dy = circle[1] - tireCircle[1];
                        float distance = std::sqrt(dx * dx + dy * dy);
                        if (distance < tireCircle[2] * 0.5)
                        {
                            hubCircle = circle;
                            result.center = cv::Point2f(hubCircle[0], hubCircle[1]);
                            result.success = true;
                            result.message = "Axle center located using hub circle detection";
                            result.debug["tire_circle"] = tireCircle;
                            result.debug["hub_circle"] = hubCircle;
                            markAxleCenter(result.resultImage, result.center, config);
                            return result;
                        }
                    }
                }
            }

            // 设置结果
            result.center = cv::Point2f(tireCircle[0], tireCircle[1]);
            result.success = true;
            result.message = "Axle center located using tire circle detection";

            // 绘制检测到的圆
            cv::Point center(cvRound(tireCircle[0]), cvRound(tireCircle[1]));
            // int radius = cvRound(tireCircle[2]);
            cv::circle(result.resultImage, center, radius, cv::Scalar(0, 0, 255), 2);

            // 标记轴心位置
            markAxleCenter(result.resultImage, result.center, config);
            return result;
        }

        // 如果没有找到圆形，尝试寻找椭圆
        std::vector<cv::RotatedRect> ellipses = detectEllipses(edges, config);

        if (!ellipses.empty())
        {
            // 计算半径并且输出
            float radius = (ellipses[0].size.width + ellipses[0].size.height) / 4.0;
            std::cout << "Radius: " << radius << std::endl;
            result.debug["radius"] = radius;
            // 使用最大的椭圆中心
            cv::RotatedRect largestEllipse = ellipses[0];
            for (const auto &ellipse : ellipses)
            {
                if (ellipse.size.area() > largestEllipse.size.area())
                {
                    largestEllipse = ellipse;
                }
            }

            // 设置结果
            result.center = largestEllipse.center;
            result.success = true;
            result.message = "Axle center located using ellipse detection";

            // 绘制椭圆
            cv::ellipse(result.resultImage, largestEllipse, cv::Scalar(0, 255, 0), 2);

            // 标记轴心位置
            markAxleCenter(result.resultImage, result.center, config);
            return result;
        }

        // 如果没有找到圆形或椭圆，返回图像中心
        result.center = cv::Point2f(image.cols / 2.0f, image.rows / 2.0f);
        result.message = "Fallback to image center - no circles or ellipses detected";

        // 标记轴心位置
        markAxleCenter(result.resultImage, result.center, config);

        return result;
    }

    AxleResult locateAxleWithoutTire(const cv::Mat &image, const LocatorConfig &config)
    {
        AxleResult result;
        result.success = false;

        // 复制输入图像作为输出基础
        if (image.empty())
        {
            result.message = "Empty input image";
            return result;
        }
        image.copyTo(result.resultImage);

        // 预处理图像
        cv::Mat processedImage = preprocessImage(image, false, config);

        // 轮廓检测
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(processedImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 如果没有找到轮廓，返回图像中心
        if (contours.empty())
        {
            result.center = cv::Point2f(image.cols / 2.0f, image.rows / 2.0f);
            result.message = "Fallback to image center - no contours detected";

            // 标记轴心位置
            markAxleCenter(result.resultImage, result.center, config);
            return result;
        }

        // 寻找最大的轮廓
        int largestContourIdx = 0;
        double largestArea = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > largestArea)
            {
                largestArea = area;
                largestContourIdx = i;
            }
        }

        // 拟合椭圆或圆形
        if (contours[largestContourIdx].size() >= 5)
        {
            cv::RotatedRect ellipse = cv::fitEllipse(contours[largestContourIdx]);

            // 设置结果
            result.center = ellipse.center;
            result.success = true;
            result.message = "Axle center located using ellipse fitting";

            // 绘制椭圆
            cv::ellipse(result.resultImage, ellipse, cv::Scalar(0, 255, 0), 2);

            // 标记轴心位置
            markAxleCenter(result.resultImage, result.center, config);
            return result;
        }

        // 如果不能拟合椭圆，使用轮廓的中心矩
        cv::Moments mu = cv::moments(contours[largestContourIdx]);
        if (mu.m00 != 0)
        {
            result.center = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
            result.success = true;
            result.message = "Axle center located using contour moments";
        }
        else
        {
            result.center = cv::Point2f(image.cols / 2.0f, image.rows / 2.0f);
            result.message = "Fallback to image center - moment calculation failed";
        }

        // 绘制轮廓
        cv::drawContours(result.resultImage, contours, largestContourIdx, cv::Scalar(0, 255, 0), 2);

        // 标记轴心位置
        markAxleCenter(result.resultImage, result.center, config);

        return result;
    }

    // AxleResult locateAxle(const cv::Mat &image, bool hasTire, const LocatorConfig &config)
    // {
    //     if (hasTire)
    //     {
    //         return locateAxleWithTire(image, config);
    //     }
    //     else
    //     {
    //         return locateAxleWithoutTire(image, config);
    //     }
    // }

    bool saveResultImage(const AxleResult &result, const std::string &filename)
    {
        if (result.resultImage.empty())
        {
            std::cerr << "Error: Empty result image" << std::endl;
            return false;
        }

        try
        {
            cv::imwrite(filename, result.resultImage);
            return true;
        }
        catch (const cv::Exception &ex)
        {
            std::cerr << "Error saving image: " << ex.what() << std::endl;
            return false;
        }
    }
    AxleResult locateAxleMultiStage(const cv::Mat &image, bool hasTire, const LocatorConfig &config)
    {
        AxleResult result;

        // 步骤1：标准处理
        result = hasTire ? locateAxleWithTire(image, config) : locateAxleWithoutTire(image, config);

        if (!result.success)
        {
            return result; // 如果标准处理失败，直接返回
        }

        // 步骤2：获取感兴趣区域(ROI)进行精细处理
        cv::Point2f initialCenter = result.center;
        int roiSize = hasTire ? static_cast<int>(config.minRadius * 3) : static_cast<int>(config.minRadius * 2);

        // 确保ROI在图像内部
        int x = std::max(0, static_cast<int>(initialCenter.x - roiSize / 2));
        int y = std::max(0, static_cast<int>(initialCenter.y - roiSize / 2));
        int width = std::min(image.cols - x, roiSize);
        int height = std::min(image.rows - y, roiSize);

        // 检查ROI是否有效
        if (width > 20 && height > 20)
        {
            cv::Rect roi(x, y, width, height);
            cv::Mat roiImage = image(roi);

            // 创建针对ROI的配置
            LocatorConfig roiConfig = config;
            roiConfig.minRadius = config.minRadius / 2;
            roiConfig.maxRadius = config.minRadius * 2;

            // 在ROI上运行优化的检测
            AxleResult roiResult;
            if (hasTire)
            {
                // 针对ROI调用特定的轮毂检测算法
                roiConfig.detectHubSpecifically = true;
                roiResult = locateAxleWithTire(roiImage, roiConfig);
            }
            else
            {
                roiResult = locateAxleWithoutTire(roiImage, roiConfig);
            }

            // 如果ROI检测成功，将结果转换回原图坐标
            if (roiResult.success)
            {
                result.center = cv::Point2f(roiResult.center.x + x, roiResult.center.y + y);
                result.message = "Refined axle center using ROI processing";

                // 重新标记原图
                image.copyTo(result.resultImage); // 确保使用原始图像
                markAxleCenter(result.resultImage, result.center, config);
            }
        }

        return result;
    }
}