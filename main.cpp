#include <iostream>
#include <string>
#include "axle_locator.h"

int main(int argc, char **argv)
{
    // 参数检查
    if (argc < 2)
    {
        std::cerr << "Usage: AP <image_path> [--with-tire/--without-tire]" << std::endl;
        return -1;
    }

    // 解析参数
    std::string imagePath = argv[1];
    bool hasTire = true; // 默认有轮胎

    if (argc > 2)
    {
        std::string option = argv[2];
        if (option == "--without-tire")
        {
            hasTire = false;
        }
    }

    // 读取图像
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not read image from " << imagePath << std::endl;
        return -1;
    }

    // 配置参数（如有必要，可以根据实际情况调整）
    AxleLocator::LocatorConfig config;
    // 示例：调整参数
    // config.minRadius = 50;
    // config.maxRadius = 250;
    if (hasTire)
    {
        config.detectHubSpecifically = true;
    }
    else
    {
        config.detectHubSpecifically = false;
    }

    // 处理图像
    std::cout << "Processing image in " << (hasTire ? "with-tire" : "without-tire") << " mode..." << std::endl;
    AxleLocator::AxleResult result = AxleLocator::locateAxleMultiStage(image, hasTire, config);

    // 输出结果
    if (result.success)
    {
        std::cout << "Axle center located at: (" << result.center.x << ", " << result.center.y << ")" << std::endl;
        std::cout << "Status: " << result.message << std::endl;
    }
    else
    {
        std::cout << "Warning: " << result.message << std::endl;
        std::cout << "Using fallback position: (" << result.center.x << ", " << result.center.y << ")" << std::endl;
    }

    // 保存结果图像
    std::string outputFilename = "axle_result.jpg";
    if (AxleLocator::saveResultImage(result, outputFilename))
    {
        std::cout << "Result image saved as: " << outputFilename << std::endl;
    }

    // 显示结果
    cv::namedWindow("Axle Location Result", cv::WINDOW_NORMAL);
    cv::imshow("Axle Location Result", result.resultImage);
    cv::waitKey(0);

    return 0;
}