#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem> // 用于检查文件路径

using json = nlohmann::json;

// 定义移动结果结构体
struct Movement
{
    double x, y, yaw;                  // 车体移动量（x,y）和角度（yaw）
    double gripper_y, gripper_z;       // 抓手移动量（y,z）
    double gripper_pitch, gripper_yaw; // 抓手角度（pitch,yaw）
};

// 配置文件键名常量
const std::string WORLD_POINTS_KEY = "world_points";
const std::string IMAGE_POINTS_KEY = "image_points";
const std::string CAMERA_KEY = "camera";
const std::string INTRINSIC_MATRIX_KEY = "intrinsic_matrix";
const std::string DISTORTION_COEFFS_KEY = "distortion_coeffs";
const std::string PARAMETERS_KEY = "parameters";

// 读取配置文件
json loadConfig(const std::string &filename)
{
    // 使用std::ifstream检查文件是否存在
    std::ifstream file(filename);
    if (!file.good())
    {
        throw std::runtime_error("Config file does not exist or cannot be opened: " + filename);
    }

    json config;
    file >> config;

    // 验证必要字段是否存在
    if (config.find(WORLD_POINTS_KEY) == config.end() ||
        config.find(IMAGE_POINTS_KEY) == config.end() ||
        config.find(CAMERA_KEY) == config.end())
    {
        throw std::runtime_error("Config file is missing required fields.");
    }
    return config;
}

// 使用PnP计算相机位姿
cv::Mat computeCameraPose(const json &config)
{
    // 提取世界坐标点
    std::vector<cv::Point3f> worldPoints;
    for (const auto &point : config[WORLD_POINTS_KEY])
    {
        worldPoints.emplace_back(cv::Point3f(point["x"], point["y"], point["z"]));
    }

    // 提取图像坐标点
    std::vector<cv::Point2f> imagePoints;
    for (const auto &point : config[IMAGE_POINTS_KEY])
    {
        imagePoints.emplace_back(cv::Point2f(point["x"], point["y"]));
    }

    // 提取相机内参和畸变系数
    cv::Mat intrinsicMatrix = (cv::Mat_<double>(3, 3) << config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][0][0], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][0][1], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][0][2],
                               config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][1][0], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][1][1], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][1][2],
                               config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][2][0], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][2][1], config[CAMERA_KEY][INTRINSIC_MATRIX_KEY][2][2]);
    std::vector<double> distortionCoeffs = config[CAMERA_KEY][DISTORTION_COEFFS_KEY];
    cv::Mat distCoeffs(distortionCoeffs, true);

    // 验证点数是否一致
    if (worldPoints.size() != imagePoints.size())
    {
        throw std::runtime_error("Number of world points and image points do not match.");
    }

    // 计算PnP
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(worldPoints, imagePoints, intrinsicMatrix, distCoeffs, rvec, tvec);
    if (!success)
    {
        throw std::runtime_error("PnP calculation failed.");
    }

    // 将旋转向量转换为旋转矩阵
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);

    // 输出平移向量和旋转矩阵
    std::cout << "Translation Vector:\n"
              << tvec << "\n";
    std::cout << "Rotation Matrix:\n"
              << rotationMatrix << "\n";

    return tvec; // 返回平移向量
}

// 计算移动量
Movement computeMovement(const json &config, const cv::Mat &cameraPose)
{
    Movement result;

    // 验证cameraPose尺寸
    if (cameraPose.rows != 3 || cameraPose.cols != 1)
    {
        throw std::runtime_error("Invalid camera pose dimensions.");
    }

    // 提取参数
    const auto &camera = config[PARAMETERS_KEY][CAMERA_KEY];
    const auto &coarse = config[PARAMETERS_KEY]["positions"]["coarse"];
    const auto &fine = config[PARAMETERS_KEY]["positions"]["fine"];

    // 计算车体移动量（x,y,yaw）
    result.x = fine["x"].get<double>() - coarse["x"].get<double>();
    result.y = fine["y"].get<double>() - coarse["y"].get<double>();
    result.yaw = fine["yaw"].get<double>() - coarse["yaw"].get<double>();

    // 抓手移动量（基于相机姿态）
    result.gripper_y = cameraPose.at<double>(1) + camera["offset"]["y"].get<double>();
    result.gripper_z = cameraPose.at<double>(2) + camera["offset"]["z"].get<double>();

    // 抓手角度（假设来自相机姿态）
    double z = cameraPose.at<double>(2);
    result.gripper_pitch = atan2(cameraPose.at<double>(1), z != 0 ? z : 1e-6) * 180 / CV_PI;
    result.gripper_yaw = atan2(cameraPose.at<double>(0), z != 0 ? z : 1e-6) * 180 / CV_PI;

    return result;
}

int main()
{
    try
    {
        // 1. 加载配置
        json config = loadConfig("cfg.json");

        // 2. 使用PnP计算相机位姿
        cv::Mat cameraPose = computeCameraPose(config);

        // 3. 计算移动量
        Movement move = computeMovement(config, cameraPose);

        // 4. 输出结果
        std::cout << "Car Movement: x=" << move.x
                  << ", y=" << move.y
                  << ", yaw=" << move.yaw << "°\n";
        std::cout << "Gripper Movement: y=" << move.gripper_y
                  << ", z=" << move.gripper_z
                  << ", pitch=" << move.gripper_pitch
                  << ", yaw=" << move.gripper_yaw << "°\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}