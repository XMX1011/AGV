#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <fstream>

#include "vulkan_utils.h"
#include "camera_capture.h"
#include "mechArmCtrl.h"
#include "laser_ranging.h"

using namespace std::this_thread;
using namespace std::chrono_literals;

bool checkNutsRemoved();
bool alignRack();
bool alignAxle();

// 使用python训练，导出onnx或者tensorrt，在c++中加载模型和使用

int main()
{
    auto start = std::chrono::system_clock::now();
    // 任务是使用vulkan加速，
    // 使用OpenCV或yolo等成熟方法
    // 进行四个主要任务
    
    // TODO: 这里的操作逻辑需要更改，暂时这样写

    // 去轮胎架上取新轮胎
    if (!alignRack())
    {
        std::cout << "机械爪未对准轮胎架" << std::endl;
        alert();
        return -1;
    }

    // 拆卸旧轮胎
    if (!checkNutsRemoved())
    {
        std::cout << "螺母未拆卸完全" << std::endl;
        alert();
        return -1;
    }
    // 等待AGV移动
    sleep(1000);
    return 0;
}
bool checkNutsRemoved()
{
    // 使用yolov8或者传统目标检测算法

    // 如果用yolov8，则需要制造数据集，改最后一层为二分类
    // 然后训练模型，导出onnx文件
    // 然后在c++中加载模型，进行目标检测
    // 检测在拆卸轮胎前，螺母是否已经拆完
    bool fullyRemoved = false;
    // TODO: 此处占位，之后实现

    //! 下面是传统目标检测算法的实现
    // 读取螺母图案
    cv::Mat nutPattern = cv::imread("../data/nut.jpg");
    // 打开相机，获取轮轴图像
    // 相机是MVTec Halcon的
    cv::VideoCapture cap(0);
    cv::Mat frame;
    cap >> frame;
    // 高斯滤波
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
    // 二值化
    cv::threshold(frame, frame, 127, 255, cv::THRESH_BINARY);
    // Hough变换
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(frame, lines, 1, CV_PI / 180, 100, 50, 10);
    // 边缘绘制
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::line(frame, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 2);
    }
    // 进行目标检测
    cv::Mat result;
    cv::matchTemplate(frame, nutPattern, result, cv::TM_CCOEFF_NORMED);
    cv::threshold(result, result, 0.8, 1, cv::THRESH_BINARY);
    cv::imshow("result", result);
    cv::waitKey(0);
    // 若检测到螺母未拆完，则返回false
    // 否则返回true
    if (cv::countNonZero(result) == 0)
    {
        fullyRemoved = true;
    }
    // 关闭窗口
    sleep(1);
    cv::destroyAllWindows();
    return fullyRemoved;
}

bool alignRack()
{
    // 判断机械爪的中轴是否和轮胎的中轴对齐
    // 若不对齐，则需要调整机械爪的位置
    // 包括给出建议的移动方向和距离
    // （像素，之后根据距离等再添加转换函数）
    // TODO: 此处占位，之后实现
    bool aligned = false;
    // 传统视觉方案
    cv::Mat rackPattern = cv::imread("../data/axle_on_rack.jpg");
    cv::Mat frame;
    cv::VideoCapture cap(0);
    cap >> frame;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::resize(frame, frame, cv::Size(640, 480));
    cv::Mat result;
    // 进行轮廓检测
    cv::Canny(frame, result, 50, 150);

    // 计算出轮胎中心点
    cv::Moments moments = cv::moments(result, false);
    cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
    cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), 2);

    // 计算相机中心点
    cv::Point2f cameraCenter(frame.cols / 2, frame.rows / 2);
    cv::circle(frame, cameraCenter, 5, cv::Scalar(0, 255, 0), 2);

    // 计算相机和轮胎中心点的横纵差值
    float dx = center.x - cameraCenter.x;
    float dy = center.y - cameraCenter.y;

    // 如果差值在一定范围内，则认为对齐
    if (abs(dx) < 10 && abs(dy) < 10)
    {
        aligned = true;
    }

    // 计算建议的移动方向和距离
    // 这里暂时用像素作为单位
    // 按x，y轴移动机械爪
    if (dx > 0)
    {
        // 向右移动
        // TODO: 这里需要根据距离等再添加转换函数
        std::cout << "向右移动" << dx << 'pixels' << std::endl;
    }
    else if (dx < 0)
    {
        // 向左移动
        // TODO: 这里需要根据距离等再添加转换函数
        std::cout << "向左移动" << dx << 'pixels' << std::endl;
    }
    if (dy > 0)
    {
        // 向下移动
        // TODO: 这里需要根据距离等再添加转换函数
        std::cout << "向下移动" << dy << 'pixels' << std::endl;
    }
    else if (dy < 0)
    {
        // 向上移动
        // TODO: 这里需要根据距离等再添加转换函数
        std::cout << "向上移动" << dy << 'pixels' << std::endl;
    }
    return aligned;
}

bool alignAxle()
{
    // 有激光测距模块
    // 判断机械爪的中心是否和车轴中心对齐
    // 若不对齐，则需要调整机械爪的位置
    // 使用机器学习或者传统视觉方案
    // 需求精度最高，1mm，可能更应该用yolo
    // 使用onnxruntime库或者TensorRT，加载模型
    // 包括给出建议的移动方向和距离
    // TODO: 此处占位，之后实现
    bool aligned = false;
    return aligned;
}
// The videos are:Weather Service: An overnight weather warning for a rather unusual meteorological event. Originally published on the Chainsawsuit Original YouTube channel on October 26, 2015.Contingency: A Lyndon Johnson-era PSA from the Department for the Preservation of American Dignity to be shown in the event the United States was conquered conventionally by an enemy force (or, if Fridge Logic comes into play, the end of the world was happening). Originally published on the Chainsawsuit Original YouTube channel on January 12, 2016.You Are On The Fastest Available Route: Dashcam footage with accompanying GPS audio, taken over roughly three-and-a-half hours in the early morning of November 21, 2014. Originally published on the Chainsawsuit Original YouTube channel on June 19, 2016.Station ID: The card for the start of the broadcast day. Lets you know that even if you don't watch, it doesn't matter — there are other receivers. Published on November 2, 2017, shortly after Local58 was given its own YouTube channel following Chainsawsuit Original's rebranding into FilmJoy.Show For Children: A black-and-white 1929 cartoon called Cadavre, about the titular character looking to find his loved one. Published on July 30, 2018.A Look Back: A montage of WCLV station idents from the 1930s to the present day, only to be interrupted by a second-party hack with a cryptic message (also showing 2- or 3-second snippets of the previous videos, and a few unrelated videos) making it a de facto trailer for the channel. Published on August 27, 2018.Real Sleep: A cult-esque instructional video informing the viewer about the truth of dreams and how to achieve real sleep, based on a real phenomenon. Published on December 19, 2018.Skywatching: An amateur video tour of nighttime constellations that ends up capturing more astrological phenomena than stars in the sky. Published on November 1, 2019.Digital Transition: A tribute from the network as the transition between analog and digital television is made, with glimpses of both the past and the future. Published on October 31, 2021.Close: By far the longest Local 58 video thus far, depicting a probe landing on a near-earth asteroid. Published October 31, 2022.NSSA-3 (atypical): a very short video showing...something...attempting to break through the station's signal. note Night Walk: A documentary about the "Woman in Profile", a local ghost story from the Mason County area. Published October 31, 2024.The popularity of these videos led to numerous imitators and thus defined a new genre of horror called Analog Horror, a term coined by Local 58 in the video Station ID.
// Gemini Home Entertainment is a Web Video Analog Horror anthology series by Remy Abode, who also created Morley Grove. The videos are presented in the form of short guides, documentaries or adverts distributed by the (fictitious) Gemini Home Entertainment video company. They depict a world where things went wrong, although the exact details of which are unclear.The videos include:"World's Weirdest Animals" - a short documentary about some of the weirdest animals living in Rural Minnesota."Storm Safety Tips" - an instructional video on how to prepare for storms and what to do during and after one."The Deep Blue" - a short documentary on maritime wildlife as well as the exploration of a mysterious portion of the ocean. (This video became private owing to Remy Abode's own dissatisfaction with the video's pace, its relative disconnect with the rest of the series, and its retroactive redundancy following a later entry which follows a similar concept.)"Artificial Computer Learning" - a demonstration guide by Regnad Computing on the development of the world's most advanced artificial intelligence."Our Solar System" - a documentary about our Solar System, presented with interesting facts."Camp Information Video" - a short information video showcasing the activities, accommodations, and mythos of the Moonlight Acres Family Camp."Lethal Omen Commercial" - an advert and short playthrough of a video game called Lethal Omen. The full game was later released online."Wilderness Survival Guide" - an instructional video by famous wildlife expert Jack Wylder on how to protect yourself in the wilderness."Sleep Image Visualizer" - a short guide how to use the Sleep Image Visualizer (SIV)."Games for Kids" - an instructional video featuring four games for groups of kids to play."Advanced Mining Vehicle" - a short informational video showcasing a new advancement in mining technology."Deep Root Disease" - an educational video on what the Deep Root Disease is and how to recognize it."Monthly Progress Report" - an informational video on Regnad Computing's latest project progress."Christmas Eve Party" - a video showing the 1985 Christmas Eve Party at a familiar location."Home Invasion Help" - a video showing what to do in the event of a home invasion."Crusader Probe Mission" - a video showing the journey and progress of the space probe Crusader 5."WRETCHED HANDS" - an inquiry regarding a bear first sighted at Moonlight Acres Camp in the 1940s."SHIFTING TENDONS" - a detailed look into the symptoms and pathology of late stage Deep Root Disease."OLD BONES" - a look into the diary of Glenn A. Arthur, the administrator of Moonlight Acres Camp during the 1930s.The series premiered on November 17, 2019, and is available for watching on its YouTube channel here.