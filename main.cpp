#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 检查图像是否有效
bool checkImage(const Mat &image, const string &name)
{
    if (image.empty())
    {
        cout << "无法读取图像文件: " << name << endl;
        return false;
    }
    return true;
}

// 使用 ORB 特征点检测和描述子匹配找到靶标区域
Rect detectTargetRegion(const Mat &image, const Mat &templateImage)
{
    // 初始化 ORB 检测器
    Ptr<ORB> orb = ORB::create();

    // 检测关键点和描述子
    vector<KeyPoint> keypointsImage, keypointsTemplate;
    Mat descriptorsImage, descriptorsTemplate;
    orb->detectAndCompute(image, Mat(), keypointsImage, descriptorsImage);
    orb->detectAndCompute(templateImage, Mat(), keypointsTemplate, descriptorsTemplate);

    if (descriptorsImage.empty() || descriptorsTemplate.empty())
    {
        cout << "未检测到足够的特征点！" << endl;
        return Rect(); // 返回空矩形
    }

    // 使用 BFMatcher 进行特征点匹配
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptorsTemplate, descriptorsImage, matches);

    // 筛选最佳匹配点
    double minDist = 1e6, maxDist = 0;
    for (const auto &match : matches)
    {
        double dist = match.distance;
        minDist = min(minDist, dist);
        maxDist = max(maxDist, dist);
    }

    // 筛选出距离小于阈值的匹配点
    vector<Point2f> goodMatchesImage, goodMatchesTemplate;
    for (const auto &match : matches)
    {
        if (match.distance < max(2 * minDist, 30.0))
        {
            goodMatchesTemplate.push_back(keypointsTemplate[match.queryIdx].pt);
            goodMatchesImage.push_back(keypointsImage[match.trainIdx].pt);
        }
    }

    if (goodMatchesImage.size() < 4)
    {
        cout << "未找到足够的匹配点！" << endl;
        return Rect(); // 返回空矩形
    }

    // 计算单应性矩阵并提取靶标区域
    Mat homography = findHomography(goodMatchesTemplate, goodMatchesImage, RANSAC, 5.0);
    if (homography.empty())
    {
        cout << "无法计算单应性矩阵！" << endl;
        return Rect();
    }

    // 获取模板图像的角点
    vector<Point2f> templateCorners = {Point2f(0, 0), Point2f((float)templateImage.cols, 0),
                                       Point2f((float)templateImage.cols, (float)templateImage.rows),
                                       Point2f(0, (float)templateImage.rows)};
    vector<Point2f> imageCorners(4);
    perspectiveTransform(templateCorners, imageCorners, homography);

    // 计算包围矩形
    Rect roiRect = boundingRect(imageCorners);
    return roiRect;
}

int main(int argc, char **argv)
{
    try
    {
        // 1. 读取输入图像和模板图像
        if (argc < 3)
        {
            cout << "用法: " << argv[0] << " <目标图像路径> <模板图像路径>" << endl;
            return -1;
        }

        Mat image = imread(argv[1], IMREAD_COLOR);
        Mat templateImage = imread(argv[2], IMREAD_COLOR);

        if (!checkImage(image, argv[1]) || !checkImage(templateImage, argv[2]))
        {
            return -1;
        }

        // 2. 动态检测靶标区域
        Rect roiRect = detectTargetRegion(image, templateImage);
        if (roiRect.empty())
        {
            cout << "未检测到靶标区域！" << endl;
            return -1;
        }

        // 提取 ROI 区域
        Mat roi = image(roiRect);

        // 3. 动态检测靶标中心点
        Point2f center(roiRect.x + roiRect.width / 2.0, roiRect.y + roiRect.height / 2.0);
        cout << "靶标中心点坐标: (" << center.x << ", " << center.y << ")" << endl;

        // 4. 可视化结果
        rectangle(image, roiRect, Scalar(0, 255, 0), 2);                    // 绘制 ROI 矩形框
        circle(image, Point(center.x, center.y), 5, Scalar(0, 0, 255), -1); // 绘制靶标中心点

        imshow("Result", image);
        waitKey(0);
    }
    catch (const exception &e)
    {
        cout << "发生异常: " << e.what() << endl;
        return -1;
    }

    return 0;
}