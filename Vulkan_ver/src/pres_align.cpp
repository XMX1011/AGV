#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <vulkan/vulkan.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

static Logger gLogger;