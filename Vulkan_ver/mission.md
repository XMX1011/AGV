# AGV

## 任务描述
任务主要是自动化
从轮胎架子上取出轮胎
（使用视觉方案对齐轮胎中心，
主要是轮胎所在平面上的横纵两个参数x,y），
从车体上拆除旧轮胎
（首先需要检测轮胎的螺栓上的螺母是否拆卸完全，
然后需要不仅是横纵两个参数x,y,
还有抓取轮胎的机械抓手的俯仰角度），
装配新轮胎
(需要检测轮胎中心是否和车轴中心对齐（x,y），
轮胎的轮毂上的螺栓孔是否与车轴上的螺栓对齐（旋转角度），
抓取轮胎的机械抓手的俯仰角度，车体左右的旋转角度
)
将轮胎放回备胎架上（和取出轮胎类似，
使用视觉方案对齐轮胎中心，
主要是轮胎所在平面上的横纵两个参数x,y
）

## 总共四个步骤，和6个主要参数

### 步骤
1. 从轮胎架上取出轮胎
2. 拆除旧轮胎
3. 装配新轮胎
4. 把旧轮胎放回备胎架

### 主要参数
1. 轮胎所在平面的横向参数 x
2. 轮胎所在平面的纵向参数 y
3. 机器爪平面和轮胎平面的距离 z
4. 机器爪平面相对轮胎平面的俯仰角度 theta
5. 机器爪平面相对轮胎平面的横向旋转角度 phi
6. 轮胎的轮毂螺栓孔与车轴螺栓的旋转角度 psi

## 系统架构
1. **数据采集模块**
   - **主相机**：500万像素黑白相机，安装在轮胎拆卸机构中心，用于对齐识别。相机均采用MVTec Halcon
   - **辅助相机**：200万像素黑白相机，安装位置相对主相机较向下，用于辅助检测轮胎的旋转角度是否对齐。
   - **激光测距**：四个激光测距仪，分别用于测量俯仰、左右角度和机械距目标的距离。

2. **预处理模块**
   - **图像预处理**：对主相机和辅助相机采集的图像进行灰度化、去噪等预处理操作。
   - **激光测距数据处理**：对激光测距测得的数据进行滤波和校正，提取有用的距离和角度信息。

3. **深度学习模型**
   - **目标检测模型**：使用YOLO、SSD或Faster R-CNN等目标检测模型，检测轮胎的位置和姿态。
   - **姿态预测模型**：使用轻量级的CNN模型或回归模型，预测轮胎的俯仰角、旋转角度和左右角度。这些模型可以是基于Transformer的模型，如EfficientNet或MobileNet，以提高预测速度和性能。

4. **传统CV模块**
   - **特征匹配**：使用SIFT或SURF等特征点检测与匹配技术，进行更精确的对齐。
   - **Hough变换**：检测轮胎边缘或轮廓，计算轮胎中心与目标位置的偏移量。
   - **螺母检测**：使用Hough圆变换或边缘检测技术，检测螺栓上的螺母是否拆卸完全。

5. **融合模块**
   - **数据融合**：将深度学习模型的预测结果与激光雷达数据进行融合，生成最终的对齐和姿态调整参数。
   - **卡尔曼滤波器**：使用卡尔曼滤波器对深度学习模型的预测结果和激光雷达数据进行融合，提高姿态预测的稳定性和精度。
   - **控制指令生成**：根据融合后的数据生成电机的控制指令，调整轮胎装卸机构的位置和姿态。

6. **控制模块**
   - **电机控制**：根据生成的控制指令，驱动电机进行精确的对齐和调整。
   - **反馈机制**：通过实时监控和反馈机制，确保对齐和调整的精度和稳定性。



## 额外的部分笔记
1. 需要自动调整ROI提升精度，相机提供了相应功能
> 相机可以通过Image Format Control 属性下Region Selector 相关参数进行ROI设置，
如图11-3所示。 
* Width：ROI区域横向的分辨率 
* Height：ROI 区域纵向的分辨率 
* Offset X：ROI 区域左上角起点位置的横坐标 
* Offset Y：ROI 区域左上角起点位置的纵坐标 
2. 曝光调整
* 有补光灯，补光灯亮度可调

3. 激光测距模块现在还没搞清楚怎么弄，还说让我调研采购

4. 像素和距离(单位mm)之间的转换关系式得学一下