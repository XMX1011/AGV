#include "coordinate_converter.h"
#include <iostream>

namespace cc
{

    /**
     * @brief 三维坐标转换，从不同z轴坐标系转换到相机坐标系等，或者反向转换
     * @param point 三维坐标
     * @param z_axis 坐标轴
     * @return 相机坐标系坐标
     */
    void calculate_position(double &Xf, double &Yf,
                            double V1, double V2,
                            double delta_f, double delta_r,
                            double R, double psi0,
                            double dt, double total_time)
    {
        double inner_integral = 0.0;
        double t = 0.0;
        const double Xf0 = Xf;
        const double Yf0 = Yf;

        while (t <= total_time)
        {
            // 计算内层积分 (V1*cosδf + V2*cosδr)/R
            const double current_inner = (V1 * cos(delta_f) + V2 * cos(delta_r)) / R;
            inner_integral += current_inner * dt;

            // 计算外层积分角度参数
            const double angle = delta_f + psi0 + inner_integral;

            // 更新外层积分
            Xf += V1 * cos(angle) * dt;
            Yf += V1 * sin(angle) * dt;

            t += dt;
        }
    }

};