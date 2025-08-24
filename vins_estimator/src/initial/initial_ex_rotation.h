#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
// 估计imu与相机之间的外参旋转

class InitialEXRotation
{
public:
	InitialEXRotation();

    /// @brief 使用视觉点对和 IMU 提供的旋转增量来估计相机与 IMU 之间的旋转外参
    /// @param corres 视觉帧中匹配的 3D 点对（可能是三角化得到的）
    /// @param delta_q_imu IMU 提供的两帧之间的旋转四元数
    /// @param calib_ric_result 输出估计得到的旋转矩阵
    /// @return 
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:

	/// @brief 使用点对计算两帧之间的相对旋转矩阵
	/// @param corres 
	/// @return 
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

    /// @brief 对给定的旋转和平移进行三角化，评估三维点的有效性（深度是否合理）
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;

    vector< Matrix3d > Rc; // 每一帧的视觉旋转估计
    vector< Matrix3d > Rimu; // 每一帧的IMU旋转估计
    vector< Matrix3d > Rc_g; // 通过imu旋转与imu-camera外参估计的相机帧间旋转估计
    Matrix3d ric; // 相机到IMU的旋转矩阵
};
