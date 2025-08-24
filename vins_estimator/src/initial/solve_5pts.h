#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>

class MotionEstimator
{
  public:

    /// @brief 给定一组 3D 点的匹配对，计算两帧之间的相对旋转 R 和平移 T
    /// @param corres 3D 点的匹配对
    /// @param R 两帧之间的相对旋转
    /// @param T 两帧之间的相对平移
    /// @return 
    bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

  private:

    /// @brief 使用给定的旋转 R 和平移 t，评估左右图像中的 2D 点是否能成功三角化为 3D 点
    /// @param l 左图像中的 2D 点
    /// @param r 右图像中的 2D 点 
    /// @param R 预测的旋转矩阵
    /// @param t 预测的平移向量
    /// @return 表示三角化质量的分数
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);

    /// @brief 使用给定的本质矩阵 E，计算两帧之间的相对旋转 R 和平移 t，两种旋转和两种平移）
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};

