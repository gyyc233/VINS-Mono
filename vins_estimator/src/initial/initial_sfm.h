#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

// SfM, Structure from Motion
// 使用了 Ceres Solver 进行非线性优化，用于在初始化阶段估计相机位姿和稀疏三维地图点

// 描述一个特征点在整个 SfM 中的信息
// 包含其在不同帧中的二维观测以及估算出的世界坐标系下的三维位置
struct SFMFeature {
    bool state;                    // 是否为有效特征点
    int id;                        // 特征点ID
    vector<pair<int,Vector2d>> observation; // 每一帧中该特征的观测（帧号+二维坐标）
    double position[3];            // 三维位置
    double depth;                  // 深度值
};

// 重投影误差残差块的优化
struct ReprojectionError3D
{
	// constructor 接受观测到的像素坐标
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	// 将世界坐标系下点投影到相机坐标系下并进行归一化投影
	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		// 将世界坐标系下的点 point 转换到相机坐标系下 p
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		// 归一化投影
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];

		// 计算重投影误差 预测-观测
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	// 创建一个 Ceres 的自动微分残差代价函数
	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  // ReprojectionError3D: 代价函数的实现结构体
	  // <误差模型, 残差维度, 参数1维度, 参数2维度, 参数3维度> 参考 operator() 函数项
	  // 2: 残差维度，表示每个误差项有两个输出（图像坐标u和v的误差）
	  // 4: 第一个可优化参数块的维度，这里是相机旋转（四元数表示，4维）
	  // 3: 第二个可优化参数块的维度，这里是相机平移（3维向量）
	  // 3: 第三个可优化参数块的维度，这里是三维空间点（x, y, z）
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();

	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:

	/// @brief 基于2d点像素-3d点对，借助cv::solvePnP重新估计相机间位姿
	/// @param R_initial 
	/// @param P_initial 
	/// @param i 
	/// @param sfm_f 
	/// @return 
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	/// @brief 给定两个相机位姿和对应的二维点，三角化得到三维点
	/// @note SVD分解求三角化 https://blog.csdn.net/Walking_roll/article/details/119984469
	/// @param Pose0 
	/// @param Pose1 
	/// @param point0 
	/// @param point1 
	/// @param point_3d 
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	
	/// @brief 对两帧之间的所有共视点进行三角化
	/// @param frame0 
	/// @param Pose0 
	/// @param frame1 
	/// @param Pose1 
	/// @param sfm_f 
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num; // 特征点数量
};