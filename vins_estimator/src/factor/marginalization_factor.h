#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

// 用于实现部分边缘功能，主要用于在滑动窗口优化中将一些旧的状态变量“边缘化”出去
// 从而控制计算复杂度并保持状态估计的实时性

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    // _cost_function _loss_function：定义残差计算与其权重
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // 指向与该残差相关的参数数据的指针
    std::vector<int> drop_set; // 表示哪些参数要被边缘化（丢弃）

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; // 在evaluate中计算的雅可比矩阵
    Eigen::VectorXd residuals; // 在evaluate中计算的残差

    // 将全局维度（如 SE(3) 的 7 维）转换为局部切空间维度（如 6 维）
    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

// 用于多线程并行处理边缘化过程
struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors; // 分配给当前线程处理的残差块子集
    Eigen::MatrixXd A; // 线性化后的雅可比矩阵
    Eigen::VectorXd b; // 线性化后的残差
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 管理所有残差块，并准备边缘化所需的数据结构
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); // 添加一个新的残差块

    // 得到每次IMU和视觉观测对应的参数块，雅克比矩阵，残差值
    void preMarginalize();

    // 开启多线程构建信息矩阵H和b ，同时从H,b中恢复出线性化雅克比和残差
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift); // 根据地址获取当前参数的值

    std::vector<ResidualBlockInfo *> factors; // 所有残差块列表

    //这里将参数块分为Xm,Xb,Xr,Xm表示被marg掉的参数块，Xb表示与Xm相连接的参数块，Xr表示剩余的参数块
    //那么m=Xm的localsize之和，n为Xb的localsize之和，pos为（Xm+Xb）localsize之和

    int m, n; // m 表示要marg掉的变量个数，所有与将被marg掉变量有约束关系的变量的localsize之和

    // global size 将被marg掉的约束边相关联的参数块，即将被marg掉的参数块以及与它们直接相连的参数快
    std::unordered_map<long, int> parameter_block_size; //global size <参数块地址，参数块的global size>，参数块包括xm和xb
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size <参数块地址，参数块排序好后的索引>，对参数块进行排序，xm排在前面，xb排成后面，使用localsize
    std::unordered_map<long, double *> parameter_block_data; // <参数块地址，参数块数据>，需要注意的是这里保存的参数块数据是原始参数块数据的一个拷贝，不再变化，用于记录这些参数块变量在marg时的状态

    // 保留下的参数信息
    std::vector<int> keep_block_size; //global size <保留下来的参数块地址，参数块的globalsize>
    std::vector<int> keep_block_idx;  //local size <保留下来的参数块地址，参数块的索引>，保留下来的参数块是xb
    std::vector<double *> keep_block_data;

    // 用于优化的线性化矩阵
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

// 该类是优化时表示上一步边缘化后保留下来的先验信息代价因子，变量marginalization_info保存了类似约束测量信息
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);

    //调用cost_function的evaluate函数计算残差 和 雅克比矩阵
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    // 指向包含所有边缘化数据的 MarginalizationInfo 对象
    MarginalizationInfo* marginalization_info;
};
