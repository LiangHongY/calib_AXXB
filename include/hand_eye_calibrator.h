/*
 * @Author: ylh 
 * @Date: 2024-05-22 20:20:24 
 * @Last Modified by: ylh 2252512364@qq.com
 * @Last Modified time: 2025-08-31 16:11:15
 */

#ifndef HAND_EYE_CALIBRATOR_H
#define HAND_EYE_CALIBRATOR_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <filesystem>

struct TimedPose {
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

struct RelativeMotion {
    Eigen::Vector3d tA, tB;
    Eigen::Quaterniond qA, qB;
};

class HandEyeCalibrator {
public:
    HandEyeCalibrator();
    using Ptr = std::shared_ptr<HandEyeCalibrator>;
    int initParams(std::string& cfgPath);
    int calibProcess();

private:
    // 内部数据
    std::vector<TimedPose> _APoses;
    std::vector<TimedPose> _BPoses;
    std::vector<TimedPose> _BInterpolatedToA; // 根据A时间戳插值B所得队列
    std::vector<RelativeMotion> _relativeMotions;
    // 配置
    Eigen::Matrix4d _T_A_B_init;
    Eigen::Quaterniond _initQuat;
    Eigen::Vector3d _initTrans;
    int _skip;
    std::string _saveResultFilePath;
    std::string _APosesPath, _BPosesPath;
    bool _onlyOptimizeRotation;
    int _maxIterations;
    double _residualThreshold;
    double _timeToleranceSeconds;

    Eigen::Quaterniond _estQuat;
    Eigen::Vector3d _estTrans;

    TimedPose interpolatePose(const TimedPose& a, const TimedPose& b, double alpha);
    bool slerpSafe(const Eigen::Quaterniond& qa, const Eigen::Quaterniond& qb, double t, Eigen::Quaterniond& out);
    Eigen::Matrix3d quatToMat(const Eigen::Quaterniond& q);

    // 读取文件，格式：timestamp(s) x y z qx qy qz qw (qx..qw order expected)
    bool loadAPosesFromFile(const std::string& filePath);
    bool loadBPosesFromFile(const std::string& filePath);

    // 对时并插值：会把B插值到A时间（默认行为）
    void timeAlignAndBuildPairs();

    // 构建相对变换对（默认使用相邻帧）AiX = BiX ,注意AX=XB使用的是各自相对增量
    void buildRelativeMotions(int skip = 1);

    // 执行优化，initQuatInit (w,x,y,z) optional (if not provided identity used), initTrans optional
    bool calibrate(const Eigen::Quaterniond& initQuat = Eigen::Quaterniond::Identity(),
                   const Eigen::Vector3d& initTrans = Eigen::Vector3d::Zero());

    // 将结果保存到文件
    bool saveResultToFile(const std::string& filePath) const;

    void clear();

    void printSummary() const;
};

#endif // HAND_EYE_CALIBRATOR_H

