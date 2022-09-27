/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{
// ORB中以cv::Mat为基本存储结构，到g2o、Eigen需要一个转换
class Converter
{
public:
    // 描述子矩阵到单行的描述子向量的转换，转换后的结果就是把cv::Mat的每一行直接串联起来
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    // 将cv::Mat格式存储的位姿统一转换成为 g2o::SE3Quat 格式
    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
//    static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

    // 将各种格式转换成为cv::Mat存储
    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);

    // 组合旋转矩阵和平移向量来构造SE3
    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

    // 将给出的数据以Eigen库中对应的数据类型存储
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

    // 将给定的cv::Mat类型的旋转矩阵转换成以vector<float>类型表示的四元数
    static std::vector<float> toQuaternion(const cv::Mat &M);
};

}// namespace ORB_SLAM

#endif // CONVERTER_H
