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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"

namespace ORB_SLAM2
{

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, bool bFixScale = true);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);
    // RANSAC迭代求解Sim3
    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:
    // 计算3D点矩阵P(列向量)的质心C、去质心矩阵Pr
    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    // 计算 P2(候选闭环关键帧的3D点) 到 P1(当前关键帧的3D点) 的Sim3变换
    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

    // 根据Sim3变换，互相投影，根据投影误差检查内点
    void CheckInliers();

    // Sim3投影 (Tcw为Sim3变换矩阵)
    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    // 根据相机内参投影，相机坐标系到像素坐标系转换
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    KeyFrame* mpKF1;        // 当前关键帧
    KeyFrame* mpKF2;        // 候选闭环关键帧

    std::vector<cv::Mat> mvX3Dc1;           // 当前关键帧的匹配地图点 在当前关键帧下的相机坐标
    std::vector<cv::Mat> mvX3Dc2;           // 候选闭环关键帧的匹配地图点 在闭环关键帧下的相机坐标
    std::vector<MapPoint*> mvpMapPoints1;   // 当前关键帧的匹配地图点
    std::vector<MapPoint*> mvpMapPoints2;   // 候选闭环关键帧的匹配地图点
    std::vector<MapPoint*> mvpMatches12;    // 当前关键帧与闭环关键帧 匹配点对 id
    std::vector<size_t> mvnIndices1;        // 当前关键帧的匹配地图点id
    std::vector<size_t> mvSigmaSquare1;     // 未使用
    std::vector<size_t> mvSigmaSquare2;     // 未使用
    std::vector<size_t> mvnMaxError1;       // 当前关键帧中的某个特征点所允许的最大不确定度(和所在的金字塔图层有关)
    std::vector<size_t> mvnMaxError2;       // 候选闭环关键帧中的某个特征点所允许的最大不确定度(和所在的金字塔图层有关)

    int N;                                  // 可靠匹配点对数
    int mN1;                                // 当前关键帧与闭环关键帧，BoW匹配点对数

    // Current Ransac Estimation
    cv::Mat mR12i;                  // 某次RANSAC得到的旋转矩阵 R12
    cv::Mat mt12i;                  // 某次RANSAC得到的平移向量 t12
    float ms12i;                    // 某次RANSAC得到的尺度因子 s12
    cv::Mat mT12i;                  // 某次RANSAC得到的Sim3变换矩阵 T12
    cv::Mat mT21i;                  // Sim3变换矩阵的逆
    std::vector<bool> mvbInliersi;  // 某次RANSAC迭代，标记内点的vector
    int mnInliersi;                 // 某次RANSAC迭代，内点的数量

    // Current Ransac State
    int mnIterations;                   // 已经RANSAC迭代的次数
    std::vector<bool> mvbBestInliers;   // 历次迭代中，内点最多的那次标记
    int mnBestInliers;                  // 历次迭代中，最多的内点数
    cv::Mat mBestT12;                   // 历次迭代中，最佳的Sim3变换矩阵 T12
    cv::Mat mBestRotation;              // 历次迭代中，最佳的旋转矩阵
    cv::Mat mBestTranslation;           // 历次迭代中，最佳的平移向量
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGB-D case
    bool mbFixScale;                    // 双目/RGB-D用固定尺度1

    // Indices for random selection
    std::vector<size_t> mvAllIndices;   // 所有有效匹配点的id(当前关键帧地图点id)

    // Projections
    std::vector<cv::Mat> mvP1im1;       // 当前关键帧的地图点相机坐标 在当前关键帧上的投影像素坐标
    std::vector<cv::Mat> mvP2im2;       // 候选闭环关键帧的地图点相机坐标 在候选闭环镇上的投影像素坐标

    // RANSAC probability
    double mRansacProb;                 // 计算RANSAC理论迭代次数时用到的概率

    // RANSAC min inliers
    int mRansacMinInliers;              // 退出RANSAC所需的 最少内点数

    // RANSAC max iterations
    int mRansacMaxIts;                  // RANSAC最大迭代次数

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;                        // 当前帧的内参矩阵
    cv::Mat mK2;                        // 候选闭环帧的内参矩阵

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
