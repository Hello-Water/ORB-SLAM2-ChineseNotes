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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
// 单目初始化（不用于双目、RGB-D）
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame，确定参考帧，构造初始化器
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);


private:
    // RANSAC计算H、F矩阵，RANSAC模型为重投影误差
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

    // 计算H、F矩阵
    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    // 对给定的H、F矩阵打分，卡方检验
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    // 从基础矩阵F、单应矩阵H，求解R21，t21，并三角化出3D点
    // 取名为ReconstructFromH()更好
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 由匹配点对、投影矩阵，计算三维点坐标
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    // 归一化特征点，不同坐标下的H矩阵不同，同一幅图像进行相同的变换，不同图像进行不同变换，缩放尺度让噪声对图像的影响在同一数量级
    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    // cheirality check，三角化出的3d点应该有正深度
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
    // 分解本质矩阵，E矩阵由F矩阵结合内参获得，分解后得到 [R1,t]、[R1,-t]、[R2,t]、[R2,-t] 四组解
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Reference to Current
    vector<Match> mvMatches12;
    vector<bool> mvbMatched1;

    // Calibration
    cv::Mat mK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;

    // Ransac max iterations
    int mMaxIterations;

    // Ransac sets，内层为每次迭代需要的点，外层为迭代次数
    vector<vector<size_t> > mvSets;   

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
