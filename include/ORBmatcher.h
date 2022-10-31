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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM2
{

// 处理数据关联问题
class ORBmatcher
{    
public:

    // 构造函数，nnration最优/次优
    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame key-points and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking) 投影地图点到当前帧，进行匹配跟踪（跟踪地图点）
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking) 将上一帧跟踪的地图点投影到当前帧，搜索匹配点（跟踪上一帧）
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in re-localisation (Tracking) 将关键帧观测到的(未匹配)地图点投影到当前帧中,进行匹配跟踪（用于重定位）
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound,
                           const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)用Sim3变换，将地图点投影到关键帧pKF(用于回环检测)
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints,
                            std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Re-localisation and Loop Detection。通过词袋，对关键帧的特征点进行跟踪（用于重定位、回环检测）
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    // 单目初始化中用于参考帧和当前帧的特征点匹配
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched,
                                std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    // 利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的匹配点对
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1, 通过Sim3变换，寻找KF1与KF2之间的匹配
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12,
                     const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    // 将地图点投影到关键帧中，搜索重复的地图点进行融合
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    // 通过Sim3变换，将地图点投影到关键帧中，搜索重复的地图点进行融合
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th,
             vector<MapPoint *> &vpReplacePoint);

public:

    static const int TH_LOW;        // 判断描述子距离时较低的阈值,主要用于基于词袋模型的加速匹配过程
    static const int TH_HIGH;       // 判断描述子距离较高的阈值,用于计算投影后能够匹配上的特征点的数目
    static const int HISTO_LENGTH;  // 判断特征点旋转用直方图的长度


protected:
    // 检查极线距离
    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                               const cv::Mat &F12, const KeyFrame *pKF);
    // 根据观察的视角来计算匹配时的搜索窗口半径大小
    float RadiusByViewingCos(const float &viewCos);

    // 筛选出在旋转角度差 落在直方图区间内 数量最多的 前三个bin的索引
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;            // 最优评分/次优评分
    bool mbCheckOrientation;    // 是否检查特征点方向
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
