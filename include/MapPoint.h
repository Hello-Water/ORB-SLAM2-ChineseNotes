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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Frame;
class Map;

// 地图点类
class MapPoint
{
public:
    // 给定地图点3D坐标 及 keyframe 构造地图点（单双目初始化、创建新的地图点、双目创建新的关键帧）
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    // 给定地图点3D坐标 及 frame 构造地图点（双目更新图像帧），idxF是MapPoint在Frame中对应特征点的编号
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);           // 设置世界坐标系下地图点位姿
    cv::Mat GetWorldPos();                          // 获取世界坐标系下地图点位姿

    cv::Mat GetNormal();                            // 获取当前地图点的平均观测方向
    KeyFrame* GetReferenceKeyFrame();               // 获取生成当前地图点的参考关键帧

    std::map<KeyFrame*,size_t> GetObservations();   // 获取观测到当前地图点的<关键帧序列,特征点>关键帧
    int Observations();                             // 获取当前地图点的被观测次数

    // 能共同观测到某些地图点的关键帧是共视关键帧
    void AddObservation(KeyFrame* pKF,size_t idx);  // 添加观测，哪个关键帧的哪个特征点能观测到该地图点
    void EraseObservation(KeyFrame* pKF);           // 删除某个关键帧对当前地图点的观测

    int GetIndexInKeyFrame(KeyFrame* pKF);          // 获取观测到当前地图点的关键帧中特征点的索引
    bool IsInKeyFrame(KeyFrame* pKF);               // 查看某个关键帧是否看到了当前地图点

    void SetBadFlag();                              // 告知可以观测到该MapPoint的Frame,该地图点已删除
    bool isBad();                                   // 判断当前地图点是否是bad

    void Replace(MapPoint* pMP);                    // 在形成闭环时，会更新KeyFrame和MapPoint之间的关系（更新地图点）
    MapPoint* GetReplaced();                        // 获取用于替换的地图点（不会立即删除）

    void IncreaseVisible(int n=1);                  // 增加可观测帧数，该MapPoint在某些帧的视野范围内
    void IncreaseFound(int n=1);                    // 能找到该地图点的帧数，地图点可以匹配到特征点
    float GetFoundRatio();                          // mnFound/mnVisible
    inline int GetFound(){                          // 被找到的次数？
        return mnFound;
    }

    // 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前地图点的最适合的描述子
    // 先获得当前地图点的所有描述子，然后计算每个描述子与其他描述子之间的距离，最好的描述子与其他描述子具有最小的距离中值
    void ComputeDistinctiveDescriptors();           // 计算具有代表性的描述子

    cv::Mat GetDescriptor();                        // 获取当前地图点的描述子，最好的特征点描述子

    void UpdateNormalAndDepth();                    // 更新平均观测方向 及 观测距离范围

    float GetMinDistanceInvariance();               // 金字塔尺度不变下的最小距离（光心与地图点之间）
    float GetMaxDistanceInvariance();               // 最大距离
    int PredictScale(const float &currentDist, KeyFrame*pKF);   // 根据当前距离、关键帧来预测尺度
    int PredictScale(const float &currentDist, Frame* pF);      // 双目预测尺度

public:
    long unsigned int mnId;                         // MapPoint的全局ID
    static long unsigned int nNextId;               // mnId = nNextId++
    long int mnFirstKFid;                           // 创建该MapPoint的第一个关键帧id
    long int mnFirstFrame;                          // 双目中，创建该MapPoint的第一帧id
    int nObs;                                       // 观测到该地图点的相机数目，单目+1，双目/RGBD +2

    // Variables used by the tracking
    float mTrackProjX;                              // 当前地图点投影到某帧上的坐标
    float mTrackProjY;
    float mTrackProjXR;                             // 当前地图点投影到某帧上的坐标（右目）
    bool mbTrackInView;                             // 该地图点是否在当前帧视野内（是否可以投影）
    int mnTrackScaleLevel;                          // 跟踪该地图点时的尺度水平
    float mTrackViewCos;                            // 跟踪到地图点时，图像帧的视角（观测向量与平均观测向量的夹角余弦）
    long unsigned int mnTrackReferenceForFrame;     // 地图点有效时，跟踪到的当前帧id，防止地图点重复添加至mvpLocalMapPoints
    long unsigned int mnLastFrameSeen;              // 观测到该地图点的上一帧图像id，决定是否进行isInFrustum判断

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;               // TODO...参与局部BA的关键帧索引id？
    long unsigned int mnFuseCandidateForKF;         // TODO...用于地图点融合的候选关键帧id？

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;             // TODO...某关键帧中参与到回环的地图点索引？
    long unsigned int mnCorrectedByKF;              // TODO...参与到回环的关键帧索引？
    long unsigned int mnCorrectedReference;         // TODO...该地图点对应到修正后的关键帧索引？
    cv::Mat mPosGBA;                                // 全局BA优化后的地图点位姿
    long unsigned int mnBAGlobalForKF;              // 参加全局BA优化的地图点对应的KeyFrame索引


    static std::mutex mGlobalMutex;                 // 全局BA中，对当前地图点进行操作时的互斥量

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;                             // 地图点在世界坐标系下的坐标

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;      // 观测到该地图点的<关键帧，对应特征点id>

     // Mean viewing direction
     cv::Mat mNormalVector;                         // 该地图点的平均观测方向，用于判断点是否在可视范围内

     // Best descriptor to fast matching
     // 每个3D点也有一个描述子，但是这个3D点对应多个二维特征点，从中选择一个最有代表性的特征点描述子
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;                             // 创建该地图点的关键帧

     // Tracking counters
     int mnVisible;                                 // 观测到该地图点的帧数
     int mnFound;                                   // 该地图点在mnFound帧中有匹配的特征点

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;                                    // bad地图点
     MapPoint* mpReplaced;                          // 被替换掉的地图点

     // Scale invariance distances
     float mfMinDistance;                           // 金字塔尺度不变下的最小距离（光心与地图点之间）
     float mfMaxDistance;                           // 最大距离

     Map* mpMap;                                    // 地图点所属的地图

     std::mutex mMutexPos;                          // 对当前地图点位姿进行操作时的互斥量
     std::mutex mMutexFeatures;                     // 对当前地图点的特征信息（参考关键帧）进行操作时的互斥量
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
