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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);     // 将关键帧插入到新关键帧列表 mlNewKeyFrames，等待处理

    // Thread Synch，线程同步
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();                         // 释放该线程、释放当前还在缓冲区(新关键帧列表)的关键帧指针
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    // 队列中(缓冲区)等待插入的关键帧数目
    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:
    // 检查关键帧队列是否为空(是否有待处理的关键帧)
    bool CheckNewKeyFrames();

    // 处理列表中的关键帧，计算Bow，加速三角化新的MapPoints；
    // 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围；
    // 插入关键帧，更新Co-visibility图和Essential图
    void ProcessNewKeyFrame();

    // 关键帧与共视关键帧通过三角化恢复初的地图点
    void CreateNewMapPoints();

    // 检查新关键帧引入的地图点，剔除质量不好的点
    void MapPointCulling();

    // 检查并融合当前关键帧与两级共视帧重复的地图点
    void SearchInNeighbors();

    // 剔除局部冗余关键帧(90%以上的地图点可以被至少3个其他关键帧观测到)
    // 对当前关键帧的共视关键帧进行处理
    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);      // 计算三维向量v的反对称矩阵

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    // 存储当前关键帧生成的地图点列表，也是等待检查的地图点列表
    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;         // 当前LocalMapping线程是否已停止
    bool mbStopRequested;   // 当前LocalMapping线程是否收到停止请求
    bool mbNotStop;         // 当前LocalMapping线程是否未停止，默认为false
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
