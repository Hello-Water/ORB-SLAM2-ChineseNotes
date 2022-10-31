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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:
    // 自定义数据类型ConsistentGroup，pair<"一致组"关键帧集合，"一致组"序号>
    typedef pair<set<KeyFrame*>, int> ConsistentGroup;
    // map<关键帧，位姿，重载KeyFrame*排序，Eigen对齐>，       std::less<KeyFrame*>重载KeyFrame*类型的比较函数
    typedef map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3>> > KeyFrameAndPose;

public:
    // 构造函数。双目/RGB-D中 bFixScale=true，意味着尺度因子s=1；单目bFixScale=false，需要进行Sim3计算
    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    // 将关键帧插入到 等待被闭环检测的关键帧列表
    void InsertKeyFrame(KeyFrame *pKF);

    // 请求重置闭环检测
    void RequestReset();

    // This function will run in a separate thread
    // 全局BA，在另外一个独立的线程
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    // 查看当前是否在运行全局BA，闭环纠正时调用
    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    // 查看是否已完成 全局BA
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();       // 请求结束当前闭环线程

    bool isFinished();          // 查看当前闭环线程是否已经结束

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool CheckNewKeyFrames();   // 查看mlpLoopKeyFrameQueue列表中是否有等待被闭环检测的关键帧

    bool DetectLoop();          // 检测回环，返回是否检测成功

    bool ComputeSim3();         // 计算待闭环帧 与 候选闭环帧群 的Sim3变换，确定闭环帧及对应的Sim3变换

    // 将闭环匹配上的 关键帧及其共视关键帧 观测到的地图点 投影到当前关键帧，进行匹配融合
    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();         // 闭环纠正。调用SearchAndFuse()

    void ResetIfRequested();    // 如果有复位指令，进行闭环线程复位
    bool mbResetRequested;      // 是否有复位指令
    std::mutex mMutexReset;

    bool CheckFinish();     // 检查是否有结束闭环线程请求
    void SetFinish();       // 设置当前闭环线程结束
    bool mbFinishRequested; // 是否有结束请求 标志位
    bool mbFinished;        // 是否结束 标志位
    std::mutex mMutexFinish;

    Map* mpMap;             // 全局地图
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;      // 等待被闭环检测的关键帧列表
    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;              // 共视一致性阈值，构造函数将其设置为3

    // Loop detector variables
    KeyFrame* mpCurrentKF;      // 当前处理的待闭环关键帧
    KeyFrame* mpMatchedKF;      // 当前关键帧的闭环关键帧
    std::vector<ConsistentGroup> mvConsistentGroups;            // 历史中满足一致性的群vector <关键帧集合，一致性程度>
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;       // 具有足够一致性的候选闭环帧vector，一致性程度>=3
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;              // 当前关键帧的共视帧vector
    std::vector<MapPoint*> mvpCurrentMatchedPoints;             // 当前关键帧 与 闭环帧的匹配地图点vector
    std::vector<MapPoint*> mvpLoopMapPoints;                    // 闭环关键帧及其共视帧的有效地图点vector
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;                            // 上一次闭环帧的id

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;          // GBA是否正在运行标志位
    bool mbFinishedGBA;         // GBA是否已结束的标志位
    bool mbStopGBA;             // 是否停止GBA的标志位
    std::mutex mMutexGBA;

    std::thread* mpThreadGBA;

    bool mbFixScale;            // Fix scale in the stereo/RGB-D case

    bool mnFullBAIdx;           // 记录进行了多少次全局BA
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
