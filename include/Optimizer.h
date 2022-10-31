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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class LoopClosing;

class Optimizer
{
public:
    // 被全局BA调用，优化传入的关键帧位姿、地图点位置。
    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=nullptr, unsigned long nLoopKF=0,
                                 bool bRobust=true);

    // 全局优化(共视图Tracking,LoopClosing)。优化地图中所有关键帧位姿、所有地图点位置。调用BundleAdjustment。
    void static GlobalBundleAdjustment(Map* pMap, int nIterations= 5, bool *pbStopFlag= nullptr,
                                       unsigned long nLoopKF= 0, bool bRobust= true);

    // 局部BA优化(LocalMapping线程使用)。优化局部地图中的关键帧位姿 和 所有观测到的地图点3D位置。
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);

    // 优化当前帧位姿 (Tracking线程使用)，返回优化后的内点数。
    // 利用当前帧观测到的地图点，上一帧位姿作为初始值，最小化重投影误差。e = (u,v) - K(Tcw*Pw)
    int static PoseOptimization(Frame* pFrame);

    // if bFixScale is true, 6DoF optimization (stereo,rgb-d), 7DoF otherwise (mono)
    // 全局本质图优化(LoopClosing). 加入闭环约束, 优化所有关键帧位姿.
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgb-d), Sim3 otherwise (mono)
    // 优化两帧之间的Sim3变换(LoopClosing线程使用)，返回优化后的匹配点对数。
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, float th2, bool bFixScale);
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
