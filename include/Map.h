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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"

#include <set>
#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);

    // 只是从set中删除了地图点的指针, 原先地图点占用的内存区域并没有得到释放
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);

    // 设置参考地图点，一般是指,设置当前帧中的参考地图点; 这些点将用于DrawMapPoints函数画图
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();      // 通知发生较大变化，用于回环中的闭环纠正、全局BA
    int GetLastBigChangeIdx();      // 获取上次map发生的最大改变次数(闭环纠正一次+1，全局BA一次+1)

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();     // 参考地图点指的是，当前帧中观测到的局部地图点

    long unsigned int MapPointsInMap();                 // 地图点个数
    long unsigned  KeyFramesInMap();                    // 关键帧个数

    long unsigned int GetMaxKFid();                     // 获取关键帧的最大id

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;               // 跟踪线程里的初始关键帧

    std::mutex mMutexMapUpdate;         // 地图更新互斥量

    // This avoids that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;     // 避免地图点id冲突

protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
