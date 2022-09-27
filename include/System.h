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


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

// 定义ORB-SLAM2的主线程（或者称之为系统）结构，其他的各个模块都是在这里开始被调用的。
// 跟踪(单目、双目、RGB-D)、激活/关闭局部建图线程、重置系统、关闭系统、保存轨迹等
class System
{
public:
    // Input sensor
    enum eSensor {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    // 构造函数，初始化SLAM系统，启动局部建图、回环、可视化线程。
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);

    // 三种运动跟踪接口（双目、RGB-D、单目），输入图像会转化为灰度图(CV_8U)进行处理。
    // 对于双目系统，双目图像必须经过同步、校准才能输入到系统； 对于RGB-D系统，深度图与RGB图必须经过配准
    // 跟踪器返回估计的相机位姿，跟踪失败则返回NULL

    // Process the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    // Process the given rgb-d frame. Depth-map must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depth-map: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    // Process the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    // 激活定位模式，只对相机位姿进行跟踪，停止局部建图线程
    void ActivateLocalizationMode();

    // This resumes local mapping thread and performs SLAM again.
    // 关闭定位模式，恢复局部建图线程，再次执行SLAM系统
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function，
    bool MapChanged();      // 在回环、全局BA后（上次调用此函数后），地图是否发生了较大变化

    // Reset the system (clear map)
    void Reset();           // 复位系统，清除地图

    // All threads will be requested to finish. It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();        // 请求关闭所有线程，并等待所有线程结束。在保存轨迹前调用。

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    // 保存（双目或RGB-D）相机轨迹为TUM格式，不适用于单目相机。在此之前要先调用Shutdown()函数。
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    // 保存（单目、双目、RGB-D）相机关键帧轨迹为TUM格式。在此之前要先调用Shutdown()函数。
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    // 保存（双目或RGB-D）相机轨迹为KITTI格式，不适用于单目相机。在此之前要先调用Shutdown()函数。
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGB-D)
    // 在跟踪之后，获取跟踪状态、跟踪到的地图点、跟踪到的特征点等信息
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

private:

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (re-localization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs re-localization if tracking fails. 跟踪失败会执行重定位。
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;      // 搜索每个新关键帧的闭环，如果有闭环，则执行位姿图优化、全局BA（新的线程）

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    // 系统在主进程中进行运动追踪线程，创建局部建图线程、回环检测线程、查看器线程。
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    std::mutex mMutexState;
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;

};

}// namespace ORB_SLAM

#endif // SYSTEM_H
