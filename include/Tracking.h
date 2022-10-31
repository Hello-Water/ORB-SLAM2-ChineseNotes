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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    // 处理不同传感器输入图像（转换为灰度图），并且调用Track()函数，返回Tcw
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    // 设置局部建图、回环检测、可视化器
    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings，修改配置文件路径
    // The focal length should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal length
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping, and you only want to localize the camera.
    // 告知系统只进行跟踪模式（只确定相机位姿）
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,      // 系统未准备好，一般是启动后在加载配置文件和词典文件
        NO_IMAGES_YET = 0,          // 当前无图像输入
        NOT_INITIALIZED = 1,        // 有图像，但未完成初始化
        OK = 2,                     // 正常工作
        LOST = 3                    // 跟踪丢失
    };

    eTrackingState mState;                  // 当前帧跟踪状态
    eTrackingState mLastProcessedState;     // 上一帧跟踪状态

    // Input sensor:MONOCULAR, STEREO, RGB-D
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;        // 双目/RGB-D时，为左目图像灰度图

    // Initialization Variables (Monocular)     // 单目初始化参数
    std::vector<int> mvIniLastMatches;          // 初始化时，上次的匹配
    std::vector<int> mvIniMatches;              // 初始化时，前两帧的匹配
    std::vector<cv::Point2f> mvbPrevMatched;    // 初始化时，参考帧的去畸变特征点
    std::vector<cv::Point3f> mvIniP3D;          // 初始化得到的3D地图点
    Frame mInitialFrame;                        // 初始化，第一帧

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    // 恢复相机位姿，原始存储信息为每帧图像的 参考关键帧 及 图像到参考关键帧的相对位姿变换
    list<cv::Mat> mlRelativeFramePoses;         // 图像到参考关键帧的相对位姿 Tlr
    list<KeyFrame*> mlpReferences;              // 参考关键帧
    list<double> mlFrameTimes;                  // 图像帧的时间戳
    list<bool> mlbLost;                         // 是否跟丢的标志

    // True if local mapping is deactivated, and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D，主要产生初始地图
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();             // 单目初始化（匹配、三角化）
    void CreateInitialMapMonocular();           // 单目生成初始化地图

    // 跟踪相关函数
    void CheckReplacedInLastFrame();    // 检查上帧中是否存在用于替换的地图点，存在则替换相应地图点
    bool TrackReferenceKeyFrame();      // 跟踪 参考关键帧 的地图点，跟踪到的点数>10，则说明跟踪到，返回true
    void UpdateLastFrame();             // 更新上一帧（位姿、根据深度较小的点产生新的地图点）
    bool TrackWithMotionModel();        // 匀速运动模型跟踪上一帧(新)地图点，跟踪到的点数>10，返回true

    bool Relocalization();

    // 更新局部地图、局部地图点、局部关键帧
    void UpdateLocalMap();
    void UpdateLocalPoints();           // 局部关键帧观测的地图点
    void UpdateLocalKeyFrames();        // 当前帧的父子关键帧、一级、二级共视关键帧

    // 跟踪 局部地图(点)，跟踪点数>30，跟踪成功，返回true
    bool TrackLocalMap();
    void SearchLocalPoints();           // (新增的)局部地图点投影到当前帧进行匹配

    // 添加新的关键帧
    bool NeedNewKeyFrame();             // 是否需要新的关键帧(跟踪到rKF的点较少 且 局部建图中的关键帧空闲(时间间隔合适))
    void CreateNewKeyFrame();           // 创建新的关键帧

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do re-localization to recover
    // "zero-drift" localization to the map.
    bool mbVO;      // 仅定位模式下的参数：匹配不到地图点时为true；匹配到足够的地图点，则为false，继续跟踪。

    // Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    // ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;    // 单目初始化使用

    // BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initialization (only for monocular)
    Initializer* mpInitializer;

    // Local Map
    KeyFrame* mpReferenceKF;    // 参考关键帧（当前帧的最大一级共视帧）
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    // Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // Map
    Map* mpMap;

    // Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;      // 去畸变参数
    float mbf;              // baseline * focal_length

    // New KeyFrame rules (according to fps)
    // 关键帧最大/小时间间隔，与帧率有关
    int mMinFrames;     // 0
    int mMaxFrames;     // fps

    // Threshold for close/far points,
    // seen as close by the stereo/RGB-D sensor are considered reliable
    // and inserted from just one frame. Far points require a match in two keyframes.
    float mThDepth;         // 远/近点阈值：近点可信度较高(单帧即可)；远点要求在两个关键帧中得到匹配

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depth-map values are scaled.
    float mDepthMapFactor;  // 深度图缩放因子

    // Current matches in frame
    int mnMatchesInliers;   // 当前帧的匹配内点

    // Last Frame, KeyFrame and Re-localisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    // Motion Model
    cv::Mat mVelocity;

    // Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    // 临时的地图点,用于提高双目和RGB-D相机的帧间效果,用完之后就扔了
    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
