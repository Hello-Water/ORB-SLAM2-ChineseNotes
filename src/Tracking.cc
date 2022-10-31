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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"
#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<mutex>
#include <unistd.h>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                   KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) :
        mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
        mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
        mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0) {

    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 去畸变参数
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check re-localisation
    // 插入关键帧（或检查重定位）的 最大/最小帧数（最大帧就是帧率）
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB   0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;
    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];            // 每一帧提取的特征点数1000
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];     // 图像金字塔缩放因子1.2
    int nLevels = fSettings["ORBextractor.nLevels"];                // 图像金字塔层数8
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];           // fast特征点的默认阈值20
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];           // 默认阈值提取不到足够特征点，则用最小阈值8

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor,
                                          nLevels, fIniThFAST, fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,
                                               nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,
                                             nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    // 远/近点阈值 = baseline * ThDepth（35倍基线长度）
    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    // 深度相机，视差图转化深度图时的因子
    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

// 输入双目图像(RGB,BGR)，转化为灰度图； 跟踪图像帧，返回当前帧Tcw
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // 构造左、右目图像帧，(提取特征点、去畸变、双目匹配）
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,
                          mpORBextractorLeft,mpORBextractorRight,
                          mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    // 跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
}

// 输入RGB-D图像(RGB,BGR)，转化为灰度图； 跟踪图像帧，返回当前帧Tcw
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 转换成为真正尺度下的深度
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,
                          mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// 输入单目图像(RGB,BGR)，转化为灰度图； 跟踪图像帧，返回当前帧Tcw
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,
                              mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,
                              mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// 跟踪：估计相机位姿、跟踪局部地图
void Tracking::Track(){
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // 存储Track最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed，地图锁
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 根据mState进行初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();     // 双目/RGB-D初始化(相机位姿、地图点、rKF)，成功后 mState = OK
        else
            MonocularInitialization();  // 单目初始化，成功后 mState=OK

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    // 进行跟踪
    else
    {
        // System is initialized. Track Frame.(mState=OK)
        bool bOK;   // 用于记录三种跟踪方式的跟踪结果

        // Initial camera pose estimation using motion model or re-localization (if tracking is lost)
        // SLAM模式，定位、地图更新（默认为false，SLAM模式）
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // 正常跟踪
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 局部建图线程可能会对原有的地图点进行替换，在这里检测并替换
                CheckReplacedInLastFrame();

                // 若速度为空(刚初始化/跟丢) 或 刚完成重定位，进行参考关键帧跟踪
                if(mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();         // 跟踪参考关键帧
                }
                // 否则，进行恒速模型跟踪；若恒速模型跟踪失败，则进行参考关键帧跟踪
                else
                {
                    bOK = TrackWithMotionModel();           // 恒速模型跟踪
                    if(!bOK)                                // 恒速模型跟踪失败，则跟踪参考关键帧
                        bOK = TrackReferenceKeyFrame();
                }
            }
            // mState!=OK，跟踪失败，进行重定位
            else
            {
                bOK = Relocalization();
            }
        }

        // 仅跟踪模式
        else
        {
            // Localization Mode: Local Mapping is deactivated
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)   // 匹配到足够地图点，继续跟踪
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else        // 未匹配到足够地图点。计算两个相机位姿，一个用于运动模型；一个用于重定位。
                {           // 如果重定位成功，就进行重定位；如果运动模型成功，就进行恒速跟踪
                    // In last frame we tracked mainly "visual odometry" points.
                    // We compute two camera poses, one from motion model and one doing re-localization.
                    // If re-localization is successful we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;     // motion mode
                    bool bOKReloc = false;  // re-localization
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }

                    bOKReloc = Relocalization();

                    // 若运动模型成功，重定位失败
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)    // 当前帧匹配点数较少，增加当前帧地图点的观测次数
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    // 若重定位成功
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // 如果有相机位姿的初步估计 和 匹配，则跟踪局部地图
        if(!mbOnlyTracking)     // SLAM模式
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else                    // 定位模式
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map, and therefore we do not perform TrackLocalMap(). Once the system re-localizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)    // 跟踪成功 且 匹配到足够多的地图点
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState = LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        // 跟踪良好，检查是否插入关键帧
        if(bOK)
        {
            // Update motion model，mVelocity = Tcw * Twl = Tcl
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3)
                                                                 .colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else    // 否则速度为空
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches，清除当前帧中观测不到的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints，删除临时地图点
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end();
                                          lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;     // 删除指针指向的MapPoint
            }
            mlpTemporalPoints.clear();  // 存储指针，不能直接调用，防止内存泄露

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considered outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points, so we discard them in the frame.
            // 允许被Huber核函数判断为外点的点传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
            // 但是估计下一帧位姿的时候我，们不想用这些外点，所以删掉

            // 删除那些在bundle adjustment中检测为outlier的地图点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 如果初始化后不久就跟踪失败，并且地图中的关键帧数<=5，只能重新Reset
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, resetting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // 跟踪成功，记录当前帧的位姿信息
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    // 跟踪失败，使用上一帧的位姿信息(相对与参考关键帧的)
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

// 双目/RGB-D 初始化地图、相机位姿为世界坐标原点
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)     // 要求特征点数>500
    {
        // Set Frame pose to the origin，当前帧位姿设置为原点
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame，满足特征点数的第一帧，设置为初始关键帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and associate to KeyFrame
        // 创建地图点，并关联到当前帧的特征点
        for(int i=0; i<mCurrentFrame.N;i++)         // 遍历当前帧特征点
        {
            float z = mCurrentFrame.mvDepth[i];     // 当前帧具有正深度的特征点
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);     // 特征点反投影到世界坐标系，得到3D点
                MapPoint *pNewMP = new MapPoint(x3D,pKFini,mpMap);  // 3D点构造为地图点，匹配到参考关键帧
                pNewMP->AddObservation(pKFini,i);           // 关键帧 与 地图点 之间建立关联
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();            // 区分度较高的特征点描述子 作为地图点描述子
                pNewMP->UpdateNormalAndDepth();                     // 更新地图点的平均观测方向、深度范围（观测距离范围）
                mpMap->AddMapPoint(pNewMP);                    // 将地图点添加到地图中

                mCurrentFrame.mvpMapPoints[i] = pNewMP;             // 当前帧特征点i 对应的 地图点
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // 局部地图中添加初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        // mLast** 指向当前帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        // 更新局部关键帧、局部地图点
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();

        // 参考关键帧
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // 更新地图的参考地图点(局部地图点)、初始关键帧
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        // 绘制当前相机位姿
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;      // 初始化成功
    }
}

// 单目初始化地图(首帧图像相机位姿为世界坐标原点)，需要两帧图像 mState=OK
// 并行计算基础矩阵、单应矩阵，恢复出最开始两帧之间的相对位姿及点云，得到初始两帧的匹配、相对运动、初始地图点
void Tracking::MonocularInitialization()
{
    // (重新)初始化第一帧
    if(!mpInitializer)      // mpInitializer默认为NULL
    {
        // Set Reference Frame，特征点数必须>100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);   // 参考帧
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());   // 参考帧去畸变特征点
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)           // 多余，不会执行该语句
                delete mpInitializer;

            // 当前帧构造初始化器，sigma为重投影误差
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
            // 初始匹配vector初始化为-1，存储匹配点id
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    // (重新)初始化第二帧
    else
    {
        // Try to initialize，若第二帧的特征点数<=100，重新构造初始化器
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);   // 匹配器(最佳/次佳特征点评分阈值0.9，检查特征点方向true)
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,
                                                       mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 匹配点<100，重新构造初始化器
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        // 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints、被三角化的特征点
        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw,
                                     mvIniP3D, vbTriangulated))
        {
            // 初始化成功后，删除那些不合格的匹配点(视差角较小，深度值为负)
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // 初始化的第一帧为世界坐标系原点，第二帧位姿(相对于第一帧)为Rcw,tcw
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();    // 创建单目的初始化地图，将3D点(Point3f)类型改为MapPoint，mState=OK
        }
    }
}

// 创建单目初始化地图，初始化成功后将三角化得到的3D点(Point3f)转换为MapPoint类型
// 创建用于初始化的两个初始关键帧、对应的字典、地图点，mState=OK
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 两个关键帧的描述子 转化为 BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and associate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        // Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections，更新该关键帧地图点的共视关键帧、及共视程度 vPairs<共视帧,权重>
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // Bundle Adjustment，全局BA优化，同时优化所有位姿和三维点
    Optimizer::GlobalBundleAdjustment(mpMap, 20);

    // Set median depth to 1，取出场景的中值深度，用于尺度归一化
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    // 平均深度<0；当前帧的 地图点的观测帧>=1的个数 <100
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, resetting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline，缩放初始基准
    // 用中值深度 去归一化 初始两帧之间的 平移量
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points，用中值深度 去归一化 初始地图点的3D世界坐标
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();   // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

// 跟踪参考关键帧。用参考关键帧的地图点，来对当前普通帧进行跟踪（特征点描述子匹配）
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We first perform an ORB matching with the reference keyframe
    // If enough matches are found, we set up a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // 参考关键帧、当前帧之间的地图点(特征点)匹配，使用描述子、BoW、旋转直方图(特征点的旋转方向统计)
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    // 将当前帧的初始位姿 设置为上一帧的位姿，通过优化3D-2D重投影误差，获得当前帧位姿
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers，剔除优化过程中标记的外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)   // 遍历当前帧特征点
    {
        if(mCurrentFrame.mvpMapPoints[i])   // 特征点对应的地图点存在
        {
            if(mCurrentFrame.mvbOutlier[i]) // 但地图点被标记为外点
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                // 清除该地图点 在当前帧中的痕迹
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

// 更新上一帧（位姿、根据深度较小的点产生新的地图点）
// 单目只更新上一帧的位姿Tlw；双目/RGB-D,根据深度值，生成临时地图点
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;      // 上一帧图像的参考关键帧
    cv::Mat Tlr = mlRelativeFramePoses.back();      // 上一帧图像到参考关键帧的位姿变换 Tlr

    mLastFrame.SetPose(Tlr * pRef->GetPose());  // 上一帧图像位姿 Tlw = Tlr * Trw

    // 若上一帧为关键帧 或 单目，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // 对双目/RGB-D相机，根据深度值，为上一帧生成新的临时地图点
    vector<pair<float,int> > vDepthIdx;     // pair<深度，特征点>
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth < mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)     // 遍历pair<深度，特征点>
    {
        int i = vDepthIdx[j].second;            // 特征点索引

        // 如果上一帧该特征点对应的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        bool bCreateNew = false;
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)  // 创建特征点对应的新的临时地图点
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first > mThDepth && nPoints>100)
            break;
    }
}

// 恒速模型跟踪。(mVelocity)Tcl * Tlw
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // 根据参考关键帧更新上一帧；单目更新Tlw，双目会额外创建临时地图点
    UpdateLastFrame();

    // mVelocity=Tcl,当前帧到上一帧的位姿变换
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));     // 清空当前帧的地图点

    // Project points seen in previous frame
    float th;   // 金字塔底层搜索半径
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    // 上一帧的地图点 投影到当前帧。如果匹配点数<20，搜索半径扩大一倍，重新匹配
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame, th,
                                              mSensor==System::MONOCULAR);
    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),
             static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,
                                              mSensor==System::MONOCULAR);
    }
    // 如果扩大搜索半径，匹配数还是不够，则跟踪失败
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    // 利用3D-2D投影关系，优化当前帧位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers，剔除外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;  // 定位模式下，匹配不到足够多的点
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

// 跟踪 局部地图(点)，进一步优化位姿。跟踪点数>=30，跟踪成功，返回true。
// 1. 更新局部地图(局部关键帧、关键点)；   2. 局部地图点投影到当前帧进行匹配
// 3. 优化位姿；    4. 更新地图点并统计跟踪效果；   5. 决定是否跟踪成功(跟踪点数>=30)
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    // 更新局部地图（局部(扩展)关键帧、局部(扩展)地图点）
    UpdateLocalMap();

    // 筛选局部地图中新增的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchLocalPoints();

    // Optimize Pose，优化位姿(前边新增了匹配关系)
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Update MapPoints Statistics
    // 更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
    mnMatchesInliers = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was successful
    // More restrictive if there was a re-localization recently
    if(mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

// 检测是否需要插入新的关键帧
// 不插入关键帧的情况：  仅定位模式；  局部建图被闭环线程叫停；  距离上次重定位较近(1秒内) 且 地图中的关键帧数多于帧率
// 插入关键帧的情况：需要插入标记 bNeedToInsertClose = 跟踪到的地图点(近点)太少 && 未跟踪到的地图点(近点)太多；
//     距离上次插入关键帧>1s || 距离上次插入关键帧较近但局部建图线程空闲 || 双目/RGB-D时，当前帧匹配内点较少或需要插入
//     && 和参考帧相比当前帧跟踪到的点太少或满足bNeedToInsertClose，同时跟踪到的内点还不能太少
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is frozen by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    // Do not insert keyframes if not enough frames have passed from last re-localisation
    const int nKFs = mpMap->KeyFramesInMap();
    if(mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // 参考关键帧中跟踪到的地图点数(最小观测数2或3)
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // 询问局部建图线程是否繁忙，是否可以插入关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // 对于双目或RGB-D，统计成功跟踪的近点的数量，和 未跟踪到但有潜力跟踪的近点
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    // 跟踪到的地图点(近点)太少且未跟踪到的地图点(近点)太多
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds，当前帧 和 参考关键帧 跟踪到点 的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise, send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();   // 局部建图线程先中断BA
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧；
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后local mapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else    // 单目局部建图繁忙时，无法插入关键帧
                return false;
        }
    }
    else
        return false;
}

// 创建新的关键帧，双目/RGb-D时会同时创建地图点
// 将当前帧构造成关键帧；(当前帧的)参考关键帧设置为该关键帧；
void Tracking::CreateNewKeyFrame()
{
    // 若局部建图线程关闭，则无法插入关键帧
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 在UpdateLocalKeyFrames函数中会将与当前帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 对于双目或RGB-D，为当前帧生成新的地图点；单目无操作
    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGB-D sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);   // 插入后，允许局部建图停止

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

// 筛选局部地图中新增的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(),
                                    vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;      // 记录地图点所在帧，防止当前帧的地图点投影计数
                pMP->mbTrackInView = false;                     // 地图点是否可以进行投影
            }
        }
    }

    // Project points in frame and check its visibility
    // 投影新增地图点，检查是否在当前帧视野内
    int nToMatch=0;
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(),
                                    vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)  // 不检查当前帧的地图点
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))  // 判断新增地图点是否在当前帧视野内
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        float th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been re-localised recently, perform a coarser search
        if(mCurrentFrame.mnId < mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints, th);
    }
}

// 更新局部地图：局部关键帧、局部地图点
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // 更新局部关键帧、局部地图点
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

// 更新局部地图点（局部关键帧可以观测到的所有地图点）
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end();
                                          itKF != itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

// 更新局部关键帧(数量要小于80)：当前帧地图点的一级共视帧、一级共视帧的父子关键帧、二级共视帧
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 遍历当前帧的特征点对应的地图点，记录所有<一级共视帧，共视地图点数>
    map<KeyFrame*,int> keyframeCounter;     // <一级共视帧，地图点数>
    for(int i = 0; i < mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])   // 地图点存在
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())               // 地图点有效
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(auto it= observations.begin(),
                        itend= observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    // 最多观测次数(max,共视程度最高)的一级共视关键帧(pKFmax)
    int max = 0;
    KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points.
    // 遍历一级共视帧，添加到局部关键帧，并找到共视程度最高的记录。
    for(map<KeyFrame*,int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;
        if(pKF->isBad())
            continue;

        if(it->second > max)
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);             // 一级共视帧添加到局部关键帧中
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; // 表明该共视帧已是当前帧的局部关键帧，防止重复添加
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 遍历局部关键帧(此时是一级共视帧)，获取其最好的10个共视帧(当前帧的二级共视帧)
    for(vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                          itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);    // 二级共视帧
        // 遍历二级共视帧，添加到局部关键帧
        for(vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF=vNeighs.end();
                                              itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                // 防止重复添加
                if(pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 遍历一级共视帧的子关键帧，添加到局部关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;  // TODO 只添加一个子关键帧？
                }
            }
        }

        // 添加一级共视帧的父关键帧 到 局部关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;      // TODO 添加完父关键帧，直接跳出"一级共视关键帧"循环？
            }
        }

    }   // 遍历一级共视帧

    // 更新当前帧的 参考关键帧(最大一级共视帧)
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;     // 当前帧的 最大一级共视关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// 重定位
// 计算当前帧特征点的词袋向量；找到当前帧的候选关键帧，通过BoW进行匹配；通过EPnP算法估计姿态
// 通过PoseOptimization对姿态进行优化求解；如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Re-localization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for re-localisation
    // 询问关键帧数据库，找到与当前帧相似的用于重定位的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we set up a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;    // 每个关键帧的解算器
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;  // 每个关键帧与当前帧的地图点匹配关系
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    // 遍历所有的候选关键帧，通过词袋进行快速匹配，用匹配结果初始化PnP Solver
    int nCandidates=0;
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC，每次迭代需要4个点的EPnP
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;       // ransac达到最大迭代次数

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reaches max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                // 只优化位姿，返回内点数量
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                if(nGood<10)
                    continue;

                // 删除外点标记对应的地图点
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],
                                                                 sFound,10,100);
                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],
                                                                      sFound,3,64);
                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                // 剔除外点
                                for(int io = 0; io < mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers, stop ransac and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

// 跟踪线程复位
void Tracking::Reset()
{
    cout << "System Resetting" << endl;

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Resetting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Resetting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Resetting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erases MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

// 根据配置文件，修改相机参数
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} // namespace ORB_SLAM
