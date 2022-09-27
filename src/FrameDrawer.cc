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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

// 构造函数，初始化图像显示画布大小
FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;

    // 初始化图像显示画布，固定画布大小为640*480
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

// 绘制图像帧，包括图像、特征点、地图点、跟踪状态
cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys;          // Initialization: KeyPoints in reference frame
    vector<int> vMatches;                   // Initialization: correspondences with reference key-points
    vector<cv::KeyPoint> vCurrentKeys;      // KeyPoints in current frame
    vector<bool> vbVO, vbMap;               // Tracked MapPoints in current frame
    int state;                              // Tracking state

    // Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state = mState;

        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        // 如果没有初始化，则获取当前帧的特征点、参考帧的特征点、及他们的匹配关系
        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        // 如果处于跟踪状态，获取当前帧的特征点、当前帧首次观测到的地图点、地图点
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        // 如果跟踪丢失，只显示当前帧的特征点
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3)     //  this should be always true
        cvtColor(im,im,CV_GRAY2BGR);    // 将图像转化为3通道彩色图

    // Draw
    // 如果尚未初始化，则进行初始化；将参考帧与当前帧的特征点连线
    if(state==Tracking::NOT_INITIALIZED)    // INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }        
    }
    // 如果处于跟踪状态，绘制(首次)跟踪到的地图点（特征点），同时用圆圈、方框框住，首次跟踪(蓝)与非首次跟踪的颜色不同(绿)
    else if(state==Tracking::OK)            // TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = vCurrentKeys.size();

        // 遍历当前帧特征点
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])     // 首次跟踪到、或在地图中
            {
                cv::Point2f pt1,pt2;    // 特征点左上角、右下角的点，用于绘制方框(方框变长一半为r)
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                // 地图中的点，同时用绿色圆圈、方框框住
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                // This is match to a "visual odometry" MapPoint created in the last frame
                // 当前帧首次跟踪到上一帧的地图点，同时用蓝色圆圈、方框框住
                else
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;     // 带有状态栏文本信息的图像帧
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

// 绘制文本信息（扩展原始图像，在扩展区域写入当前帧的跟踪信息）
void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;     // 状态信息字符串
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    // 计算字符串文字所占用的图像区域的大小
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,
                                        1,1,&baseline);

    // 根据状态信息字符串大小，扩展图像的高度（行数），将原始图像复制到扩展图像，扩展区域为黑色背景，写入状态信息为白色字体
    imText = cv::Mat(im.rows + textSize.height + 10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,
                                                                         im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,
                1,cv::Scalar(255,255,255),1,8);

}

// 更新跟踪线程信息至 绘制图像帧线程（图像、特征点、地图、跟踪状态）
void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    // 如果跟踪器上一帧还未初始化，只获取初始化帧的特征点、当前帧的匹配信息
    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    // 如果跟踪器上一帧 跟踪正常，更新跟踪器当前帧的地图点信息（地图中已有的点，VO首次跟踪到的点）
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;
                }
            }
        }
    }
    // 更新跟踪线程的状态（跟踪器的上一帧处理状态）
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
