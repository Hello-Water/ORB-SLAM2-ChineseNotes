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
#include <unistd.h>
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{
// 空的构造函数，初始化一些成员变量
LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

// LocalMapping线程主函数
void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // 设置Local Mapping线程繁忙，Tracking无法插入关键帧
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        // 检查关键帧队列是否为空(是否有待处理的关键帧)
        if(CheckNewKeyFrames())     // 有待处理的关键帧
        {
            // BoW conversion and insertion in Map
            // 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图、扩展树，插入到地图等
            ProcessNewKeyFrame();

            // Check recent MapPoints，检查新关键帧引入的地图点，剔除质量不好的点
            MapPointCulling();

            // Triangulate new MapPoints，三角化出新的地图点(当前关键帧 与 共视帧)
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())    // 关键帧队列处理完毕 (没有待插入的关键帧)
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // 检查 并 融合 当前关键帧 与 两级共视帧 重复的地图点
                SearchInNeighbors();
            }

            mbAbortBA = false;  // 关键帧插入完毕，为false；只要有待处理的关键帧，BA就会暂停true；

            // 若关键帧队列处理完毕 且 没有被要求暂停Local Mapping
            // 进行Local BA，并剔除局部冗余关键帧
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA，要求当前地图中关键帧个数>2
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                // 检查冗余局部关键帧(当前关键帧的共视帧是否冗余)
                KeyFrameCulling();
            }
            // 将当前关键帧插入到闭环关键帧队列
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }

        // 若暂停局部建图
        else if(Stop())
        {
            // Safe area to stop，没有完全结束，等待
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())   // 是否有结束线程请求
                break;          // 跳出while循环，将LocalMapping线程设置为Finish
        }

        ResetIfRequested(); // 若有请求复位指令，则LocalMapping线程复位

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }   // while循环

    SetFinish();
}

// 将关键帧插入到新关键帧列表 mlNewKeyFrames，等待处理
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;   // 有待处理的关键帧，BA暂停
}

// 检查关键帧队列是否为空(是否有待处理的关键帧)
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

// 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图、扩展树，插入到地图等
void LocalMapping::ProcessNewKeyFrame()
{
    {   // 取出队列中最前边的关键帧进行处理
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // 获取当前帧的所有地图点，进行遍历。若来自局部地图点，更新其观测、方向距离范围、描述子；若是当前帧生成的，待检验
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 地图点不是来自当前帧的观测 (比如来自局部地图点)
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                // this can only happen for new stereo points inserted by the Tracking
                // 如果当前帧中已经包含了这个地图点,但是这个地图点中却没有包含这个关键帧的信息
                else
                {
                    // 这些地图点只能来自双目或RGB-D跟踪过程中新生成的地图点，
                    // 放入mlpRecentAddedMapPoints，等待MapPointCulling检测
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Co-visibility Graph
    // 更新该关键帧地图点的共视帧，及共视程度，更新扩展树(父子关系)
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

// 检查新关键帧引入的地图点，剔除质量不好的点
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // 根据相机类型设置不同的观测阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // 遍历新增的地图点
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        // 从队列中删除bad点
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // mnFound/mnVisible (匹配到地图点/在视野范围内)
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 从该地图点被建立的关键帧到当前关键帧，帧间差>=2，且该地图点的观测数较小
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 从该地图点被建立，已经过去了3个关键帧
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        // 否则，就是质量高的地图点，保留(不是bad点、匹配程度高、观测帧数多、离初始创建帧近)
        else
            lit++;
    }
}

// 三角化出新的地图点(当前关键帧 与 共视帧)
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in co-visibility graph
    // 检索共视程度最高的nn个关键帧
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // 构造匹配器
    ORBmatcher matcher(0.6,false);          // 最优/次优比值0.6，检查方向

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    // Search matches with epipolar restriction and triangulate
    // 遍历共视关键帧，搜索匹配并用极线约束剔除误匹配，最终三角化
    int nnew=0;
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        // ProcessNewKeyFrame之后，mlNewKeyFrames可能为空，CheckNewKeyFrames为false
        if(i>0 && CheckNewKeyFrames())  // 有关键帧进来，暂停创建地图点？
            return;

        // Check first that baseline is not too short
        // 首先检查基线(当前关键帧与共视帧之间的距离)是否太短了
        KeyFrame* pKF2 = vpNeighKFs[i];
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)    // 双目/RGB-D基线(帧间距离)要长于其自身的基线
        {
            if(baseline < pKF2->mb)
                continue;
        }
        else                // 单目，基线/场景中值深度 的比值要大于0.01
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix，计算两帧之间的基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fulfill epipolar constraint
        // 通过词袋对两关键帧未匹配的特征点快速匹配，用对极约束抑制离群点，生成新的匹配点对
        // 未匹配的特征点指的是，特征点未匹配到地图点
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match，遍历所有匹配到的特征点，三角化出地图点
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays，两个投影射线，在世界坐标系下的视差角
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1;      // 初始化为一个很大的值
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)        // 当前帧右目存在，计算当前帧地图点的双目视差角
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)   // 共视帧右目存在，计算共视帧地图点的双目视差角
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0
                    && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {   // 匹配点对形成的夹角较大，但又小于单帧双目的视差角，三角法恢复3D点
                // Linear Triangulation Method
                // 已知匹配点对{x,x'}，投影矩阵{P,P'}，x=PX, x'=P'X，利用(x^)PX = 0
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
                x3D = vt.row(3).t();
                if(x3D.at<float>(3)==0)
                    continue;
                // Euclidean coordinates，归一化
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            // 匹配点对夹角较小，用当前帧、共视帧中 双目夹角较大的那帧恢复3D点
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            // 非双目 且 夹角过大
            else
                continue; // No stereo and very low parallax

            // Check triangulation in front of cameras
            // 检查地图点是否在相机前方
            cv::Mat x3Dt = x3D.t();      // 转换为行向量
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;
            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            // Check re-projection error in first keyframe
            // 检查当前帧的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)   // 当前帧是单目
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else            // 当前帧是双目/RGB-D
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            // Check re-projection error in second keyframe
            // 检查共视帧的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;

            if(!bStereo2)   // 共视帧是单目
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else            // 共视帧是双目/RGB-D
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            // Check scale consistency
            // 检查尺度一致性，距离比例 与 金字塔层比例 不能差太多
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;    // 不考虑金字塔的距离比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];
            // ratioFactor = 1.5 * mfScaleFactor;
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is successful，三角化成功，3D点构建为地图点
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

// 检查并融合 当前关键帧 与 两级共视帧 重复的地图点
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;      // 存储待融合的关键帧
    // 遍历一级共视帧
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;  // 一级共视帧与当前帧融合标记(用当前关键帧id)

        // Extend to some second neighbors
        // 遍历二级共视帧(5个)
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(),
                                              vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId
                              || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            // 为什么二级共视帧没有标记？
            // pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
        }
    }

    // 双向融合：将当前关键帧的地图点投影到两级共视帧；将两级共视帧的地图点投影到当前关键帧。
    // Search matches by projection from current KF in target KFs
    // 将当前帧的地图点投影到共视帧中，寻找匹配特征点及其对应的地图点，比较两个地图点的观测数，进行融合
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    // 将两级共视帧的地图点投影到当前帧，寻找匹配特征点及其对应的地图点，存储待融合地图点
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    // 遍历待融合的两级共视帧，找到待融合的地图点
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(),
                                    vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        // 遍历每个共视帧的所有地图点
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(),
                                        vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;    // 标记待融合的地图点(用当前关键帧id)
            vpFuseCandidates.push_back(pMP);
        }
    }
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    // Update points，更新当前关键帧的地图点的描述子、平均观测方向、深度等
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in co-visibility graph
    // 更新当前关键帧的连接关系：共视帧、共视程度、生成树(父子帧)
    mpCurrentKeyFrame->UpdateConnections();
}

// 根据两帧的世界位姿，计算F12
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

// 请求停止当前LocalMapping线程，相关标志位置为true
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

// 停止(暂停)当前LocalMapping线程
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)   // 收到停止请求 且 已停止(mbNotStop默认为false)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

// 检查当前LocalMapping线程是否已停止
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

// 检查是否收到停止请求
bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

// 释放该线程、释放当前还在缓冲区(新关键帧列表)的关键帧指针
void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

// 是否可以接收关键帧
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

// 设置可接收关键帧状态
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

// 设置当前线程不要停止 是否成功
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

// 中断BA
void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

// 剔除局部冗余关键帧(90%以上的地图点可以被至少3个其他关键帧观测到)
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    // 遍历当前关键帧的共视关键帧
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;

        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;           // 地图点的可观测数阈值
        int nRedundantObservations=0;   // 每个共视帧中的冗余地图点个数
        int nMPs=0;                     // 每个共视帧中的有效地图点个数
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;

                    if(pMP->Observations()>thObs)
                    {   // 获取能观测到地图点pMP的所有关键帧，遍历
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;     // 记录地图点pMP的可观测数
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(),
                                                                   mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;

                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;
                            if(scaleLeveli <= scaleLevel+1)
                            {
                                nObs++;
                                if(nObs >= thObs)
                                    break;
                            }
                        }

                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  // 遍历每个共视帧中的地图点

        if(nRedundantObservations > 0.9*nMPs)
            pKF->SetBadFlag();      // 该共视帧是冗余关键帧
    }
}

// 计算三维向量v的反对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

// 请求复位当前线程
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

// 若有请求复位指令，则LocalMapping线程复位
void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

// 请求结束当前线程
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// 检查是否有结束请求
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

// 设置当前线程结束
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

// 检查当前线程是否结束
bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
