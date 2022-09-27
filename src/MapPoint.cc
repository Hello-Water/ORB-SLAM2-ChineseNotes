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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;  // mnID = nNextID++, mnID是地图点的全局ID
mutex MapPoint::mGlobalMutex;           // 全局BA中，对当前地图点进行操作时的互斥量

// 给定地图点3D坐标 及 keyframe 构造地图点（单双目初始化、创建新的地图点、双目创建新的关键帧）
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F); // 平均观测方向初始化为0向量

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    // 跟踪线程和局部建图线程都会创建地图点，添加互斥锁，防止地图点的ID混乱
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

// 给定地图点3D坐标 及 frame 构造地图点（双目中更新图像帧），idxF是MapPoint在Frame中对应特征点的编号
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;     // 世界坐标系下，相机光心指向3d点的向量（当前帧观测方向）
    mNormalVector = mNormalVector/cv::norm(mNormalVector);  // 转化为单位法向量

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);    // (某层金字塔图像)相机光心到3D点的距离
    const int level = pFrame->mvKeysUn[idxF].octave;    // 关键点所在的金字塔层级
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];  // 对应金字塔层的缩放因子
    const int nLevels = pFrame->mnScaleLevels;          // 金字塔层数

    mfMaxDistance = dist*levelScaleFactor;              // 最大深度（对应金字塔第0层）
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];    // 最小深度（对应金字塔最顶层，此处为第7层）

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor); // 特征点的描述子拷贝给地图点

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    // 跟踪线程和局部建图线程都会创建地图点，添加互斥锁，防止地图点的ID混乱
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

// 设置地图点在世界坐标系下的坐标
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);     // 全局BA时的线程锁
    unique_lock<mutex> lock(mMutexPos);         // 对地图点操作的线程锁
    Pos.copyTo(mWorldPos);
}

// 获取地图点在世界坐标系下的坐标
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

// 世界坐标系下，地图点被多个相机观测的平均观测方向
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

// 获取创建当前地图点的参考关键帧
KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

// 给地图点添加观测（记录哪个关键帧的哪个特征点观测到的地图点）
// 增加观测的相机数目nObs,单目+1，双目/RGBD +2，是建立共决关键帧的核心函数
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    // mObservations 是 map<KeyFrame *, size_t> 类型
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))      // 判断pKF是否已存在
        return;
    mObservations[pKF]=idx;             // 添加pKF以及对应的特征点索引idx

    if(pKF->mvuRight[idx]>=0)           // 双目或RGB-D
        nObs+=2;
    else                                // 单目
        nObs++;
}

// 删除某个关键帧对当前地图点的观测
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;    // 地图点好坏标记

    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))          // 观测中存在该关键帧pKF
        {
            int idx = mObservations[pKF];       // 取出特征点idx
            if(pKF->mvuRight[idx]>=0)           // 判断单目/双目/RGB-D
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);         // 删除key-value对

            // 如果该帧pKF是创建地图点时的参考帧，令mpRefKF指向观测的第一个<KeyFrame *, size_t>对
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            // 当观测到该地图点的相机数目<=2时，将该地图点标记为bad
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)    // 如果是bad点，则告知可以观测到该MapPoint的Frame，该MapPoint已被删除
        SetBadFlag();
}

// 获取能够观测到该地图点的所有关键帧及对应特征点索引
map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

// 该地图点被观测到的数目
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

// 提醒可以观测到该地图点的Frame，该地图点被标记为bad，已被删除
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);   // 删除可观测到该地图点的所有关键帧上对应的特征点（地图点）
    }

    mpMap->EraseMapPoint(this);                // 从所属的地图中删除该点
}

// 获取用来替换的地图点
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

// 用pMP替换该地图点，并更新观测关系，TODO：替换观测关系的问题？LoopClosing中使用
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)   // 是同一个地图点则跳过
        return;

    // 清除当前地图点的信息
    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;          // 暂存当前地图点的原有观测
        mObservations.clear();      // 清除当前地图点的原有观测
        mbBad=true;                 // bad点标记
        nvisible = mnVisible;       // 暂存当前地图点的可视次数、可匹配次数
        nfound = mnFound;
        mpReplaced = pMP;           // 当前地图点已被替换
    }

    // 遍历所有能观测到原地图点的关键帧，都要复制到替换的地图点上
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;     // 当前地图点的原有观测 关键帧

        // 用于替换的地图点的观测关键帧中，没有pKF（但该地图点可匹配到pKF中的相应特征点）
        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);    // 原关键帧上相应索引位置（地图点）替换为pMP
            pMP->AddObservation(pKF,mit->second);           // 将该关键帧添加到pMP的观测中
        }
        // 用于替换的地图点的观测帧中，有pKF（说明原来特征点对应两个地图点，那么要删除该特征点的地图点对应关系）
        else
        {
            pKF->EraseMapPointMatch(mit->second);           // 删除原有关键帧中的相应特征点索引
        }
    }

    // TODO：将当前地图点的观测数据等其他数据都"叠加"到新的地图点上，直接叠加？
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);     // 删除该地图点
}

// 没有经过 MapPointCulling 检测的 MapPoints, 认为是坏掉的点
bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

// 该地图点在其它n帧的视野范围内，通过Frame::isInFrustum()函数判断
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

// 该地图点在其它n帧中有匹配点
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

// mnFound/mnVisible
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

// 计算该地图点最有代表性的描述子（最好的描述子与其他描述子具有最小的距离中值）
// 插入关键帧后，要判断是否更新代表当前点的描述子
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    // 检索该地图点所有观测的描述子
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // 遍历该地图点的所有观测（关键帧），取出对应的特征点描述子
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them，计算描述子之间的距离
    const size_t N = vDescriptors.size();
    float Distances[N][N];      // 对称矩阵
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;      // 与自己的距离为0
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with the least median distance to the rest
    // 取出与其他描述子具有最小中值距离 的描述子
    int BestMedian = INT_MAX;   // 最小的中值
    int BestIdx = 0;            // 最小中值对应的索引
    for(size_t i=0;i<N;i++)
    {
        // 取出第ℹ行，即第i个描述子与其他描述子之间的距离
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());  // 升序排列
        int median = vDists[0.5*(N-1)];              // 获得中值

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

// 获取当前地图点的描述子（最具代表性的描述子）
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

// 地图点在pKF帧中对应的特征点索引
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

// 检查帧pKF是否在当前地图点的观测帧中
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

// 更新地图点的平均观测方向、深度范围（观测距离范围）
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        // 取出当前地图点的observations, RefKF, Pos
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        // 所有 关键帧对 当前地图点 的 观测方向（归一化后） 求和
        normal = normal + normali/cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();                       // 参考帧指向地图点的向量（世界坐标系下）
    const float dist = cv::norm(PC);                               // 该地图点到参考关键帧的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;    // 地图点对应的参考帧特征点在金字塔的第几层
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];      // 对应金字塔层的缩放因子,s^n
    const int nLevels = pRefKF->mnScaleLevels;                          // 总的金字塔层数，默认为8

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;                              // 观测到该地图点的距离上限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];    // 观测到该地图点的距离下限
        mNormalVector = normal/n;                                           // 该地图点的平均观测方向（对上边的normal和 求平均）
    }
}

// 获取最小距离不变量 = 0.8 * mfMinDistance
float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

// 获取最大距离不变量 = 1.2 * mfMaxDistance
float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

/**
// 预测地图点对应的特征点在图像金字塔的第几层，给定当前距离、关键帧
//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
 */
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;  // ratio = 1.2^m
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);    // m = log(ratio)/log1.2
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

// 预测地图点对应的特征点在图像金字塔的第几层，给定当前距离、图像帧
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

} //namespace ORB_SLAM
