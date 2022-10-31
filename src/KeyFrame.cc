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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;      // 下一个关键帧的id

// 关键帧构造函数，将普通帧F构造为关键帧
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId = nNextId++;

    mGrid.resize(mnGridCols);           // 每张图像的网格列数
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);    // 每张图像的网格行数
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];        // 复制每个网格内的特征点信息
    }

    SetPose(F.mTcw);    
}

// 计算当前帧的词袋
void KeyFrame::ComputeBoW()
{
    // 当前帧的词袋向量、特征向量为空时，进行计算
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        // 将当前帧的描述子转换为词袋向量mBowVec(map<WordId, WordValue>)，特征向量mFeatVec(map<NodeId, vector<unsigned int>>)
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

// 设置当前关键帧的位姿为Tcw_(Tcw)，Twc, Ow, Cw
void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);    // 作用域为整个成员函数，函数执行后自动解锁
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);     // 双目基线中心在左目相机系下的坐标
    Cw = Twc*center;                                                                 // 双目基线中心在世界系下的坐标
}

// 获取当前关键帧的位姿 Tcw
cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

// 获取当前关键帧位姿的逆 Twc
cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

// 获取当前帧（左目）相机光心坐标 Ow
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

// 获取当前帧(双目)基线中心坐标 Cw
cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

// 获取当前帧的旋转矩阵 Rcw
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

// 获取当前帧的平移向量 tcw
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

// 添加连接，为当前关键帧添加共视帧pKF, 及两帧之间的权重weight(共视地图点数)
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))        // 该关键帧的共视帧中目前不存在pKF
            mConnectedKeyFrameWeights[pKF]=weight;          // 添加pKF为当前关键帧的共视帧
        else if(mConnectedKeyFrameWeights[pKF]!=weight)     // 该关键帧的共视帧中存在pKF，但权重不同
            mConnectedKeyFrameWeights[pKF]=weight;          // 更新权重
        else
            return;
    }

    UpdateBestCovisibles();     // 更新当前关键帧的共视关系，按照权重 重新排序
}

// 更新当前帧的共视帧顺序，按照权重，从大到小对共视关键帧进行排序，得到排序后的共视帧vector 及 权重vector
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    // 将共视关系数据类型从map<KeyFrame*,int>转换为vector<pair<int,KeyFrame*>>，方便排序
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(auto mit=mConnectedKeyFrameWeights.begin(),
             mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    // 对共视关系进行排序,sort是从小到大排序
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        // 用push_front实现从大到小排列
        lKFs.push_front(vPairs[i].second);      // push_front是往前存储list
        lWs.push_front(vPairs[i].first);
    }
    // 深拷贝
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

// 获取当前关键帧的所有共视关键帧集合（无序）（共视帧至少有15个共视地图点）
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(auto mit=mConnectedKeyFrameWeights.begin(); mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

// 获取当前关键帧的所有共视关键帧 (按照权重从大到小排序)
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

// 获取当前关键帧的 前N个共视帧（按权重从大到小排列）
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

// 获取当前帧 >w个共视地图点的 共视帧
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    // 从mvOrderedWeights中找出第一个大于w的那个迭代器
    auto it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end())      // 迭代器指向mvOrderedWeights末尾
        return vector<KeyFrame*>();
    else
    {
        int n = it - mvOrderedWeights.begin();      // >w的个数
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

// 获取当前帧与pKF帧的权重（共视地图点个数）
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

// 为当前关键帧添加地图点，idx为地图点pMP在当前帧的特征点索引
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

// 从当前关键帧中删除idx特征点对应的地图点
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

// 从当前关键帧删除地图点pMP
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);    // 先获取地图点在当前关键帧中的索引
    if(idx>=0)                                      // 根据索引，删除地图点
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

// 用地图点pMP 替换当前帧中 idx对应的地图点
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

// 获取当前帧的（有效）地图点集合
set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    // 遍历当前帧的地图点
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])    // 地图点是否存在
            continue;

        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())       // 有效地图点
            s.insert(pMP);
    }
    return s;
}

// 该帧的地图点中，观测数 >=minObs的地图点数目（遍历特征点，找对应的地图点）
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    //  遍历特征点 对应的地图点
    int nPoints=0;
    const bool bCheckObs = minObs>0;    // 是否检查观测数
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)                         // 对应的地图点存在
        {
            if(!pMP->isBad())           // 且有效
            {
                if(bCheckObs)           // 检查观测数
                {
                    if(mvpMapPoints[i]->Observations() >= minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

// 获取当前帧的所有地图点
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

// 获取当前帧idx特征点 对应的地图点
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

// 更新该关键帧地图点的共视帧，及共视程度
// step1 遍历该关键帧观测到的地图点，找到每个地图点的所有观测帧，统计每个地图点的共视帧 及 与该关键帧的共视程度
// step2 为满足共视程度阈值的共视帧，添加连接（与该关键帧），如果没有共视帧满足阈值要求，则选择一个共视程度最大的帧，添加连接
// step3 对满足阈值要求的vPairs<共视帧,权重>按照权重进行从大到小排序，得到排序后的共视帧vector，权重vector
void KeyFrame::UpdateConnections()
{
    // 遍历该关键帧观测到的地图点，找到每个地图点的所有观测帧，统计每个地图点的共视帧 及 与该关键帧的共视程度
    map<KeyFrame*,int> KFCounter;   // 默认值为<NULL,0>
    vector<MapPoint*> vpMP;         // 该关键帧观测到的地图点
    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }
    // For all map points in keyframe check in which other keyframes are they seen
    // Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(!pMP)
            continue;
        if(pMP->isBad())
            continue;

        // 遍历每个地图点的所有观测帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId == mnId)
                continue;
            KFCounter[mit->first]++;    // 统计每个地图点的共视帧 及 与该关键帧的共视程度
        }
    }

    // This should not happen
    if(KFCounter.empty())
        return;

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    // 为满足共视程度阈值的共视帧，添加连接（与该关键帧），如果没有共视帧满足阈值要求，则选择一个共视程度最大的帧，添加边
    int nmax=0;                             // 记录最大共视程度
    KeyFrame* pKFmax=NULL;                  // 记录最大共视程度的关键帧
    int th = 15;                            // 共视程度阈值
    vector<pair<int,KeyFrame*> > vPairs;    // 存储满足阈值要求的 pair<共视帧指针,共视程度>
    vPairs.reserve(KFCounter.size());
    // 遍历KFCounter<共视帧，共视程度>
    for(map<KeyFrame*,int>::iterator mit=KFCounter.begin(), mend=KFCounter.end(); mit != mend; mit++)
    {
        if(mit->second > nmax)      // 记录最大共视程度 及对应的共视帧
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second >= th)       // 对满足共视程度阈值的共视帧，与该关键帧添加连接
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);      // 与该关键帧添加连接
        }
    }
    // 如果没有共视帧满足阈值要求，则选择一个共视程度最大的帧，添加边
    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    // 对vPairs进行排序，得到排序后的共视帧vector，权重vector
    sort(vPairs.begin(),vPairs.end());      // 默认从小到大
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)             // 实现从大到小
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);
        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFCounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // 更新生成树 mbFirstConnection默认为true,
        if(mbFirstConnection && mnId!=0)    // 该帧不能是关键帧序列的第一帧，可以是共视帧中的第一帧
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();    // 初始化该关键帧的父关键帧为共视程度最高的那个共视关键帧
            mpParent->AddChild(this);                       // 该关键帧就作为其父帧的子帧
            mbFirstConnection = false;                          // 共视帧第一帧标志位置为false
        }

    }
}

// 将pKF添加为该关键帧的子帧（当前帧与pKF具有最大共视关系）
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

// 将pKF从该关键帧的子帧中删除
void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

// 改变当前帧的父关键帧，替换为pKF
void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

// 获取当前关键帧的子关键帧集合
set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

// 获取当前关键帧的父关键帧
KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

// 判断pKF是否是当前关键帧的子关键帧
bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

// 为当前帧添加回环边pKF（两帧之间形成闭环关系）
void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;              // 形成闭环之后，该关键帧在优化时不能被删除
    mspLoopEdges.insert(pKF);     // 将pKF插入回环边 集合
}

// 获取该关键帧的回环边集合（可以与当前帧形成闭环关系）
set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

// 该关键帧不能被删除（形成回环关系的关键帧，不能在优化的过程中删除）
// mbNotErase作用：
// 表示要删除该关键帧及其连接关系但是这个关键帧有可能正在回环检测或者计算sim3操作，这时候虽然该关键帧冗余，但是却不能删除；
// 仅设置mbNotErase为true，这时候调用SetBadFlag函数时，不会将这个关键帧删除，只会把mbTobeErase变成true，代表这个关键帧可以删除但不到时候,
// 先记下来以后处理。在闭环线程里调用SetErase()会根据mbToBeErased来删除之前可以删除还没删除的帧。
void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

// 可以删除该关键帧
void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

// TODO：真正执行删除关键帧的操作，删除该关键帧与其他共视帧、地图点之间的关系...需要进一步理解
// mbNotErase作用：
// 表示要删除该关键帧及其连接关系但是这个关键帧有可能正在回环检测或者计算sim3操作，这时候虽然该关键帧冗余，但是却不能删除；
// 仅设置mbNotErase为true，这时候调用SetBadFlag函数时，不会将这个关键帧删除，只会把mbTobeErase变成true，代表这个关键帧可以删除但不到时候,
// 先记下来以后处理。在闭环线程里调用SetErase()会根据mbToBeErased来删除之前可以删除还没删除的帧。
void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)             // 第1个关键帧不能被删除
            return;
        else if(mbNotErase)     // 此时不应该被删除，所以设置为即将被删除，为后续删除提供标志位
        {
            mbToBeErased = true;
            return;
        }
    }

    // 遍历该关键帧的<共视帧，权重>，删除每个共视帧与当前帧的连接关系
    for(auto mit = mConnectedKeyFrameWeights.begin(),
             mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    // 遍历该关键帧的地图点，删除每个地图点观测中的该关键帧
    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        // 清空当前关键帧的共视关系
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        // 更新生成树，主要是处理好父子关键帧，不然会造成整个关键帧图的断裂，或者混乱
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);     // 将当前帧的父关键帧放入候选父关键帧集合中

        // Assign at each iteration one child with a parent (the pair with highest co-visibility weight)
        // Include that child as new parent candidate for the rest
        // 每迭代一次就为其中一个子关键帧寻找父关键帧（最高共视程度），找到父帧的子关键帧可以作为其他子关键帧的候选父关键帧
        while(!mspChildrens.empty())
        {
            bool bContinue = false;     // 是否继续迭代，默认为false

            int max = -1;               // 共视程度
            KeyFrame* pC;               // 子关键帧
            KeyFrame* pP;               // 父关键帧

            // 遍历当前帧的子关键帧，找到子关键帧的最大共视帧，作为子关键帧的父帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;   // 当前帧的子关键帧
                if(pKF->isBad())        // 跳过无效的子关键帧
                    continue;
                // Check if a parent candidate is connected to the keyframe
                // 遍历子关键帧pKF的共视关键帧，检查子关键帧是否有候选父关键帧
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {   // sParentCandidates中刚开始存的是这里子关键帧的“爷爷”，也是当前关键帧的候选父关键帧
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end();
                        spcit!=spcend; spcit++)
                    {
                        // 当前帧的子关键帧的共视帧 与 当前帧的父帧 id相同，形成共视
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]); // 子关键帧与子关键帧的共视帧 之间的权重
                            if(w>max)                   // 更新共视程度最高的子关键帧
                            {
                                pC = pKF;               // 子关键帧
                                pP = vpConnected[i];    // 目前子关键帧的（最大）共视帧，作为子关键帧的父帧
                                max = w;                // 最大权重
                                bContinue = true;       // 说明子帧找到了可以作为其新父关键帧的帧
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);             // 改变帧pC（子关键帧）的父关键帧
                sParentCandidates.insert(pC);       // pC作为其他子帧的候选父关键帧
                mspChildrens.erase(pC);             // 从当前帧的子关键帧集合中删除pC子帧
            }
            else
                break;
        }

        // If a child has no co-visibility links with any parent candidate, assign to the original parent of this KF
        // 如果还有子关键帧没有找到新的父关键帧，那么直接将该关键帧的父帧作为其子关键帧的父帧
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);         // 该关键帧的父帧删除 该关键帧本身
        mTcp = Tcw * mpParent->GetPoseInverse();    // mTcp = Tcw * Twp
        mbBad = true;                               // 该关键帧已被删除
    }

    // 地图和关键帧数据库中删除该关键帧
    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

// 返回当前关键帧是否已被删除
bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

// 删除当前关键帧与pKF帧之间的连接关系（从当前帧的共视帧中删除pKF）
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

// 获取当前帧中，某个特征点的邻域中的特征点id
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// 判断某个点(x,y)是否在当前关键帧的图像中
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

// 在双目和RGB-D情况下,将特征点i反投影到世界坐标系
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);   // 相机坐标

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

// 评估当前关键帧的场景深度，（q=2表示中值，单目情况下使用）
// 对当前关键帧下 所有地图点的深度 进行从小到大排序,返回距离头部 第1/q处的深度值作为当前场景的平均深度
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    // 由于只计算相机坐标系下的深度值，故只取旋转矩阵的第3行，平移向量的第3个元素
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);   // 旋转矩阵的第3行
    Rcw2 = Rcw2.t();                                            // 旋转矩阵第3行的转置
    float zcw = Tcw_.at<float>(2,3);                      // 深度
    // 遍历所有特征点，计算对应地图点在相机坐标系下的深度
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;                 // 相机坐标系下的深度值
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());              // 从小到大排列vDepths

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
