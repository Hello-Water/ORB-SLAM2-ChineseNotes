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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>
#include <unistd.h>

namespace ORB_SLAM2
{
// 构造函数，初始化成员变量。
LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale) :
        mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
        mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false),
        mbFinishedGBA(true), mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

// 闭环线程的主函数
void LoopClosing::Run()
{
    mbFinished =false;
    while(1)    // 线程循环，直到有结束线程指令,mbFinishRequested=true
    {
        // Check if there are keyframes in the queue
        // 检查是否有待闭环的关键帧(待检测队列是否为空)
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check co-visibility consistency
            if(DetectLoop())        // 得到足够一致性的候选帧群 mvpEnoughConsistentCandidates
            {
               // Compute similarity transformation [sR|t], in the stereo/RGB-D case s=1
               if(ComputeSim3())    // 计算待闭环帧 与 候选闭环帧群 的Sim3变换，确定闭环帧及对应的Sim3变换
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();     // 外部若有复位指令，进行复位

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

// 将关键帧插入到 等待被闭环检测的关键帧列表
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

// 检查待闭环检测的关键帧列表是否为空
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

// 检测回环，返回是否检测成功
bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();     // 取出待闭环检测帧列表的第一帧，作为当前关键帧
        mlpLoopKeyFrameQueue.pop_front();               // 从队列中删除
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();                     // 正在处理，设置为不可被删除
    }

    // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 如果距离上次闭环比较近（10关键帧帧内）
    if(mpCurrentKF->mnId < mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the co-visibility graph
    // We will impose loop candidates to have a higher similarity than this
    // 获取当前正在处理关键帧的共视帧，并计算与共视帧的相似度得分，记录最低得分。
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 在关键帧数据库中寻找候选闭环关键帧（用上一步的最低得分阈值minScore）
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())  // 如果不存在候选闭环帧，则将当前帧添加到关键帧数据库
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate, check consistency with previous loop candidates.
    // Each candidate expands a co-visibility group (keyframes connected to the loop candidate in the co-visibility graph)
    // A group is consistent with a previous group if they share at least a keyframe【vbConsistentGroup】
    // We must detect a consistent loop in several consecutive keyframes to accept it【在连续关键帧上检测到一致性闭环】
    // 遍历候选闭环帧，找到具有足够一致性的“候选帧群”
    mvpEnoughConsistentCandidates.clear();              // 具有足够一致性的“候选帧群”
    vector<ConsistentGroup> vCurrentConsistentGroups;   // 当前累积的一致性“候选帧群”vector
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);    // 累积的一致性群的标记(该群是否是一致的)
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        // 构造某个候选闭环帧的 “候选帧群”【“当前群”】
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();    // “候选帧群” = 候选帧 及其 共视帧
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;         // 是否足够一致，一致性程度>=3
        bool bConsistentForSomeGroup = false;   // 是否与“历史一致性群”具有一致性
        // 遍历 历史中已经具有一致性的“候选帧群”，检查“当前群”是否具有一致性(与“历史帧群”共享至少一个关键帧)
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            // 遍历“当前群”中的关键帧，查看其是否在“历史一致性群iG”中。若在，说明“当前群” 与 “历史一致性群iG”是一致的
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if(bConsistent)     // 若 “当前群” 与 “历史一致性群iG”是一致的，更新相关群组及标志位
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;   // “历史一致性群iG”的一致性程度
                int nCurrentConsistency = nPreviousConsistency + 1;         // “当前群”的一致性程度
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; // this avoids to include the same group more than once
                }
                if(nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true; // this avoids to insert the same candidate more than once
                }
            }   // 若 “当前群” 与 “历史一致性群iG”是一致的
        }   // 遍历“历史一致性群”

        // If the group is not consistent with any previous group, insert with consistency counter set to zero
        // 若“当前群”与“历史一致性群” 不具有一致性，则“当前群”的一致性程度置为0，并插入到 vCurrentConsistentGroups
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }   // 遍历候选闭环帧

    // Update Co-visibility Consistent Groups，更新“一致性群”
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database，当前处理的待闭环关键帧 添加到关键帧数据库中
    mpKeyFrameDB->add(mpCurrentKF);

    // 未检测到闭环，将当前处理的待闭环关键帧设置为"可删除"
    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else    // 检测到闭环
    {
        return true;
    }

    // 下边是无效的两行代码，不会执行。
    mpCurrentKF->SetErase();
    return false;
}

// 计算待闭环帧 与 候选闭环帧群 的Sim3变换，确定闭环帧及对应的Sim3变换
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    // 对于每一个候选闭环帧，都要计算Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we set up a Sim3Solver
    // 首先计算ORB匹配，如果有足够的匹配，再进行Sim3计算
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;   // 记录每个候选闭环帧是否应该被忽略(有效帧 且 有足够的ORB匹配)
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; // candidates with enough matches，有效候选帧(具有足够ORB匹配的候选闭环帧)
    // 遍历具有足够一致的候选闭环帧，筛选出有足够ORB匹配的帧
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();     // 设置为不可删除，防止其他线程删除该帧

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else    // 为有效的候选帧构造Sim3Solver
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,
                                                 vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }   // 遍历具有足够一致的候选闭环帧，筛选出有效候选帧

    bool bMatch = false;    // 标记有效候选帧是否匹配成功()
    // Perform alternatively RANSAC iterations for each candidate
    // until one is successful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;
            // 有效的候选帧
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations，对Sim3Solver执行5次RANSAC迭代
            vector<bool> vbInliers;
            int nInliers;   // 迭代的最优内点数
            bool bNoMore;   // 迭代失败

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
            // If Ransac reaches max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            // 如果计算出了Sim3变换，用Sim3投影匹配出更多点并优化。因为之前 SearchByBoW 匹配可能会有遗漏
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(),
                                                    static_cast<MapPoint*>(NULL));
                // 取出BoW匹配的内点
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                }

                // 使用求得的Sim3变换进行匹配（跳过已经ORB匹配的点）、优化，得到最终的内点数
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);
                // 闭环帧到当前帧的Sim3变换
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t), s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF,vpMapPointMatches,
                                                             gScm, 10, mbFixScale);

                // If optimization is successful, stop ransac and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;      // 闭环帧
                    // 世界坐标系到闭环帧的变换
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),
                                   Converter::toVector3d(pKF->GetTranslation()),
                                   1.0);
                    mg2oScw = gScm * gSmw;      // 世界坐标系到当前帧的变换
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }   // 找到一个闭环帧就退出循环

    // 遍历完所有有效候选闭环帧，还是没有匹配到，则清空mvpEnoughConsistentCandidates
    // 并把当前待闭环帧设置为“可删除”
    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // 遍历(检索)闭环帧 及其共视帧，取出所有有效的闭环地图点
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF = mpCurrentKF->mnId;  // 防止重复添加地图点
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    // 通过Sim3变换，将闭环地图点投影到当前 待闭环帧，寻找更多匹配点
    matcher.SearchByProjection(mpCurrentKF, mScw,mvpLoopMapPoints,
                               mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    // 匹配到足够多的地图点
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i] != mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();   // 其他候选闭环帧可以被删除
        return true;
    }
    // 最终未匹配到足够多的地图点，则清空mvpEnoughConsistentCandidates
    else    // 并把当前待闭环帧设置为“可删除”
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

// 闭环纠正。调用SearchAndFuse()。
// 调整待闭环帧及其共视帧的位姿，以及观测到的地图点位置；用当前帧与闭环帧匹配的地图点替换当前帧的地图点；
// 闭环帧及其共视帧 与 当前待闭环帧及其共视帧 进行匹配，更新匹配关系(重复则融合)；
// 获取仅由闭环新增的二级共视帧，优化，添加闭环
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 先暂停局部建图线程
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    // 若正在进行全局BA，则将其停止
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrieve keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mg2oScw;       // 当前待匹配帧帧及其共视帧 的矫正Sim3变换
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        // 遍历 当前关键帧及其共视帧，获取每一帧的Sim3(未纠正 和 已纠正的)
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(),
                                        vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            cv::Mat Tiw = pKFi->GetPose();  // 共视帧pKFi的位姿
            if(pKFi != mpCurrentKF)
            {
                cv::Mat Tic = Tiw * Twc;    // 当前帧 到 共视帧的变换
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            // Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints observed by current keyframe and neighbors,
        // so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];  // 未经Sim3调整的位姿

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                // 将地图点用未矫正的位姿投影到对应帧，再用矫正过的位姿反投影得到矫正的地图点
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);    // 地图点位姿
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);

                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;  // 防止重复矫正
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *= (1./s); // [R t/s;0 1]  // 尺度归一化

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)  // 用pLoopMP替换pCurMP
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications. 闭环共视帧地图点投影到当前共视帧，替换当前共视帧的地图点
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the co-visibility graph will appear attaching both sides of the loop
    // 更新当前共视帧组之间的两级共视相连关系，得到因闭环生成的二级共视帧
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;    // 仅由闭环生成的二级共视帧
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(),
                                    vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();    // 闭环前的二级共视帧

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(),
                                        vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);     // 从闭环后的二级共视帧 中 删除 闭环前的，留下仅因闭环生成的二级共视帧
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(),
                                        vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);         // 再从中删除 一级共视帧
        }
    }

    // Optimize graph，进行本质图优化，优化本质图中所有关键帧的位姿和地图点
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF,
                                      NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this, mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

// 将闭环关键帧及其共视关键帧 观测到的地图点 投影到已矫正的当前关键帧及其共视帧上，进行匹配融合(替换当前帧的地图点)
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(),
                                        mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;
        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {   // 闭环共视帧地图点 替换 当前共视帧地图点
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}

// 请求复位指令
void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)   // 等待复位完成
            break;
        }
        usleep(5000);
    }
}

// 有复位指令，进行复位
void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid = 0;
        mbResetRequested=false;
    }
}

// 全局BA线程，主函数（nLoopKF传入当前关键帧的id）
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx = mnFullBAIdx;

    // 优化所有关键帧位姿、及地图中的地图点
    Optimizer::GlobalBundleAdjustment(mpMap, 10, &mbStopGBA, nLoopKF, false);

    // Update all MapPoints and KeyFrames, 更新所有的地图点和关键帧
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA, and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    // 在global BA过程中local mapping线程仍然在工作，这意味着在global BA时可能有新的关键帧产生，但是并未包括在GBA里，
    // 所以和更新后的地图并不一致。需要通过spanning tree来传播
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx != mnFullBAIdx)  // 该过程中GBA意外结束（mnFullBAIdx发生了变化）
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            // 从第一个关键帧开始矫正关键帧，更新扩展树。（刚开始只保存初始关键帧）
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();    // 初始关键帧位姿
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF != nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose() * Twc;  // 初始关键帧 到 子关键帧的变换矩阵
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;   // 优化后的子关键帧位姿
                        pChild->mnBAGlobalForKF = nLoopKF;          // 该子关键帧已经参与关键帧nLoopKF引发的GBA标记

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];
                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF == nLoopKF)
                {   // 该地图点已经参与了nLoopKF引发的GBA
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else    // 该地图点未参与全局BA，根据其参考关键帧的位姿来更新
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                    // 参考关键帧必须参与到此次GBA中
                    if(pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    // Map to non-corrected camera，获取GBA优化前的相机坐标
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                    // Back project using corrected camera
                    // 用矫正后的参考关键帧位姿，将其反投影到世界坐标系
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc + twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
