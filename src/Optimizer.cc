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

#include "Optimizer.h"
#include "Converter.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include<mutex>

namespace ORB_SLAM2
{

// 全局优化(Tracking, LoopClosing共视图)。优化地图中所有关键帧位姿、所有地图点位置。
void Optimizer::GlobalBundleAdjustment(Map* pMap, int nIterations, bool* pbStopFlag,
                                       unsigned long nLoopKF, bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

// 被GlobalBundleAdjustment调用，优化传入的关键帧位姿、地图点位置。
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF,
                                 const bool bRobust)
{
    vector<bool> vbNotIncludedMP;   // 记录不参与优化的地图点
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;

    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;  // 记录参与优化的最大关键帧id

    // Set KeyFrame vertices，添加关键帧顶点。（位姿顶点）
    for(size_t i=0; i<vpKFs.size(); i++)
    {   // 遍历所有关键帧，添加位姿顶点。
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);   // 初始关键帧不可被优化
        optimizer.addVertex(vSE3);

        if(pKF->mnId>maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++)
    {   // 遍历所有地图点，添加地图点顶点.
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);      // 地图点被边缘化
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        int nEdges = 0;     // 每个地图点参与的边数
        // SET EDGES
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {   // 遍历每个地图点的可观测关键帧,添加对应的边.
            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0)
            {   // 单目
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else {  // 双目
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        // 某个地图点未构成边,则从优化器中删除该地图点,并记录.
        if (nEdges == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    // Keyframes,更新优化后的关键帧位姿.
    for(size_t i=0; i<vpKFs.size(); i++)
    {   // 遍历所有关键帧
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();

        // 形成闭环的当前帧id==0,即初始关键帧.创建初始地图时就调用GBA.
        // 那么pKF就是第二帧,优化后的位姿可直接覆盖pKF的位姿.
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {   // 否则,优化后的位姿要写入到指定的成员变量mTcwGBA
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;     // 记录该pKF参与到触发全局BA(标记为形成闭环的关键帧id)
        }
    }

    // Points,更新优化后的地图点位置.
    for(size_t i=0; i<vpMP.size(); i++)
    {   // 遍历所有地图点.
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)  // 创建地图时就调用GBA.
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {               // 否则,优化后的地图点位置要写入到指定的成员变量mPosGBA
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

// 优化当前帧位姿(Tracking)，返回优化后的内点数。
int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;

    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;        // 当前帧中，有效参与优化的3D-2D点对

    // Set Frame vertex，添加位姿顶点(只有一个)，待优化的当前帧位姿Tcw
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);  // 待优化的变量，不能设置为Fix
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices，添加边(一元边)
    const int N = pFrame->N;    // 当前帧观测到的地图点个数

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;    // 单目边
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;    // 双目边
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const float deltaStereo = sqrt(7.815);  // 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);    // 对地图点加锁

    for(int i=0; i<N; i++)
    {   // 遍历当前帧观测到的地图点
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i] < 0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);    // 卡方阈值

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }

    // 如果没有足够多的3D-2D匹配点对，不进行优化
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 优化4次，每次优化迭代10次。每次优化后要进行卡方检验确定内点/外点，外点对应的边不参与下次优化，
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {   // 4次优化
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));  // 设置优化初始值
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);    // 优化迭代10次

        nBad = 0; // 记录每次优化的外点数

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {   // 处理单目误差边
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])     // 如果这条边来自于外点，计算重投影误差
            {
                e->computeError();
            }

            const float chi2 = e->chi2();   // 卡方检验，_error * (\Omega) * _error
            // 重投影误差较大
            if(chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;     // 设置为外点
                e->setLevel(1);                   // 外点对应的边不参与优化
                nBad++;
            }
            // 重投影误差较小，则为内点，对应的边参与下次优化
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if(it == 2)     // 只有前两次优化使用RobustKernel，因为重投影误差已经明显夏季爱嗯
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {   // 处理双目误差边
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        // 边的个数要>=10（3D-2D匹配点对数）
        if(optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    // 将优化后的位姿 设置为当前帧的位姿（注意数据类型的转换）
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;  // 返回优化后的内点数
}

// 局部BA优化(LocalMapping线程使用)。优化局部地图中的局部关键帧位姿 和 局部地图点3D位置。
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    // 构造局部关键帧：当前关键帧 + 当前关键帧的共视帧。并对局部关键帧进行标记。
    list<KeyFrame*> lLocalKeyFrames;    // 局部关键帧
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;    // 进行局部BA时，对当前关键帧标记
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;   // 进行局部BA时，对共视关键帧标记
        if(!pKFi->isBad())                  // TODO 按照下边局部地图点的写法才会防止重复添加
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // 构造局部地图点：局部关键帧观测到的有效地图点。并对局部地图点进行标记。
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;    // 对局部地图点进行标记
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 构造固定关键帧(二级共视帧)：可以观测到局部地图点的关键帧，但不在局部关键帧中。并对其进行标记。
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;   // 标记固定关键帧
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;

    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    if(pbStopFlag)  // 外部(LocalMapping线程)要求停止局部BA线程
        optimizer.setForceStopFlag(pbStopFlag);

    // Set Local KeyFrame vertices，添加局部关键帧顶点(待优化位姿)。
    unsigned long maxKFid = 0;  // 记录参与局部BA的最大局部关键帧id
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);  // 不对初始关键帧进行优化
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices，添加固定关键帧顶点(位姿不被优化，只提供约束)。
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);           // 固定关键帧位姿不被优化，只提供约束
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // 参与构成边的可能最大地图点个数
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size()) * lLocalMapPoints.size();

    // 单目
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;    // 记录构成的边(二元边)
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;                 // 记录构成边的关键帧
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;           // 记录构成边的地图点
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // 双目
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // Set MapPoint vertices，遍历所有局部地图点，添加局部地图点顶点，及对应的边。
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {   // 遍历所有局部地图点，添加局部地图点顶点
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;   // 地图点顶点id
        vPoint->setId(id);
        vPoint->setMarginalized(true);      // 将3D地图点边缘化。
        optimizer.addVertex(vPoint);

        // Set edges，遍历局部地图点的可观测帧，添加对应边。
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {   // 遍历局部地图点的可观测帧
            KeyFrame* pKFi = mit->first;
            if(!pKFi->isBad())
            {   // 判断关键帧pKFi是否是bad
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];     // 3D地图点在该帧上的特征点

                // Monocular observation
                if(pKFi->mvuRight[mit->second] < 0)
                {   // 单目
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo observation
                else
                {   // 双目
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)      // 开始BA之前，再次确认外部是否有停止BA的指令
        if(*pbStopFlag)
            return;

    // 第一次优化，迭代5次
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;
    if(pbStopFlag)      // 再次检查外部是否有停止BA的指令
        if(*pbStopFlag)
            bDoMore = false;

    if (bDoMore) {
        // Check inlier observations，剔除误差较大的边，不参与第二次优化
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);         // 第二次不优化误差较大的边
            }

            e->setRobustKernel(0);    // 第二次优化不使用核函数
        }   // 单目

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }   // 双目

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;    // 记录第二次优化后，将要被删除的<关键帧，地图点>
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations，检查第二次优化后的外点ToErase
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }   // 单目

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {   // 遍历将要被删除的<关键帧，地图点>，删除对应观测关系
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // Keyframes，更新局部关键帧位姿。(当前关键帧及其共视帧)
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();    // 优化后的四元数位姿
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    // Points，更新局部地图点。(局部关键帧观测到的地图点)
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();    // 更新平均观测方向和深度
    }
}

// 全局本质图优化(LoopClosing). 加入闭环约束, 优化所有关键帧位姿, 地图点.
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);    // 关闭调试输出

    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-16);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);          // 存储优化前的位姿(已经过Sim3调整)
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);  // 存储优化后的位姿
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);                              // 记录关键帧id对应的Sim3顶点

    const int minFeat = 100;    // 最小共视程度

    // Set KeyFrame vertices, 添加关键帧位姿顶点.
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {   // 遍历所有关键帧, 添加位姿顶点
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        // CorrectedSim3, 经过Sim3传播调整过的关键帧位姿, 优先使用
        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);
        if(it != CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {   // 如果该关键帧在闭环时没有通过Sim3传播调整过，用跟踪时的位姿，尺度为1
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF == pLoopKF)
            VSim3->setFixed(true);  // 闭环帧不被优化, 但初始关键帧做了优化.

        VSim3->setId(nIDi);         // 使用当前关键帧的id
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;  // 双目/RGB-D的尺度因子固定,不优化; 优化单目的尺度因子

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }

    // 记录因闭环新生成的连接关系<一级共视帧,二级共视帧集合> 构造成的边<小id, 大id>
    set<pair<long unsigned int, long unsigned int> > sInsertedEdges;
    // 单位矩阵, 闭环边的信息矩阵
    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges, 添加新增闭环边. LoopConnections类型为map<一级共视帧, 二级共视帧集合>
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(),
                                                          mend = LoopConnections.end(); mit!=mend; mit++)
    {   // 遍历因闭环新生成共视帧对<一级共视帧, 二级共视帧集合>, 添加约束边
        KeyFrame* pKF = mit->first;                         // 一级共视帧
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;  // 二级共视帧
        const g2o::Sim3 Siw = vScw[nIDi];       // 取出一级共视帧的位姿
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {   // 遍历二级共视帧
            const long unsigned int nIDj = (*sit)->mnId;

            // 一级共视帧来自当前帧, 二级共视帧来自闭环帧, 二者的共视程度要>=100, 才可以构成约束
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];   // 取出二级共视帧的位姿
            const g2o::Sim3 Sji = Sjw * Swi;    // 计算一级共视帧到二级共视帧之间的Sim3变换

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges, 添加普通边(未经Sim3调整的位姿构成的边):跟踪的边, 闭环匹配的边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {   // 遍历所有关键帧, 添加普通边 (未经Sim3调整的位姿构成的边)
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        if(iti != NonCorrectedSim3.end())       // 优先使用未经Sim3调整的位姿
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent(); // 父关键帧
        // Spanning tree edge, 添加生成树的边 (父子边)
        if(pParentKF)
        {   // 添加父子边
            int nIDj = pParentKF->mnId;         // 父关键帧的id

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);
            if(itj != NonCorrectedSim3.end())   // 优先使用未经Sim3变换的位姿
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;          // 子关键帧到父关键帧之间的变换

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges, 添加普通闭环边.(与当前帧形成闭环关系的所有关键帧)
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {   // 遍历当前帧的所有闭环帧, 添加相应闭环边
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId < pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);
                if(itl!=NonCorrectedSim3.end())     // 优先使用未经Sim3调整的位姿
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;          // 当前帧到闭环帧的Sim3变换
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Co-visibility graph edges, 添加共视边(共视程度超过100的边)
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {   // 遍历的当前帧的共视帧, 添加相应边
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {   // 防止重复添加
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {   // 共视帧不是bad, 且id要小于当前帧id
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
                    if(itn!=NonCorrectedSim3.end())         // 优先使用未经Sim3调整的位姿
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;              // 当前关键帧 到 共视帧 之间的变换

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 更新优化后的关键帧位姿.
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();    // g2o优化后的位姿
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1./s); // [R t/s;0 1], 转换为尺度为1的变换矩阵的形式

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 更新优化后的地图点位置. 将地图点用未优化的位姿投影到对应帧，再用优化后的位姿反投影得到优化的地图点
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {   // 遍历所有地图点,更新地图点位置
        MapPoint* pMP = vpMPs[i];
        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF == pCurKF->mnId)    // 该地图点被闭环调整过
        {
            nIDr = pMP->mnCorrectedReference;       // 被修正过的参考关键帧帧id
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];                     // 该地图点的参考关键帧位姿 (Sim3调整过,但未经g2o优化)
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];   // g2o优化后的参考关键帧位姿

        // 将地图点用未优化的位姿投影到对应帧，再用优化后的位姿反投影得到优化的地图点
        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

// 优化两帧之间的Sim3变换(LoopClosing线程使用)，返回优化后的匹配点对数。
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;

    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex，添加顶点(Sim3变换)，(x,y,z,qw,qx,qy,qz)
    g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);     // Sim3变换初始值
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);    // cx1
    vSim3->_principle_point1[1] = K1.at<float>(1,2);    // cy1
    vSim3->_focal_length1[0] = K1.at<float>(0,0);       // fx1
    vSim3->_focal_length1[1] = K1.at<float>(1,1);       // fy1
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices，添加边(二元边)，要将地图点投影到两帧上。
    const int N = vpMatches1.size();    // 两帧匹配到的(共视)地图点
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();  // pKF1的所有地图点(有对应特征点的地图点)
    // 两类边 (两个投影方向)
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;   // 参与优化的匹配点对数

    for(int i=0; i<N; i++)
    {   // 遍历每对匹配点，添加两类边。i是pKF1中的索引
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i + 1;    // pKF1上的地图点构成的图顶点的id
        const int id2 = 2 * (i+1);    // pKF2上的对应地图点构成的图顶点的id

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        // 添加地图点顶点(匹配点对相机坐标 对应的图顶点)
        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {   // 匹配地图点对存在且有效，且在图像帧上都有对应的特征点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;    // 地图点的世界坐标转化为相机坐标
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);            // 不优化地图点
                optimizer.addVertex(vPoint1);     // 添加pKF1地图点到图顶点

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);            // 不优化地图点
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // 添加两类二元边。一类是投影到pKF1上，一类是投影到pKF2上。
        // Set edge x1 = S12*X2，闭环候选帧地图点投影到当前关键帧 的边 -- 正向投影
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));   // 位姿顶点
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);

        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1，当前帧地图点投影到闭环候选帧 的边 -- 反向投影
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);

        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);

        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers，第一次优化后，进行卡方检验，剔除误差较大的边。
    int nBad = 0;   // 卡方检验出的外点数
    for(size_t i=0; i<vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        // 卡方检验，e12->chi2() = _error12 * (\Omega1) * _error12
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {   // 误差较大
            size_t idx = vnIndexEdge[i];        // 构造边的第i个匹配点对，匹配点对在pKF1中的索引
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;    // 额外的迭代次数
    if(nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    // 剩余有效匹配点数郭少，则不进行优化
    if(nCorrespondences-nBad < 10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    // 第二次优化后，统计通过卡方检验的匹配点数
    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
