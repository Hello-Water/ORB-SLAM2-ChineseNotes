/**
 * PnP：已知世界坐标系下3D点坐标、对应的2D像素坐标、相机内参；求解相机位姿(外参)
 * EPnP：用世界坐标系、相机坐标系下的4个控制点表示对应坐标系下的3D点，然后ICP求解位姿，具体步骤为：
 * Step1: 用四个控制点来表达所有3D点（sigma为求和）
 *        p_w = sigma(alphas_j * pctrl_w_j), j从0到4，世界坐标系
 *        p_c = sigma(alphas_j * pctrl_c_j), j从0到4，相机坐标系
 *        sigma(alphas_j) = 1,  j从0到4
 * Step2: 根据相机投影模型（1个点对）
 *        s * u = K * sigma(alphas_j * pctrl_c_j), j从0到4
 * Step3: 将step2的式子展开, 消去s，得：（1个点对可以得到2个方程）
 *        sigma(alphas_j * fx * Xctrl_c_j) + alphas_j * (u0-u)*Zctrl_c_j = 0
 *        sigma(alphas_j * fy * Xctrl_c_j) + alphas_j * (v0-u)*Zctrl_c_j = 0
 * Step4: 将step3中的12个未知数提成列向量，TODO：难点是 求解4个控制点的相机坐标
 *        Mx = 0, 计算得到初始的解x后可以用 Gauss-Newton 来优化得到四个相机坐标系下的控制点
 * Step5: 根据得到的 p_w 和对应的 p_c，用ICP求解出R、t
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "PnPsolver.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;

namespace ORB_SLAM2 {

// 构造函数，根据当前帧及地图点匹配关系，初始化成员变量、设置RANSAC参数
PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches) :
        pws(0), us(0), alphas(0), pcs(0),       // 这里的四个变量都是指针
        maximum_number_of_correspondences(0), number_of_correspondences(0),
        mnInliersi(0), mnIterations(0), mnBestInliers(0), N(0) {

    // 根据当前帧中匹配到的地图点数，初始化容器大小
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());       // 被使用特征点在原始特征点容器中的索引，不连续
    mvAllIndices.reserve(F.mvpMapPoints.size());            // 被使用特征点的索引，连续的

    int idx = 0;        // 特征点索引，连续的
    // 遍历给定的地图点
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
        MapPoint *pMP = vpMapPointMatches[i];
        // 地图点存在且有效
        if (pMP) {
            if (!pMP->isBad()) {
                const cv::KeyPoint &kp = F.mvKeysUn[i];             // 对应的特征点

                mvP2D.push_back(kp.pt);
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);     // 特征点所在金字塔层的缩放因子平方

                cv::Mat Pos = pMP->GetWorldPos();                   // 地图点的世界坐标
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

                mvKeyPointIndices.push_back(i);                     // 被使用特征点在原始特征点容器中的索引，不连续
                mvAllIndices.push_back(idx);                        // 被使用特征点的索引，连续的

                idx++;
            }   // 地图点有效
        }       // 地图点存在
    }

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}

PnPsolver::~PnPsolver() {
    delete[] pws;
    delete[] us;
    delete[] alphas;
    delete[] pcs;
}

// 设置RANSAC参数
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations,
                                    int minSet, float epsilon, float th2) {
    // ransac参数
    mRansacProb = probability;          // 计算理论迭代次数时用到的概率
    mRansacMinInliers = minInliers;     // 退出ransac时需要的最小内点数（给定值，会根据匹配数调整）
    mRansacMaxIts = maxIterations;      // 设置ransac最大迭代次数（会有调整）
    mRansacEpsilon = epsilon;           // 期望得到的 内点数/总数 比值（一个阈值，会有调整）
    mRansacMinSet = minSet;             // 求解问题需要的最小样本数（最小集，默认为4）

    N = mvP2D.size();                   // 匹配到的特征点数目，number of correspondences
    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    // 调整最小内点数，max(给定内点数、最小集、理论最小内点数)
    int nMinInliers = N * mRansacEpsilon;   // 根据epsilon 和 特征点数 确定的理论最小内点数
    if (nMinInliers < mRansacMinInliers)
        nMinInliers = mRansacMinInliers;
    if (nMinInliers < minSet)
        nMinInliers = minSet;
    mRansacMinInliers = nMinInliers;

    // 根据最小内点数 调整 epsilon比值
    if (mRansacEpsilon < (float) mRansacMinInliers / N)
        mRansacEpsilon = (float) mRansacMinInliers / N;

    // 确定最大迭代次数
    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;        // 理论上的最大迭代次数
    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));
    mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

    // 计算不同金字塔层上的特征点在进行内点检验的时候,所使用的不同误差阈值
    mvMaxError.resize(mvSigma2.size());
    for (size_t i = 0; i < mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i] * th2;
}

// 获取当前迭代的位姿变换矩阵Tcw（refine后的 或 历次迭代中最优的）
cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers) {
    bool bFlag;     // 是否达到最大迭代次数
    return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
}

// EPnP迭代计算,返回Tcw
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers) {
    bNoMore = false;    // 是否达到最大迭代次数
    vbInliers.clear();
    nInliers = 0;

    // 每次RANSAC迭代需要的特征点数，默认为4组3D-2D对应点（n=4）
    set_maximum_number_of_correspondences(mRansacMinSet);

    // 若所有的特征点数目<要求的最小内点数，直接退出，返回空矩阵
    if (N < mRansacMinInliers) {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;   // 每次ransac迭代可用的特征点索引
    int nCurrentIterations = 0;         // 当前迭代次数
    // 进行迭代
    while (mnIterations < mRansacMaxIts || nCurrentIterations < nIterations) {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();        // 清空匹配点对计数number_of_correspondences，为新的一次迭代作准备

        vAvailableIndices = mvAllIndices;   // 第一次迭代，可用的特征点为所有特征点

        // Get min set of points，随机选取4组（mRansacMinSet=4）
        for (short i = 0; i < mRansacMinSet; ++i) {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randi];

            // 添加3D-2D匹配点对至pws、us
            add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z,
                               mvP2D[idx].x, mvP2D[idx].y);

            // 从"可用索引表"中删除已经被使用的点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        // 使用EPnP算法，计算某次迭代的相机位姿R，t，返回重投影误差
        compute_pose(mRi, mti);

        // Check inliers
        // 通过上边求解的位姿进行3D-2D投影，统计内点数目
        CheckInliers();

        // 若当前次迭代的内点数目 > 退出ransac所需的内点数
        if (mnInliersi >= mRansacMinInliers) {
            // If it is the best solution so far, save it
            // 若当前次迭代的内点数是历次迭代中最多的，更新最佳结果
            if (mnInliersi > mnBestInliers) {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3, 3, CV_64F, mRi);
                cv::Mat tcw(3, 1, CV_64F, mti);
                Rcw.convertTo(Rcw, CV_32F);
                tcw.convertTo(tcw, CV_32F);
                mBestTcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(mBestTcw.rowRange(0, 3).col(3));
            }

            // 若对R，t求精成功，则返回求精后的Tcw
            if (Refine()) {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
                for (int i = 0; i < N; i++) {
                    if (mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }   // while迭代

    // 已经迭代的次数超过了程序中给定的最大迭代次数
    if (mnIterations >= mRansacMaxIts) {
        bNoMore = true;
        if (mnBestInliers >= mRansacMinInliers) {   // 最优内点数满足要求
            nInliers = mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
            for (int i = 0; i < N; i++) {
                if (mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return cv::Mat();
}

// 使用最优内点的匹配对，再次EPnP求解，优化相机位姿；然后用优化后的位姿再次3D-2D投影，统计内点数目
bool PnPsolver::Refine() {
    // 最优内点的索引
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());
    for (size_t i = 0; i < mvbBestInliers.size(); i++) {
        if (mvbBestInliers[i]) {
            vIndices.push_back(i);
        }
    }

    // 根据最优内点数，设置匹配对关系
    set_maximum_number_of_correspondences(vIndices.size());
    reset_correspondences();
    for (size_t i = 0; i < vIndices.size(); i++) {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers
    CheckInliers();

    mnRefinedInliers = mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    // TODO: 难道不应该是mnInliersi > mvbBestInliers?
    if (mnInliersi > mRansacMinInliers) {
        cv::Mat Rcw(3, 3, CV_64F, mRi);
        cv::Mat tcw(3, 1, CV_64F, mti);
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(mRefinedTcw.rowRange(0, 3).col(3));
        return true;    // 若求精后的内点数>退出ransac要求的内点数
    }

    return false;
}

// 使用求解得到的位姿mRi、mti 进行3D-2D投影，统计内点数目
void PnPsolver::CheckInliers() {
    mnInliersi = 0;
    for (int i = 0; i < N; i++) {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        float Xc = mRi[0][0] * P3Dw.x + mRi[0][1] * P3Dw.y + mRi[0][2] * P3Dw.z + mti[0];
        float Yc = mRi[1][0] * P3Dw.x + mRi[1][1] * P3Dw.y + mRi[1][2] * P3Dw.z + mti[1];
        float invZc = 1 / (mRi[2][0] * P3Dw.x + mRi[2][1] * P3Dw.y + mRi[2][2] * P3Dw.z + mti[2]);

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        float distX = P2D.x - ue;
        float distY = P2D.y - ve;

        // 计算重投影误差，满足阈值要求则为内点，否则不是内点
        float error2 = distX * distX + distY * distY;
        if (error2 < mvMaxError[i]) {
            mvbInliersi[i] = true;
            mnInliersi++;
        }
        else {
            mvbInliersi[i] = false;
        }
    }
}

// 设置最大匹配点对
void PnPsolver::set_maximum_number_of_correspondences(int n) {
    if (maximum_number_of_correspondences < n) {
        if (pws != 0) delete[] pws;
        if (us != 0) delete[] us;
        if (alphas != 0) delete[] alphas;
        if (pcs != 0) delete[] pcs;

        maximum_number_of_correspondences = n;
        pws = new double[3 * maximum_number_of_correspondences];
        us = new double[2 * maximum_number_of_correspondences];
        alphas = new double[4 * maximum_number_of_correspondences];
        pcs = new double[3 * maximum_number_of_correspondences];
    }
}

void PnPsolver::reset_correspondences(void) {
    number_of_correspondences = 0;
}

// 添加3D-2D匹配点对 至 pws、us
void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v) {
    pws[3 * number_of_correspondences] = X;
    pws[3 * number_of_correspondences + 1] = Y;
    pws[3 * number_of_correspondences + 2] = Z;

    us[2 * number_of_correspondences] = u;
    us[2 * number_of_correspondences + 1] = v;

    number_of_correspondences++;
}

// 确定4个控制点 世界坐标系下的坐标 (SVD分解)
void PnPsolver::choose_control_points(void) {
    // Take C0 as the reference points centroid:
    // 第一个控制点，定为所有地图点的质心
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];
    for (int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;


    // Take C1, C2, and C3 from PCA on the reference points:
    // 对所有特征点去质心，构成nx3矩阵PW0；然后对该矩阵进行PCA(主成分分析)，确定其他三个控制点
    CvMat *PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

    double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
    CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);   // (A^T)A，A的转置 乘以 A
    CvMat DC = cvMat(3, 1, CV_64F, dc);             // 特征值
    CvMat UCt = cvMat(3, 3, CV_64F, uct);           // 特征向量

    // 对所有特征点去质心，构成nx3矩阵PW0
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

    cvMulTransposed(PW0, &PW0tPW0, 1);  // (A^T)A,order=1    A(A^T),order=0

    // SVD分解（特征值分解，DC特征值，UCt特征向量）
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    cvReleaseMat(&PW0);     // 释放内存

    // 由特征值、特征向量确定其他3个控制点
    for (int i = 1; i < 4; i++) {       // 3个控制点
        double k = sqrt(dc[i - 1] / number_of_correspondences);
        for (int j = 0; j < 3; j++)     // 每个控制点的3个坐标值
            cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];       // 为3个控制点加上质心
    }
}

// 计算世界坐标系下4个控制点的系数alphas，在相机坐标系下系数不变（其实就是控制点坐标系下的坐标）
// 每一个3D点，都有一组alphas与之对应
void PnPsolver::compute_barycentric_coordinates(void) {
    // 初始化去质心后的3个控制点坐标（第一个控制点为原点）
    double cc[3 * 3], cc_inv[3 * 3];
    CvMat CC = cvMat(3, 3, CV_64F, cc);
    CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

    // 3个控制点 去 质心（减去第一个控制点的坐标），每个控制点是cc中的列向量（cws中的行向量表示控制点）
    for (int i = 0; i < 3; i++)
        for (int j = 1; j < 4; j++)
            cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

    cvInvert(&CC, &CC_inv, CV_SVD);
    double *ci = cc_inv;                // 去质心后控制点矩阵的逆

    // 遍历所有3D地图点，求出对应的alphas系数
    for (int i = 0; i < number_of_correspondences; i++) {
        double *pi = pws + 3 * i;       // 第i个3D地图点，首地址
        double *a = alphas + 4 * i;     // 第i个控制点系数，首地址

        // 计算4个控制点系数（先计算后3个: a1,a2,a3）
        for (int j = 0; j < 3; j++)
            a[1 + j] = ci[3 * j]     * (pi[0] - cws[0][0]) +
                       ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                       ci[3 * j + 2] * (pi[2] - cws[0][2]);
        // 再计算a0=1-a1-a2-a3
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

// 构造M矩阵，每对匹配点可以填充2行 (用于求vi，进一步求4个控制点在相机坐标系下的坐标，12x1的列向量)
// 利用相机坐标系到像素坐标系的变换，用相机坐标系下的4个控制点 表示 相机坐标系下的3D点
// as为控制点系数（世界坐标系下 与 相机坐标系下相同），[u,v]是3D地图点的像素坐标
void PnPsolver::fill_M(CvMat *M, const int row, const double *as,
                       const double u, const double v) {
    double *M1 = M->data.db + row * 12;     // M矩阵，第row个点对，对应的第一行起始位置
    double *M2 = M1 + 12;                   // M矩阵，第row个点对，对应的第二行起始位置

    // 每个点对形成的两个方程，对应到4个控制点 的 系数（Mx=0,x为4个控制点组成的列向量）
    for (int i = 0; i < 4; i++) {
        M1[3 * i] = as[i] * fu;
        M1[3 * i + 1] = 0.0;
        M1[3 * i + 2] = as[i] * (uc - u);

        M2[3 * i] = 0.0;
        M2[3 * i + 1] = as[i] * fv;
        M2[3 * i + 2] = as[i] * (vc - v);
    }
}

// 计算4个控制点在相机坐标系下的坐标（ 根据已计算出的betas、vi(ut) ）
// 特征向量ut是经SVD得到的，维度为(12,12)，特征值0对应的特征向量为最后4行
void PnPsolver::compute_ccs(const double *betas, const double *ut) {
    // 初始化为0
    for (int i = 0; i < 4; i++)
        ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

    // 直接设置零空间自由度为4 (即N=4)
    for (int i = 0; i < 4; i++) {       // 4个特征向量
        const double *v = ut + 12 * (11 - i);
        for (int j = 0; j < 4; j++)     // 4个控制点
            for (int k = 0; k < 3; k++) // 3个坐标
                ccs[j][k] += betas[i] * v[3 * j + k];
    }
}

// 计算3D地图点在相机坐标系下的坐标（用相机坐标系下的4个控制点、控制系数计算）
void PnPsolver::compute_pcs(void) {
    for (int i = 0; i < number_of_correspondences; i++) {
        double *a = alphas + 4 * i;
        double *pc = pcs + 3 * i;

        for (int j = 0; j < 3; j++)
            pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
    }
}

// 使用EPnP算法，计算某次迭代的相机位姿R，t，返回重投影误差
double PnPsolver::compute_pose(double R[3][3], double t[3]) {
    // 1. 确定4个控制点cws[4][3]
    choose_control_points();

    // 2. 计算每个地图点对应的4个控制点系数a[4]
    compute_barycentric_coordinates();

    // 3. 构造M矩阵
    CvMat *M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++)
        fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

    // 4. svd求解Mx=0 (求不同情况下4个控制点在相机坐标系下的坐标)
    // 4.1 求MtM的特征向量vi (Ut)
    double mtm[12 * 12], d[12], ut[12 * 12];
    CvMat MtM = cvMat(12, 12, CV_64F, mtm);
    CvMat D = cvMat(12, 1, CV_64F, d);
    CvMat Ut = cvMat(12, 12, CV_64F, ut);

    cvMulTransposed(M, &MtM, 1);    // (A^T)A,order=1    A(A^T),order=0
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    cvReleaseMat(&M);

    // 4.2 计算L (N=4时，维度6x10，L是beta的系数)
    //     和 rho (4个控制点两两之间的距离，6x1)
    //     用于求beta，L*beta = rho，不同的N，L有不同的维度
    double l_6x10[6 * 10], rho[6];
    CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    CvMat Rho = cvMat(6, 1, CV_64F, rho);

    compute_L_6x10(ut, l_6x10);
    compute_rho(rho);

    // 4.3 分情况计算beta(N=2,3,4，与相机模型有关)
    double Betas[4][4], rep_errors[4];  // 其实值计算了3种情况
    double Rs[4][3][3], ts[4][3];

    // 4.3.1 N=4时的beta值、R|t、重投影误差
    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);           // beta近似解
    gauss_newton(&L_6x10, &Rho, Betas[1]);            // 高斯-牛顿优化beta
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);  // 计算当前情况下R,t 导致的平均重投影误差

    // 4.3.2 N=2时的beta值、R|t、重投影误差
    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

    // 4.3.3 N=3时的beta值、R|t、重投影误差
    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

    // 5. 比较不同情况下的重投影误差，选择最小的
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    // 6. 保存最小重投影误差对应的R，t
    copy_R_and_t(Rs[N], ts[N], R, t);

    // 7. 返回最小重投影误差
    return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
                             double R_dst[3][3], double t_dst[3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            R_dst[i][j] = R_src[i][j];
        t_dst[i] = t_src[i];
    }
}

double PnPsolver::dist2(const double *p1, const double *p2) {
    return
            (p1[0] - p2[0]) * (p1[0] - p2[0]) +
            (p1[1] - p2[1]) * (p1[1] - p2[1]) +
            (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double *v1, const double *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// 计算(平均)重投影误差，匹配对像素坐标-3D点重投影像素坐标
double PnPsolver::reprojection_error(const double R[3][3], const double t[3]) {
    double sum2 = 0.0;
    for (int i = 0; i < number_of_correspondences; i++) {
        double *pw = pws + 3 * i;
        double Xc = dot(R[0], pw) + t[0];
        double Yc = dot(R[1], pw) + t[1];
        double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
        double ue = uc + fu * Xc * inv_Zc;
        double ve = vc + fv * Yc * inv_Zc;

        double u = us[2 * i],
               v = us[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }

    return sum2 / number_of_correspondences;
}

// ICP求解相机位姿（被compute_R_and_t调用），用3d点在 世界坐标系 和 相机坐标系下 的坐标
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3]) {
    // 计算3D地图点在 相机坐标系、世界坐标系 下的质心
    double pc0[3], pw0[3];
    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;
    for (int i = 0; i < number_of_correspondences; i++) {
        const double *pc = pcs + 3 * i;
        const double *pw = pws + 3 * i;

        for (int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        pc0[j] /= number_of_correspondences;
        pw0[j] /= number_of_correspondences;
    }

    // 计算H=B^T*A（即ABt），A是3D点在世界坐标系下的去质心矩阵，B是3D点在相机坐标系下的去质心矩阵
    double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
    CvMat ABt = cvMat(3, 3, CV_64F, abt);       // H=B^T*A
    CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);   // svd分解得到的特征值
    CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);   // svd分解得到的左特征矩阵
    CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);   // svd分解得到的右特征矩阵

    cvSetZero(&ABt);
    for (int i = 0; i < number_of_correspondences; i++) {
        double *pc = pcs + 3 * i;
        double *pw = pws + 3 * i;
        // 去质心后的对应坐标乘累加
        for (int j = 0; j < 3; j++) {
            abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    // 对H矩阵进行svd分解
    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

    // R=U*V^T
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);
    // 旋转矩阵行列式 det(R)=1>0
    const double det =
            R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
            R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];
    if (det < 0) {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    // t = pc0 - R*pwo
    t[0] = pc0[0] - dot(R[0], pw0);
    t[1] = pc0[1] - dot(R[1], pw0);
    t[2] = pc0[2] - dot(R[2], pw0);
}

void PnPsolver::print_pose(const double R[3][3], const double t[3]) {
    cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
    cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
    cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

// 保证3D点在相机坐标系下的深度为正（在相机前方）
void PnPsolver::solve_for_sign(void) {
    if (pcs[2] < 0.0) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                ccs[i][j] = -ccs[i][j];

        for (int i = 0; i < number_of_correspondences; i++) {
            pcs[3 * i] = -pcs[3 * i];
            pcs[3 * i + 1] = -pcs[3 * i + 1];
            pcs[3 * i + 2] = -pcs[3 * i + 2];
        }
    }
}

// 根据beta、vi求控制点在相机坐标系下的坐标、3d点在相机坐标系下坐标，恢复相机R、t
// 然后使用该位姿R,t，计算重投影误差
double PnPsolver::compute_R_and_t(const double *ut, const double *betas,
                                  double R[3][3], double t[3]) {
    compute_ccs(betas, ut);             // 4个控制点在相机坐标系下的坐标

    compute_pcs();                      // 3D地图点在相机坐标系下的坐标，用控制点计算

    solve_for_sign();                   // 保证3D点在相机坐标系下的深度为正

    estimate_R_and_t(R, t);             // (3D-3D)ICP估计R，t

    return reprojection_error(R, t);    // 返回此R,t下的重投影误差
}

// N = 4，beta近似值
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
void PnPsolver::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
    // 从每行种取出4个(其他置为0)，求近似解
    double l_6x4[6 * 4], b4[4];
    CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
    CvMat B4 = cvMat(4, 1, CV_64F, b4);

    // 提取L_6x10矩阵中第0,1,3,6列元素，得到L_6x4
    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
    }

    // SVD求解 L_6x4 * B4 = Rho
    cvSolve(&L_6x4, Rho, &B4, CV_SVD);

    if (b4[0] < 0) {    // b11理论上 >=0
        betas[0] = sqrt(-b4[0]);          // b1
        betas[1] = -b4[1] / betas[0];       // b2
        betas[2] = -b4[2] / betas[0];       // b3
        betas[3] = -b4[3] / betas[0];       // b4
    }
    else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// N = 2，beta近似值
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
void PnPsolver::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
    double l_6x3[6 * 3], b3[3];
    CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
    CvMat B3 = cvMat(3, 1, CV_64F, b3);

    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
    }

    cvSolve(&L_6x3, Rho, &B3, CV_SVD);

    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0)
        betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// N = 3，beta近似值
// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
void PnPsolver::find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                                    double *betas) {
    double l_6x5[6 * 5], b5[5];
    CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
    CvMat B5 = cvMat(5, 1, CV_64F, b5);

    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
        cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
    }

    cvSolve(&L_6x5, Rho, &B5, CV_SVD);

    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }

    if (b5[1] < 0)
        betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

// 计算N=4时的 L矩阵，维度为6x10；ut是特征分解后的12x12矩阵
void PnPsolver::compute_L_6x10(const double *ut, double *l_6x10) {
    // 4个特征向量
    const double *v[4];
    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    // 计算中间向量dv，4个12x1的特征向量；每个向量分为4组，两两做差，有6种组合；每种是3维向量
    double dv[4][6][3];
    for (int i = 0; i < 4; i++) {
        int a = 0, b = 1;   // a,b用来标记做差的下标
        // dv[i][j]=v[i]^[a]-v[i]^[b]
        // a,b的取值有6种组合 0-1 0-2 0-3, 1-2 1-3, 2-3
        for (int j = 0; j < 6; j++) {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];            // 类似于x坐标
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];    // 类似于y坐标
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];    // 类似于z坐标

            b++;

            if (b > 3) {
                a++;
                b = a + 1;
            }
        }
    }

    // 用dv生成L矩阵，6是4个控制点做差的6种情况
    for (int i = 0; i < 6; i++) {
        double *row = l_6x10 + 10 * i;      // 每一行的首地址
        // 计算每行的10个元素
        row[0] = dot(dv[0][i], dv[0][i]);           // b11
        row[1] = 2.0f * dot(dv[0][i], dv[1][i]);    // b12
        row[2] = dot(dv[1][i], dv[1][i]);           // b22
        row[3] = 2.0f * dot(dv[0][i], dv[2][i]);    // b13
        row[4] = 2.0f * dot(dv[1][i], dv[2][i]);    // b23
        row[5] = dot(dv[2][i], dv[2][i]);           // b33
        row[6] = 2.0f * dot(dv[0][i], dv[3][i]);    // b14
        row[7] = 2.0f * dot(dv[1][i], dv[3][i]);    // b24
        row[8] = 2.0f * dot(dv[2][i], dv[3][i]);    // b34
        row[9] = dot(dv[3][i], dv[3][i]);           // b44
    }
}

// 计算四个控制点任意两点间的距离，总共6个距离
void PnPsolver::compute_rho(double *rho) {
    rho[0] = dist2(cws[0], cws[1]);
    rho[1] = dist2(cws[0], cws[2]);
    rho[2] = dist2(cws[0], cws[3]);
    rho[3] = dist2(cws[1], cws[2]);
    rho[4] = dist2(cws[1], cws[3]);
    rho[5] = dist2(cws[2], cws[3]);
}

// 计算A，b（Ax=b高斯牛顿法优化beta，betas为当前次迭代得到的beta值）
void PnPsolver::compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                             double betas[4], CvMat *A, CvMat *b) {
    for (int i = 0; i < 6; i++) {
        const double *rowL = l_6x10 + i * 10;
        double *rowA = A->data.db + i * 4;

        // 计算当前行的雅可比矩阵
        rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        // 当前行的非齐次项 rho-L*beta
        cvmSet(b, i, 0, rho[i] -
                        (       // 直接计算向量内积
                                rowL[0] * betas[0] * betas[0] +
                                rowL[1] * betas[0] * betas[1] +
                                rowL[2] * betas[1] * betas[1] +
                                rowL[3] * betas[0] * betas[2] +
                                rowL[4] * betas[1] * betas[2] +
                                rowL[5] * betas[2] * betas[2] +
                                rowL[6] * betas[0] * betas[3] +
                                rowL[7] * betas[1] * betas[3] +
                                rowL[8] * betas[2] * betas[3] +
                                rowL[9] * betas[3] * betas[3]
                        ));
    }
}

// gauss-newton优化beta，使得两个坐标系下，控制点之间的距离差，误差最小
void PnPsolver::gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                             double betas[4]) {
    // 只进行5次迭代
    const int iterations_number = 5;

    // 求解增量方程 Ax=B
    double a[6 * 4], b[6], x[4];
    CvMat A = cvMat(6, 4, CV_64F, a);
    CvMat B = cvMat(6, 1, CV_64F, b);
    CvMat X = cvMat(4, 1, CV_64F, x);

    for (int k = 0; k < iterations_number; k++) {
        compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,
                                     betas, &A, &B);
        // QR分解求当前次迭代的增量 x
        qr_solve(&A, &B, &X);

        // 用增量x 更新betas
        for (int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}

// QR分解，求解增量方程 AX=b（X即delta_beta）,Q是正交阵(列向量互相垂直)，R是上三角阵
// A=QR, (QR)x=b, Rx=(Q^T)b, x=(R^{-1})(Q^T)b
// 若A不是方阵，假设A的维度为mxn,m>=n，则 A = QR = Q|R1| = [Q1 Q2]|R1| = (Q1)(R1)
// Q1维度mxn，R1维度nxn，                        |0 |          |0 |
void PnPsolver::qr_solve(CvMat *A, CvMat *b, CvMat *X) {
    static int max_nr = 0;      // 已存在的最大行数
    static double *A1, *A2;     // TODO：A1，A2这两个中间变量意义是什么？

    const int nr = A->rows;     // A矩阵的行、列数
    const int nc = A->cols;

    // max_nr != 0 说明已经存在数组A1，A2；
    // max_nr < nr 说明已存在的数组无法满足内存要求，需要按照实际A矩阵的行数分配内存
    if (max_nr != 0 && max_nr < nr) {
        delete[] A1;
        delete[] A2;
    }
    if (max_nr < nr) {
        max_nr = nr;
        A1 = new double[nr];
        A2 = new double[nr];
    }

    // 遍历A矩阵的每一列
    double *pA = A->data.db, *ppAkk = pA;
    for (int k = 0; k < nc; k++) {
        double *ppAik = ppAkk,            // 指向 对角线元素 下边的元素，用于遍历
                eta = fabs(*ppAik);     // 存储当前列对角线元素 下面的所有元素绝对值的 最大值
        // 遍历 每列 对角线元素 下边的每一行
        for (int i = k + 1; i < nr; i++) {
            double elt = fabs(*ppAik);
            if (eta < elt)
                eta = elt;
            ppAik += nc;                    // 指向下一行
        }

        // eta == 0 说明该列对角线以下元素都为0，TODO：会导致列不满秩？
        if (eta == 0) {
            A1[k] = A2[k] = 0.0;
            cerr << "God damnit, A is singular, this shouldn't happen." << endl;
            return;
        }
        else {
            // 对每一列对角线以下元素 最大值归一化 后 求平方和
            double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
            for (int i = k; i < nr; i++) {
                *ppAik *= inv_eta;
                sum += *ppAik * *ppAik;
                ppAik += nc;
            }

            // sigma符号 与 当前列对角线元素的符号 保持一致
            double sigma = sqrt(sum);
            if (*ppAkk < 0)
                sigma = -sigma;

            // TODO:原理？
            *ppAkk += sigma;
            A1[k] = sigma * *ppAkk;
            A2[k] = -eta * sigma;
            // 遍历某列 后边的列
            for (int j = k + 1; j < nc; j++) {
                double *ppAik = ppAkk, sum = 0;
                for (int i = k; i < nr; i++) {
                    sum += *ppAik * ppAik[j - k];   // 每列对角线及以下的元素 分别乘以其后边的每一列元素，然后求和
                    ppAik += nc;
                }

                double tau = sum / A1[k];
                ppAik = ppAkk;
                for (int i = k; i < nr; i++) {
                    ppAik[j - k] -= tau * *ppAik;   // 更新某列 对角线以下的元素 后边的每一列元素值
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;    // 下一列的对角线元素
    }

    // b <- Qt b
    double *ppAjj = pA, *pb = b->data.db;
    for (int j = 0; j < nc; j++) {
        double *ppAij = ppAjj, tau = 0;
        for (int i = j; i < nr; i++) {
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1[j];
        ppAij = ppAjj;
        for (int i = j; i < nr; i++) {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b
    double *pX = X->data.db;
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for (int i = nc - 2; i >= 0; i--) {
        double *ppAij = pA + i * nc + (i + 1), sum = 0;

        for (int j = i + 1; j < nc; j++) {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2[i];
    }
}

void PnPsolver::relative_error(double &rot_err, double &transl_err,
                               const double Rtrue[3][3], const double ttrue[3],
                               const double Rest[3][3], const double test[3]) {
    double qtrue[4], qest[4];

    mat_to_quat(Rtrue, qtrue);
    mat_to_quat(Rest, qest);

    double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
                           (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
                           (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
                           (qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
                      sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
                           (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
                           (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
                           (qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
                      sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

    rot_err = min(rot_err1, rot_err2);

    transl_err =
            sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
                 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
                 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
            sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPsolver::mat_to_quat(const double R[3][3], double q[4]) {
    double tr = R[0][0] + R[1][1] + R[2][2];
    double n4;

    if (tr > 0.0f) {
        q[0] = R[1][2] - R[2][1];
        q[1] = R[2][0] - R[0][2];
        q[2] = R[0][1] - R[1][0];
        q[3] = tr + 1.0f;
        n4 = q[3];
    }
    else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
        q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
        q[1] = R[1][0] + R[0][1];
        q[2] = R[2][0] + R[0][2];
        q[3] = R[1][2] - R[2][1];
        n4 = q[0];
    }
    else if (R[1][1] > R[2][2]) {
        q[0] = R[1][0] + R[0][1];
        q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
        q[2] = R[2][1] + R[1][2];
        q[3] = R[2][0] - R[0][2];
        n4 = q[1];
    }
    else {
        q[0] = R[2][0] + R[0][2];
        q[1] = R[2][1] + R[1][2];
        q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
        q[3] = R[0][1] - R[1][0];
        n4 = q[2];
    }
    double scale = 0.5f / double(sqrt(n4));

    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}

} //namespace ORB_SLAM
