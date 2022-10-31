/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "MapPoint.h"
#include "Frame.h"

namespace ORB_SLAM2 {

    // EPnP：已知世界坐标系下3D点坐标、对应的2D像素坐标、相机内参；求解相机位姿(外参)
    class PnPsolver {

    public:
        PnPsolver(const Frame &F, const vector<MapPoint *> &vpMapPointMatches);

        ~PnPsolver();

        // 设置Ransac迭代参数
        void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300,
                                 int minSet = 4, float epsilon = 0.4, float th2 = 5.991);

        // 获取内点标记、内点数量
        cv::Mat find(vector<bool> &vbInliers, int &nInliers);

        // 迭代计算，返回相机位姿 Tcw
        cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

    private:

        // 使用求解得到的位姿Tcw 进行3D-2D投影，统计内点数目
        void CheckInliers();

        // 使用最优内点的匹配对，再次EPnP求解，优化相机位姿；用优化后的位姿再次3D-2D投影，统计内点数目
        // 返回的结果(bool类型)表示经过优化后的内点数,能否达到退出RANSAC的要求
        bool Refine();

        // ------------- Functions from the original EPnP code --------------

        // 设置最大匹配数，每次RANSAC计算的过程中使用的匹配点对的最大值（至少为n）
        void set_maximum_number_of_correspondences(const int n);

        // 清空当前已有的匹配点对计数,为进行新的一次迭代作准备
        void reset_correspondences(void);

        // 添加匹配点对
        void add_correspondence(const double X, const double Y, const double Z,
                                const double u, const double v);

        // EPnP计算相机位姿，旋转、平移
        double compute_pose(double R[3][3], double T[3]);

        // 计算相对误差， 真值-估计值
        void relative_error(double &rot_err, double &transl_err,
                            const double Rtrue[3][3], const double ttrue[3],
                            const double Rest[3][3], const double test[3]);

        void print_pose(const double R[3][3], const double t[3]);

        // 计算(平均)重投影误差，匹配对像素坐标-3D点重投影像素坐标
        double reprojection_error(const double R[3][3], const double t[3]);

        // 从给定的匹配3D点中计算出4个控制点（世界坐标系下）
        void choose_control_points(void);

        // 计算4个控制点的系数 alpha_ij
        void compute_barycentric_coordinates(void);

        // 构造M矩阵，每对匹配点可以填充2行
        void fill_M(CvMat *M, const int row, const double *alphas, const double u, const double v);

        // 计算4个控制点在相机坐标系下的坐标（根据计算出的beta、vi）
        void compute_ccs(const double *betas, const double *ut);

        // 计算3D点在相机坐标系下的坐标（根据4个控制点相机坐标、控制点系数计算）
        void compute_pcs(void);

        // 保证3D点在相机坐标系下的深度为正（在相机前方）
        void solve_for_sign(void);

        // 计算N=4时的 L矩阵
        void compute_L_6x10(const double *ut, double *l_6x10);

        // 计算N=4、N=2、N=3时的beta近似解，将其他beta变量置为0
        void find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas);
        void find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho, double *betas);
        void find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho, double *betas);

        // 高斯牛顿法优化beta
        void gauss_newton(const CvMat *L_6x10, const CvMat *Rho, double current_betas[4]);

        // 计算(构造)A，b（Ax=b高斯牛顿法优化beta，cb为beta的粗略解）
        void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                          double cb[4], CvMat *A, CvMat *b);

        // QR分解，求解增量方程 AX=b（X即delta_beta）
        void qr_solve(CvMat *A, CvMat *b, CvMat *X);

        // 两个三维向量的内积（点乘）
        double dot(const double *v1, const double *v2);

        // 两点之间的距离平方
        double dist2(const double *p1, const double *p2);

        // 计算四个控制点任意两点间的距离，总共6个距离
        void compute_rho(double *rho);

        // 根据beta、vi求控制点在相机坐标系下的坐标、3d点在相机坐标系下坐标，恢复相机R、t
        // 返回使用该位姿，得到的重投影误差
        double compute_R_and_t(const double *ut, const double *betas,
                               double R[3][3], double t[3]);

        // ICP求解相机位姿（被compute_R_and_t调用），3d点在 世界坐标系 和 相机坐标系下 的坐标
        void estimate_R_and_t(double R[3][3], double t[3]);

        void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
                          double R_src[3][3], double t_src[3]);

        // 旋转矩阵 转换为 四元数
        void mat_to_quat(const double R[3][3], double q[4]);


        double uc, vc, fu, fv;

        double *pws, *us, *alphas, *pcs;            // 3d点 世界坐标、像素坐标、控制点系数(其实也是坐标)、相机坐标
        int maximum_number_of_correspondences;      // 每次RANSAC计算的过程中使用的匹配点对数的最大值
        int number_of_correspondences;              // 当前迭代中,已经采样的匹配点的个数，默认值为4

        double cws[4][3], ccs[4][3];                // 控制点世界坐标、相机坐标
        double cws_determinant;                     // 控制点世界坐标矩阵的行列式？

        vector<MapPoint *> mvpMapPointMatches;      // 构造EPnP时，给定的地图点

        // 2D Points
        vector<cv::Point2f> mvP2D;
        vector<float> mvSigma2;                     //

        // 3D Points
        vector<cv::Point3f> mvP3Dw;

        // Index in Frame
        vector<size_t> mvKeyPointIndices;

        // Current Estimation
        double mRi[3][3];                   // 当前RANSAC迭代得到的旋转矩阵
        double mti[3];                      // 当前RANSAC迭代得到的平移向量
        cv::Mat mTcwi;                      // 当前RANSAC迭代得到的 变换矩阵T
        vector<bool> mvbInliersi;           // 当前RANSAC迭代中，是否是内点
        int mnInliersi;                     // 当前RANSAC迭代中，内点数目

        // Current Ransac State
        int mnIterations;                   // RANSAC已经迭代的次数
        vector<bool> mvbBestInliers;        // 已经RANSAC迭代中，最优迭代的内点标记
        int mnBestInliers;                  // 已经RANSAC迭代中，最优迭代的内点数目
        cv::Mat mBestTcw;                   // 已经RANSAC迭代中，最优迭代的 变换矩阵

        // Refined
        cv::Mat mRefinedTcw;                // 优化后的位姿变换矩阵
        vector<bool> mvbRefinedInliers;     // 优化后的内点标记
        int mnRefinedInliers;               // 优化后的内点数目

        // Number of Correspondences
        int N;                              // 当前帧与地图点匹配的特征点数目（采样总体）

        // Indices for random selection [0 .. N-1]
        vector<size_t> mvAllIndices;        // 特征点索引（供RANSAC使用）

        // RANSAC probability
        double mRansacProb;                 // 计算RANSAC迭代次数理论值时用到的概率,和Sim3Slover中的一样

        // RANSAC min inliers
        int mRansacMinInliers;              // 正常退出RANSAC时，需要达到的最少内点个数

        // RANSAC max iterations
        int mRansacMaxIts;                  // RANSAC最大迭代次数

        // RANSAC expected inliers/total ratio
        float mRansacEpsilon;               // RANSAC中，最小内点数占全部点个数的比例

        // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
        float mRansacTh;

        // RANSAC Minimum Set used at each iteration
        int mRansacMinSet;                  // 每次RANSAC需要的特征点数，默认为4组3D-2D对应点

        // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
        vector<float> mvMaxError;           // 不同图层上的特征点在进行内点验证的时候,使用的不同的误差阈值

    };

} //namespace ORB_SLAM

#endif // PNPSOLVER_H
