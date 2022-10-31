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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor. 拷贝构造函数 mLastFrame = Frame(mCurrentFrame)
    // 系统自动生成的拷贝函数对于所有涉及分配内存的操作都将是浅拷贝,自定义拷贝构造实现深拷贝
    Frame(const Frame &frame);

    // Constructor for stereo cameras. 双目相机帧的构造函数
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
          ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc,
          cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras. RGB-D相机帧的构造函数
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
          ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
          const float &bf, const float &thDepth);

    // Constructor for Monocular cameras. 单目相机帧的构造函数
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,
          ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    // 提取图像的ORB特征，提取的关键点存放在mvKeys，描述子存放在mDescriptors
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation. 计算词袋
    void ComputeBoW();

    // Set the camera pose. 设置相机位姿mTcw = Tcw
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    // 根据相机位姿,计算相机的旋转,平移和 相机中心
    void UpdatePoseMatrices();

    // Returns the camera center. 返回当前帧位姿的相机中心
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation. 获取旋转矩阵的逆 Rwc = {Rcw}^{-1}
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // 判断地图点是否在当前帧的视野内
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r,
                                     const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    // 计算双目图像之间的特征点匹配关系，若存在匹配，计算深度,并且存储对应的右目特征点坐标
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depth map.
    // 对于RGB-D输入,如果某个特征点的深度值有效,那么计算出假想的"右目图像中对应特征点的坐标"
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Back-projects a keypoint (if stereo/depth info available) into 3D world coordinates.
    // 当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for re-localization. 用于重定位的ORB字典
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;      // 去畸变参数

    // Stereo baseline multiplied by fx.
    float mbf;              // baseline * fx

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;         // 近/远点深度阈值

    // Number of KeyPoints.
    int N;

    // Vector of key-points (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;      // 原始特征点
    std::vector<cv::KeyPoint> mvKeysUn;                 // 矫正后的特征点

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" key-points have a negative value. 单目中，两个容器中的值为-1
    std::vector<float> mvuRight;                        // 对于双目，存储左目特征点对应的右目特征点的横坐标（纵坐标相同）
    std::vector<float> mvDepth;                         // 特征点对应的深度

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;                           // map<WordId, WordValue>
    DBoW2::FeatureVector mFeatVec;                      // map<NodeId, vector<unsigned int>>

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;            // 每一行是一个特征点的描述子

    // MapPoints associated to key-points, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;                // 该帧中每个特征点对应的地图点，没有对应地图点则为空指针

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;                       // 该帧中的外点标记（观测不到map中的3D点）

    // Key-points are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 对图像分区域能够降低重投影地图点时的匹配复杂度，坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe. 参考关键帧
    KeyFrame* mpReferenceKF;

    // Scale pyramid info. 图像金字塔的相关信息
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;        // 每层图像相对于初始图像缩放因子的平方，高斯模糊时用的方差
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once). 无畸变的图像边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    // 标记是否进行了初始化计算，TODO：如果这个标志被置位，说明再下一帧的帧构造函数中要进行这个“特殊的初始化操作”？
    static bool mbInitialComputations;


private:

    // Un-distort key-points given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor). 用于RGB-D的去畸变
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    // 计算去畸变后的图像边界
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign key-points to the grid for speed up feature matching (called in the constructor).
    // 将提取到的特征点分配到图像网格中，加速匹配
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; // mOW == mtwc，相机到世界系的平移量
};

}// namespace ORB_SLAM

#endif // FRAME_H
