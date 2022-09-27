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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();  // 左目相机中心
    cv::Mat GetStereoCenter();  // 双目相机中心
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    // 计算mBowVec，并且将描述子分散在叶子层上，即mFeatVec记录了属于第i个node的ni个描述子
    void ComputeBoW();

    // Co-visibility graph functions
    void AddConnection(KeyFrame* pKF, const int &weight);   // 为当前帧与pKF添加连接，权重为共同观测到的3d点数量
    void EraseConnection(KeyFrame* pKF);                    // 删除当前帧与pKF的连接（共视关系）
    void UpdateConnections();                               // 更新连接（共视）关系
    void UpdateBestCovisibles();                            // 按照权重，对连接的关键帧进行排序
    std::set<KeyFrame *> GetConnectedKeyFrames();           // 获取当前帧的共视关键帧（无序的集合）
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();  // 获取当前帧的共视关键帧（已按权值排序）
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);  // 获取当前帧的前N个共视关键帧
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);         // 获取当前帧 >w个共视点的 共视帧
    int GetWeight(KeyFrame* pKF);                                       // 获取当前帧与pKF的权重（共视点数量）

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);           // pKF作为当前帧的子帧添加
    void EraseChild(KeyFrame* pKF);         // 删除当前帧的pKF子帧
    void ChangeParent(KeyFrame* pKF);       // 将当前帧的父帧替换为pKF
    std::set<KeyFrame*> GetChilds();        // 获取当前帧的所有子帧（无序的集合）
    KeyFrame* GetParent();                  // 获取当前帧的父帧
    bool hasChild(KeyFrame* pKF);           // 判断当前帧是否有pKF子帧

    // Loop Edges，回环边
    void AddLoopEdge(KeyFrame* pKF);        // 为当前帧和pKF帧添加回环边（两帧之间形成闭环关系）
    std::set<KeyFrame*> GetLoopEdges();     // 获取当前帧的回环帧集合（与当前帧形成闭环关系的所有关键帧）

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);             // 为当前关键帧添加地图点，及其在该帧上的特征点索引
    void EraseMapPointMatch(const size_t &idx);                     // 删除该帧中特征点idx对应的地图点
    void EraseMapPointMatch(MapPoint* pMP);                         // 删除该帧观测到的地图点pMP
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);    // 该帧中idx特征点对应的地图点替换为pMP
    std::set<MapPoint*> GetMapPoints();                             // 获取该关键帧的所有地图点集合
    std::vector<MapPoint*> GetMapPointMatches();                    // 获取该关键帧的所有地图点vector(有序？)
    int TrackedMapPoints(const int &minObs);                        // 该帧的地图点中，观测数>=minObs的地图点数目
    MapPoint* GetMapPoint(const size_t &idx);                       // 获取该帧idx特征点对应的地图点

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);     // 将该帧中第i个特征点反投影到世界坐标系（如果有双目/深度信息）

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();     // 设置当前帧在优化时不可被删除
    void SetErase();        // 当前帧可以（即将）被删除，不进入回环检测

    // Set/check bad flag
    void SetBadFlag();      // 将当前关键帧设置为无效，进行删除关键帧操作
    bool isBad();           // 当前关键帧是否无效（已被删除）

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);     // 评估当前关键帧的场景深度，q=2表示中值

    static bool weightComp( int a, int b){          // 比较两个权重大小，a是否大于b
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){    // 比较两个关键帧的id大小，pKF1_id是否小于pKF2_id
        return pKF1->mnId < pKF2->mnId;
    }

    // The following variables are accessed from only 1 thread or never change (no mutex needed).
    // 单线程参数，无需线程锁
public:

    static long unsigned int nNextId;               // 下一关键帧的id
    long unsigned int mnId;                         // 当前关键帧id
    const long unsigned int mnFrameId;              // 转化为该关键帧的普通帧FrameId

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;                   // 图像每行有几个网格
    const int mnGridRows;                   // 图像每列有几个网格
    const float mfGridElementWidthInv;      // 1/网格宽度
    const float mfGridElementHeightInv;     // 1/网格高度

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;     // 当前帧跟踪到的参考帧id
    long unsigned int mnFuseTargetForKF;            // 与当前帧融合的关键帧id

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;               // TODO...参与局部BA的关键帧索引id，可被优化的关键帧？
    long unsigned int mnBAFixedForKF;               // TODO...只提供信息，不被优化的关键帧id?

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;                  // 该关键帧的回环关键帧id
    int mnLoopWords;                                // 该关键帧与其回环帧共同单词数
    float mLoopScore;                               // 该关键帧与其回环帧词袋匹配得分
    long unsigned int mnRelocQuery;                 // 该关键帧的重定位关键帧id
    int mnRelocWords;                               // 该关键帧与其重定位帧共同单词数
    float mRelocScore;                              // 该关键帧与其重定位帧词袋匹配得分

    // Variables used by loop closing
    cv::Mat mTcwGBA;                                // 该关键帧经全局优化后的位姿
    cv::Mat mTcwBefGBA;                             // 该关键帧全局优化前的位姿
    long unsigned int mnBAGlobalForKF;              // TODO...触发 该关键帧参与的 全局BA的 关键帧id?

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight;      // negative value for monocular points
    const std::vector<float> mvDepth;       // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;                   // std::map<WordId, WordValue>
    DBoW2::FeatureVector mFeatVec;              // std::map<NodeId, std::vector<unsigned int> >

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;                               // 父帧到当前帧的位姿变换矩阵

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;     // 尺度因子的平方
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed through a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;     // 左目光心世界坐标
    cv::Mat Cw;     // Stereo middle point. Only for visualization

    // MapPoints associated to key-points
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;     // 当前帧所属的关键帧数据库
    ORBVocabulary* mpORBvocabulary;     // 当前帧的词袋

    // Grid over the image to speed up feature matching
    // 该关键帧某网格中的特征点，mGrid[i][j]表示第i列第j行的网格
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<KeyFrame*,int> mConnectedKeyFrameWeights;      // 与该关键帧连接（至少15个共视地图点）的关键帧与权重map
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;    // 按照共视关键帧中的权重从大到小排序后的 共视关键帧
    std::vector<int> mvOrderedWeights;                      // 共视关键帧中从大到小排序后的权重

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;                 // 是否是共视关键帧中的第一帧（是否是第一次生成树）
    KeyFrame* mpParent;                     // 该关键帧的父帧
    std::set<KeyFrame*> mspChildrens;       // 该关键帧的子帧集合
    std::set<KeyFrame*> mspLoopEdges;       // 该关键帧的回环边集合

    // Bad flags
    bool mbNotErase;        // 当前关键帧已和其他关键帧形成了回环关系，因此在各种优化的过程中不应该被删除
    bool mbToBeErased;      // 该关键帧即将被删除
    bool mbBad;             // 该关键帧已被删除

    float mHalfBaseline;    // Only for visualization，双目相机基线长度的一半

    Map* mpMap;             // TODO:该关键帧的地图点所在的地图？

    std::mutex mMutexPose;          // 与位姿有关的互斥锁
    std::mutex mMutexConnections;   // 与共视帧有关的互斥锁
    std::mutex mMutexFeatures;      // 与地图点（特征点）有关的互斥锁
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
