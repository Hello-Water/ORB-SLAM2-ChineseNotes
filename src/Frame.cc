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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;         // 下一帧的id
bool Frame::mbInitialComputations=true;     // 是否进行初始化，下一帧是第一帧，置为true，进行初始化
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame(){}

// Copy Constructor，拷贝构造函数，调用该函数时，隐藏的this指针指向目标帧
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
    mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    // vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS]
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// 双目相机帧的构造函数
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
             ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
             :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight),
             mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
             mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info，金字塔的尺度信息
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();      // 左目图像特征点个数

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();   // 对特征点去畸变

    ComputeStereoMatches(); // 双目匹配，匹配成功的点计算深度

    // 初始化本帧的地图点vector,地图点是否为外点的vector
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    // 该过程一般是在第一帧或者是相机标定参数发生变化之后进行的操作，计算去畸变后的图像边界
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);     // 计算去畸变后的图像边界

        // 每个像素占据多少个网络列、行
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;    // 初始化后，标志位置为false
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();             // 将特征点分配到图像网格中
}

// RGB-D相机帧的构造函数
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
             ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf, const float &thDepth)
             :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
             mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // 获取图像的深度，并且根据这个深度推算其右图中匹配的特征点的视差
    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;    // 假想的基线长度，后边计算假想的右目图像上的特征点

    AssignFeaturesToGrid();
}

// 单目帧的构造函数
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
             :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
             mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information，有关双目信息置为-1
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// 将提取的ORB特征点分配到图像网格中
void Frame::AssignFeaturesToGrid()
{
    // TODO: 为什么乘以0.5?
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;   // 特征点所在网格的坐标
        if(PosInGrid(kp,nGridPosX,nGridPosY))       // 网格在图像内
            mGrid[nGridPosX][nGridPosY].push_back(i);    // 将特征点索引放入相应的网格
    }
}

// 提取图像的ORB特征点，提取的关键点存放在mvKeys，描述子存放在mDescriptors
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)     // flag=0是左目图像，flag=1是右目图像
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

// 设置当前帧相机位姿，更新位姿矩阵
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

// 根据 Tcw 计算 mRcw、mtcw 和 mRwc、mOw
void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

// 判断地图点pMP是否在该帧视野内（地图点是否可以在该帧上进行投影）
// viewingCosLimit, 当前相机指向地图点向量和地图点的平均观测方向夹角余弦阈值
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;         // 地图点pMP是否可以在该帧上进行重投影，默认为false

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P + mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // 将地图点投影到当前帧的像素归一化坐标
    const float invz = 1.0f/PcZ;
    const float u = fx*PcX*invz + cx;
    const float v = fy*PcY*invz + cy;

    // 检查地图点的像素坐标是否在去畸变后的图像内
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the
    // 检查地图点到相机中心的距离，是否在有效距离范围内
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;                   // 当前帧对地图点pMP的观测向量
    const float dist = cv::norm(PO);
    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle，检查观测角
    cv::Mat Pn = pMP->GetNormal();              // 地图点的平均观测方向
    const float viewCos = PO.dot(Pn)/dist;   // P0与Pn之间的夹角余弦值
    if(viewCos<viewingCosLimit)                 // 角度过大
        return false;

    // Predict scale in the image
    // 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;                  // 通过上述检查，地图点pMP可以在该帧上进行投影
    pMP->mTrackProjX = u;                       // 地图点的左目像素横坐标
    pMP->mTrackProjXR = u - mbf*invz;           // 地图点的右目像素横坐标
    pMP->mTrackProjY = v;                       // 地图点的左目像素纵坐标
    pMP->mnTrackScaleLevel= nPredictedLevel;    // 根据地图点到光心距离，预测该地图点对应的特征点所在金字塔的尺度层级
    pMP->mTrackViewCos = viewCos;               // 该帧的观测向量与平均观测向量的夹角余弦值

    return true;
}

// 找到在 以x,y为中心,半径为r的圆形内 且 金字塔层级在[minLevel, maxLevel]的特征点
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r,
                                        const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;    // 存储搜索结果
    vIndices.reserve(N);

    // 圆最左侧边界所在网格列坐标
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    // 圆最右侧边界所在网格列坐标
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    // 圆最上侧边界所在网格行坐标
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    // 圆最下侧边界所在网格行坐标
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    // 初步检查金字塔层数范围是否符合要求，(minLevel>=0) || (maxLevel>0) || （maxLevel>minLevel）
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // 遍历外切正方形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到vIndices
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];         // 该网格里的所有特征点
            if(vCell.empty())
                continue;

            // 遍历该网格中的所有特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];  // 获取去畸变后的特征点
                if(bCheckLevels)                                // 检查该特征点所在金字塔层是否在范围内
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 检查该特征点到圆心之间的距离，若在圆内，添加至vIndices输出
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// 关键点kp的位置是否在网格中，计算所在网格坐标
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

// 计算当前帧特征点（描述子）对应的词袋Bow，mBowVec 和 mFeatVec
void Frame::ComputeBoW()
{
    if(mBowVec.empty())     // 是否已经计算过
    {
        // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

// 对特征点进行去畸变操作
void Frame::UndistortKeyPoints()
{
    // 如果第一个畸变参数为0，不需要矫正。（k1,k2,p1,p2,k3）
    // 第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points, 构造特征点矩阵
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Un-distort points，去畸变
    mat=mat.reshape(2); // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1); // 调整回一个通道

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0); // 只更新特征点的x,y坐标，其他属性保留
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

// 计算去畸变后的图像边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0;             mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols;     mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0;             mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols;     mat.at<float>(3,1)=imLeft.rows;

        // Un-distort corners，对四个边界点进行矫正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // 校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0), mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0), mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1), mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1), mat.at<float>(3,1));

    }
    else    // 如果畸变参数为0，就直接获得图像边界
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * 双目匹配，为左目中的每个特征点在右目中进行匹配（SAD算法）
 * 1. 行特征点统计. 统计img_right每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断
 * 2. 粗匹配. 根据步骤1的结果，对img_left第i行的orb特征点pi，在img_right的第i行上的orb特征点集中搜索相似orb特征点, 得到qi
 * 3. 精确匹配. 以点qi为中心，半径为r的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
 * 4. 亚像素精度优化. 步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素插值(抛物线插值)获取float精度的真实视差
 * 5. 最优视差值/深度选择. 通过胜者为王算法（WTA）获取最佳匹配点
 * 6. 删除离群点(outliers). 块匹配相似度阈值判断，归一化SAD最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);      // 初始化mvuRight, mvDepth
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW)/2;     // orb特征相似度阈值，(max+min)/2

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;           // 左目，金字塔底层图像的高

    // Assign key-points to row table
    // 二维vector存储每一行的orb特征点的列坐标的索引
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);     // 为每一行预留内存

    const int Nr = mvKeysRight.size();      // 右目图像特征点总数
    // 行特征点统计。考虑用图像金字塔尺度作为偏移，左图中对应右图的一个特征点可能存在于多行，而非唯一的一行
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;                 // 特征点iR的y坐标，及行号
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];    // 在金字塔底层的2个像素偏移，对应到其他金字塔层尺度的变化
        const int maxr = ceil(kpY+r);            // 可能的最大行号
        const int minr = floor(kpY-r);           // 可能的最小行号

        for(int yi=minr;yi<=maxr;yi++)             // 将特征点iR保存在可能的行号中
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search，对于矫正后的双目图，在列方向(x)存在最大视差maxd和最小视差mind
    // 左图中任何一点p，在右图上的匹配点的范围为应该是[p-maxd, p-mind], 而不需要遍历每一行所有的像素
    const float minZ = mb;
    const float minD = 0;           // 最小视差为0，对应无穷远
    const float maxD = mbf/minZ;    // 最大视差对应的距离是相机的基线

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;       // 保存SAD块匹配相似度距离和左图特征点索引
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)               // 遍历左图每一个特征点iL，在右图搜索最相似的特征点iR
    {
        const cv::KeyPoint &kpL = mvKeys[iL];   // 左图特征点、所在金字塔层、像素坐标
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;             // 左目特征点所在的行
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];    // 左目特征点所在行的 右目特征点索引
        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;             // 在右目上的搜索范围
        const float maxU = uL-minD;
        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;     // 初始化最佳相似度距离
        size_t bestIdxR = 0;                    // 初始化最佳匹配点索引
        const cv::Mat &dL = mDescriptors.row(iL);   // 左目第iL个特征点的描述子

        // Compare descriptor to right key-points
        // 粗配准。左图特征点iL与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的描述子距离和索引
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 左图特征点iL与待匹配点iR的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;     // 待匹配特征点的列坐标(x)
            if(uR>=minU && uR<=maxU)        // uR在搜索范围内
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);                  // iR特征点的描述子
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);      // iL与iR特征点描述子之间的距离

                if(dist<bestDist)           // 更新最佳相似度距离、右目索引
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 图像块滑动窗口用SAD(Sum of absolute differences，差的绝对和)实现精确匹配
        if(bestDist<thOrbDist)      // 最佳相似度距离小于阈值
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;               // 最佳匹配右目特征点x坐标
            const float scaleFactor = mvInvScaleFactors[kpL.octave];    // 左目特征点，金字塔逆尺度
            const float scaleduL = round(kpL.pt.x*scaleFactor);         // 尺度缩放后的左、右目特征点坐标
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search，滑窗搜索
            const int w = 5;        // 滑窗宽度的一半
            // 左目特征点所在的金字塔层, 图像块patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave]
                    .rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            // 图像块均值归一化，降低亮度变化对相似度计算的影响,（图像块每个像素的灰度值-图像块中心的灰度值）
            IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;     // 初始化最佳相似度
            int bestincR = 0;           // 滑窗优化得到的列坐标偏移量
            const int L = 5;            // 滑动窗口的滑动范围为（-L, L）
            vector<float> vDists;       // 图像块相似度
            vDists.resize(2*L+1);

            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸，搜索不能越界
            const float iniu = scaleduR0+L-w;       // 此处应为iniu = scaleduR0-L-w
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 在搜索范围内，在右目特征点所在金字塔层，从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++)
            {
                // 右目特征点所在金字塔层的图像块
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave]
                        .rowRange(scaledvL-w,scaledvL+w+1)
                        .colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                // 计算图像块相似度
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;      // L+incR 为优化后的右目匹配点列坐标(x)
            }

            // 最佳偏移量，一般不在滑窗的边界
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            // 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线来得到最小SAD的亚像素坐标
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 亚像素精度的修正量应该是在[-1,1]之间（1个像素），否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate，根据亚像素精度偏移量deltaR调整最佳匹配的x坐标
            float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR+deltaR);
            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)        // 如果存在负视差，则约束为0.01
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL] = mbf/disparity;    // 计算深度信息
                mvuRight[iL] = bestuR;          // 保存相应的右目x坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    // 删除离群点，块匹配相似度阈值判断
    // 归一化SAD最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    sort(vDistIdx.begin(),vDistIdx.end());          // 按照相似度距离排序（升序）
    const float median = vDistIdx[vDistIdx.size()/2].first;  // 排序后的距离中值
    const float thDist = 1.5f*1.4f*median;                   // 根据距离中值，设置阈值

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)            // 小于阈值的符合要求
            break;
        else                                    // >=阈值，则无匹配
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// 计算RGB-D的立体深度信息（输入深度图），假想的右目特征点横坐标，及特征点深度
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);     // 初始化两个向量为-1
    mvDepth = vector<float>(N,-1);

    // 遍历特征点
    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];     // 去畸变前的特征点
        const cv::KeyPoint &kpU = mvKeysUn[i];  // 去畸变后的特征点

        const float &v = kp.pt.y;               // 去畸变前的特征点坐标
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u); // 未去畸变的特征点所在位置对应的深度

        // 如果获取到的深度点合法(d>0), 那么就保存这个特征点的深度,
        // 并且计算出等效的在假想的右目图中该特征点所匹配的特征点的横坐标
        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;   // 此处用的是去畸变后的横坐标
        }
    }
}

// 当某个特征点的深度信息 或者 双目信息有效时,将它反投影到三维世界坐标系中
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];                 // 特征点i的深度（已去畸变）
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;       // 去畸变后的像素坐标
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;         // 相机坐标
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc + mOw;                   // 相机坐标系转化到世界坐标系
    }
    else    // 深度不合法，返回空矩阵
        return cv::Mat();
}

} //namespace ORB_SLAM
