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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include<mutex>


namespace ORB_SLAM2
{

class KeyFrame;
class Frame;

// 关键帧数据库，用于回环检测、重定位
class KeyFrameDatabase
{
public:
    KeyFrameDatabase(const ORBVocabulary &voc);

   void add(KeyFrame* pKF);     // 添加关键帧、更新数据库的倒排索引

   void erase(KeyFrame* pKF);   // 删除关键帧、更新数据库的倒排索引

   void clear();                // 清空关键帧数据库

   // Loop Detection，在数据库中找到与pKF帧可能闭环的关键帧，minScore相似分数阈值
   std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

   // Re-localization，在数据库中找到与F帧相似的关键帧
   std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

protected:
  // Associated vocabulary，预先训练好的字典
  const ORBVocabulary* mpVoc;

  // Inverted file，倒排索引，mvInvertedFile[i]表示：包含了第i个word id的所有关键帧
  std::vector<list<KeyFrame*> > mvInvertedFile;

  // Mutex，互斥锁
  std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif
