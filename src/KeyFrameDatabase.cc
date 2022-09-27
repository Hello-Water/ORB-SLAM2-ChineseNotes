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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

// 构造函数，根据字典大小，给存储倒排索引的数据库重新指定大小
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

// 添加新的关键帧pKF、根据关键帧的词袋向量，更新数据库的倒排索引
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    // 将该关键帧词袋向量里每一个单词添加到数据库
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);  // vit->first = word_id
}

// 从关键帧数据库中删除关键帧pKF，根据关键帧的词袋向量，更新数据库的倒排索引
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry，遍历pKF中的word,找到含有该word的关键帧列表，从中删除pKF
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word，包含该word的所有关键帧lKFs列表
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        // 遍历含有该word的关键帧列表lKFs
        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)   // 从关键帧列表lKFs中删除该关键帧pKF
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

// 清空关键帧倒排索引数据库，重新指定为训练好的字典大小
void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

/**
 * 在关键帧数据库中，找到与该关键帧pkF 可能闭环的候选闭环关键帧（注意是在pKF帧的非共视关键帧中搜索）
 * step1 找出和当前帧有公共单词的所有关键帧lKFsSharingWords，不包括当前帧连接的共视关键帧
 * step2 找出共同单词较多的（最大数目的80%以上）关键帧进行相似度计算得分，得到lScoreAndMatch
 * step3 计算共视关键帧组的累加得分，得到<累加得分，每组得分最高关键帧>对，lAccScoreAndMatch
 * step4 选择大于最高组累加得分75%以上的组中的最优帧，作为最终的候选闭环帧
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    // 取出与当前关键帧pKF相连的所有关键帧（>15个共视地图点），这些相连关键帧都是局部相连，在闭环检测的时候将被剔除
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;       // 用于存储可能闭环的候选关键帧（有相同的word,且不属于局部相连共视帧）

    // Search all keyframes that share a word with current keyframes，Discard keyframes connected to the query keyframe
    // 找出和当前帧具有公共单词的所有关键帧，不包括与当前帧连接（也就是共视）的关键帧
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历当前帧pKF的单词word（word_id = vit->first）
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 数据库中，包含某word的所有关键帧，遍历关键帧列表
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi= *lit;
                if(pKFi->mnLoopQuery != pKF->mnId)              // pKFi未标记为pKF的闭环候选帧
                {
                    pKFi->mnLoopWords=0;                        // pKFi帧中有多少个共有（回环）单词
                    if(!spConnectedKeyFrames.count(pKFi))     // pKFi不在pKF的共视关键帧中
                    {
                        pKFi->mnLoopQuery = pKF->mnId;          // 有相同单词，标记pKFi为pKF的候选闭环帧
                        lKFsSharingWords.push_back(pKFi);     // 添加到候选闭环帧列表
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }   // 互斥锁区域

    // 候选闭环帧为空，返回空向量
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 遍历所有候选闭环帧，找到最大共有单词数，据此确定共有单词数最小阈值
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords > maxCommonWords)
            maxCommonWords = (*lit)->mnLoopWords;
    }
    int minCommonWords = maxCommonWords*0.8f;

    // 遍历所有候选闭环帧，对大于单词数最小阈值的帧 计算评分
    int nscores=0;                                      // 满足单词数阈值的帧数
    list< pair<float,KeyFrame*> > lScoreAndMatch;       // 存储候选闭环帧的<score,pKFi>评分匹配对
    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnLoopWords > minCommonWords)      // 该帧pKFi应大于共有单词数最小阈值
        {
            nscores++;

            // 计算pKFi与pKF字典向量的相似度评分
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
            pKFi->mLoopScore = si;
            if(si>=minScore)    // 评分满足阈值要求，放入匹配对
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    // 候选闭环帧的评分匹配对为空，返回空向量
    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    // 单单计算当前帧pKF和某一候选帧的相似性是不够的，这里将候选帧的前十个共视关键帧归为一组，计算累计得分
    // 遍历lScoreAndMatch，计算上述候选帧的共视关键帧组的总得分，得到最高得分bestAccScore，并以此决定阈值minScoreToRetain
    list<pair<float,KeyFrame*> > lAccScoreAndMatch;     // 累加评分匹配对<accScore, KeyFrame*>
    float bestAccScore = minScore;                      // 累加评分阈值
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);    // 候选帧的10个共视帧

        float accScore = it->first;
        float bestScore = it->first;    // 记录每组共视帧中得分最高的关键帧
        KeyFrame* pBestKF = pKFi;
        // 遍历10个共视帧，累加得分
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 共视帧必须在候选闭环帧列表中，且共有单词数量满足阈值要求
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore += pKF2->mLoopScore;
                if(pKF2->mLoopScore > bestScore)    // 更新每组共视帧中得分最高的关键帧
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        // 把<累加得分,每组中得分最高的关键帧>列为一对
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));

        // 更新最高累加得分
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 确定累加得分阈值
    float minScoreToRetain = 0.75f*bestAccScore;

    // 遍历上述累加得分匹配对，筛选出满足累加得分阈值的关键帧
    set<KeyFrame*> spAlreadyAddedKF;        // 已加入最终候选闭环帧的关键帧集合，防止重复添加候选闭环帧
    vector<KeyFrame*> vpLoopCandidates;     // 最终的候选闭环帧
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first > minScoreToRetain)    // 累加得分满足阈值要求
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))     // 防止重复添加
            {
                vpLoopCandidates.push_back(pKFi);   // 将pKFi添加至最终的候选闭环帧vector中
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}

// 在关键帧数据库中，找到与该F帧相似的 候选重定位关键帧组（F为需要重定位的帧）
// 流程与候选闭环关键帧的确定基本一致，只是搜索范围不同， 重定位可以在当前帧F的共视帧中进行搜索
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
