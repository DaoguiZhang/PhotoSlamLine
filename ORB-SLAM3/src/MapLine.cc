/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapLine.h"
#include "LSDmatcher.h"
//#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM3
{

long unsigned int MapLine::nNextId=0;
mutex MapLine::mGlobalMutex;

//constructors
MapLine::MapLine():
    mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mnLineVisible(1), mnLineFound(1), mbLineBad(false),
    mpLineReplaced(static_cast<MapLine*>(NULL)),
    mbRetrived(false)
{
    mpLineReplaced = static_cast<MapLine*>(NULL);
    mWorldPlucker.setZero();
    // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);//指的是啥？我暂且忘记了？？？
     mnId=nNextId++;
}

//construct a MapLine with 3D Line position
MapLine::MapLine(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos, const Eigen::Vector3f &LsColor, const Eigen::Vector3f &LeColor, 
         KeyFrame* pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnLineVisible(1), mnLineFound(1), mbLineBad(false),
    mpLineReplaced(static_cast<MapLine*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId()),
    mbRetrived(false)
{
    //set line position and color
    SetLineWorldPos(LsPos, LePos);
    SetLineColorRGB(LsColor, LeColor);
    ComputePluckerLineFromWorldLine();

    mLineNormalVector.setZero();

    mbLineTrackInViewR = false;
    mbLineTrackInView = false;

    // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);//指的是啥？我暂且忘记了？？？
    mnId=nNextId++;
}

/*
MapLine::MapLine(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
    mnOriginMapId(pMap->GetId()),
    mbRetrived(false)
{
    mInvDepth=invDepth;
    mInitU=(double)uv_init.x;
    mInitV=(double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector.setZero();

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}*/

MapLine::MapLine(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos,  Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0),  mnLoopLineForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnLineVisible(1),
    mnLineFound(1), mbLineBad(false), mpLineReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId()),
    mbRetrived(false)
{
    //set line position
    SetLineWorldPos(LsPos, LePos);
    //SetLineColorRGB(LsColor, LeColor);
    ComputePluckerLineFromWorldLine();

    Eigen::Vector3f Ow;
    if(pFrame -> NLleft == -1 || idxF < pFrame -> NLleft){
        Ow = pFrame->GetCameraCenter();
    }
    else{
        Eigen::Matrix3f Rwl = pFrame->GetRwc();
        Eigen::Vector3f tlr = pFrame->GetRelativePoseTlr().translation();
        Eigen::Vector3f twl = pFrame->GetOw();

        Ow = Rwl * tlr + twl;
    }
    Eigen::Vector3f LmPos = (mLineWorldPos.head<3>() + mLineWorldPos.tail<3>()) / 2.0f;
    mLineNormalVector = LmPos - Ow;
    mLineNormalVector = mLineNormalVector / mLineNormalVector.norm();

    Eigen::Vector3f PC = LmPos - Ow;
    const float dist = PC.norm();
    const int level = (pFrame -> NLleft == -1) ? pFrame->mvKeyLinesUn[idxF].octave
                                              : (idxF < pFrame -> NLleft) ? pFrame->mvKeyLines[idxF].octave
                                                                         : pFrame -> mvKeyLinesRight[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    //pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);   //to do next...(Line feature descriptor)

    // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId=nNextId++;
}

void MapLine::SetLineWorldPos(const Eigen::Vector3f &LsPos, const Eigen::Vector3f &LePos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    //mWorldPos = Pos;
    mLineWorldPos.head<3>() = LsPos;
    mLineWorldPos.tail<3>() = LePos;
    mLsWorldPos = LsPos;
    mLeWorldPos = LePos;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> MapLine::GetLineWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return std::make_pair(mLsWorldPos, mLeWorldPos);
}

void MapLine::SetLineColorRGB(const Eigen::Vector3f &LsColor, const Eigen::Vector3f &LeColor) {
    unique_lock<mutex> lock2(mGlobalMutex);
    //mColorRGB = Color;
    mLsColorRGB = LsColor;
    mLeColorRGB = LeColor;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> MapLine::GetLineColorRGB() {
    //return mColorRGB;
    unique_lock<mutex> lock(mMutexPos);
    return std::make_pair(mLsColorRGB, mLeColorRGB);
}

void MapLine::setRetrived(const bool retrived)
{
    unique_lock<mutex> lock(mMutexRetrival);
    this->mbRetrived = retrived;
}

bool MapLine::isRetrived()
{
    unique_lock<mutex> lock(mMutexRetrival);
    return this->mbRetrived;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> MapLine::GetLineNormal() {
    unique_lock<mutex> lock(mMutexPos);
    //return mNormalVector;
    return std::make_pair(mLineNormalVector, mLineNormalVector);
}

Eigen::Vector3f MapLine::GetLineNormalVector()
{
    unique_lock<mutex> lock(mMutexPos);
    return mLineNormalVector;
}


KeyFrame* MapLine::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapLine::AddLineObservation(KeyFrame* pKF, int idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int,int> indexes;
    if(mLineObservations.count(pKF)){
        indexes = mLineObservations[pKF];
    }
    else{
        indexes = tuple<int,int>(-1,-1);
    }
    if(pKF -> NLleft != -1 && idx >= pKF -> NLleft){
        get<1>(indexes) = idx;
        //std::cerr << "MapLine 1->idx: " << idx << std::endl;
    }
    else{
        //std::cerr << "MapLine 0->idx: " << idx <<std::endl;
        get<0>(indexes) = idx;
    }
    mLineObservations[pKF]=indexes;
    if(!pKF->mpCamera2 && pKF->mvuLineRight[idx].first>=0 && pKF->mvuLineRight[idx].second>=0)
        nObs+=2;
    else
        nObs++;
}

void MapLine::EraseLineObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mLineObservations.count(pKF))
        {
            tuple<int,int> indexes = mLineObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
            if(leftIndex != -1){
                if(!pKF->mpCamera2 && pKF->mvuLineRight[leftIndex].first>=0 && pKF->mvuLineRight[leftIndex].second>=0)
                    nObs-=2;
                else
                    nObs--;
            }
            if(rightIndex != -1){
                nObs--;
            }
            mLineObservations.erase(pKF);
            if(mpRefKF==pKF)
                mpRefKF=mLineObservations.begin()->first;
            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }
    if(bBad)
        SetBadFlag();
}


std::map<KeyFrame*, std::tuple<int,int>>  MapLine::GetLineObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mLineObservations;
}

int MapLine::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}


void MapLine::SetBadFlag()
{
    std::map<KeyFrame*, tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbLineBad=true;
        obs = mLineObservations;
        mLineObservations.clear();
    }
    for(std::map<KeyFrame*, tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        int leftIndex = get<0>(mit -> second), rightIndex = get<1>(mit -> second);
        if(leftIndex != -1){
            pKF->EraseMapLineMatch(leftIndex);
        }
        if(rightIndex != -1){
            pKF->EraseMapLineMatch(rightIndex);
        }
    }
    mpMap->EraseMapLine(this);
}

void MapLine::PreSave(std::set<KeyFrame*>& spKF, std::set<MapLine*>& spML)
{
    mLineBackupReplacedId = -1;
    if(mpLineReplaced && spML.find(mpLineReplaced) != spML.end())
        mLineBackupReplacedId = mpLineReplaced->mnId;

    mLineBackupObservationsId1.clear();
    mLineBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for(std::map<KeyFrame*,std::tuple<int,int> >::const_iterator it = mLineObservations.begin(), end = mLineObservations.end(); it != end; ++it)
    {
        KeyFrame* pKFi = it->first;
        if(spKF.find(pKFi) != spKF.end())
        {
            mLineBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mLineBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        }
        else
        {
            EraseLineObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if(spKF.find(mpRefKF) != spKF.end())
    {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

int MapLine::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapLine::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}

void MapLine::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapLine*>& mpMLid)
{
    mpRefKF = mpKFid[mBackupRefKFId];
    if(!mpRefKF)
    {
        cout << "ERROR: ML without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
    }
    mpLineReplaced = static_cast<MapLine*>(NULL);
    if(mLineBackupReplacedId>=0)
    {
        map<long unsigned int, MapLine*>::iterator it = mpMLid.find(mLineBackupReplacedId);
        if (it != mpMLid.end())
            mpLineReplaced = it->second;
    }
    mLineObservations.clear();
    for(map<long unsigned int, int>::const_iterator it = mLineBackupObservationsId1.begin(), end = mLineBackupObservationsId1.end(); it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 = mLineBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int,int>(it->second,it2->second);
        if(pKFi)
        {
           mLineObservations[pKFi] = indexes;
        }
    }

    mLineBackupObservationsId1.clear();
    mLineBackupObservationsId2.clear();
}

bool MapLine::isBad()
{
    unique_lock<mutex> lock1(mMutexFeatures,std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos,std::defer_lock);
    lock(lock1, lock2);

    return mbLineBad;
}


tuple<int,int> MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mLineObservations.count(pKF))
        return mLineObservations[pKF];
    else
        return tuple<int,int>(-1,-1);
}

void MapLine::Replace(MapLine* pML)
{
    if(pML->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,tuple<int,int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mLineObservations;
        mLineObservations.clear();
        mbLineBad=true;
        nvisible = mnLineVisible;
        nfound = mnLineFound;
        mpLineReplaced = pML;
    }

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;
        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(!pML->IsInKeyFrame(pKF))
        {
            if(leftIndex != -1){
                pKF->ReplaceMapLineMatch(leftIndex, pML);
                pML->AddLineObservation(pKF,leftIndex);
            }
            if(rightIndex != -1){
                pKF->ReplaceMapLineMatch(rightIndex, pML);
                pML->AddLineObservation(pKF,rightIndex);
            }
        }
        else
        {
            if(leftIndex != -1){
                pKF->EraseMapLineMatch(leftIndex);
            }
            if(rightIndex != -1){
                pKF->EraseMapLineMatch(rightIndex);
            }
        }
    }
    pML->IncreaseFound(nfound);
    pML->IncreaseVisible(nvisible);
    pML->ComputeDistinctiveDescriptors();
    pML->ComputePluckerLineFromWorldLine();
    if (mpMap)
    {
        mpMap->EraseMapLine(this);
    }
    
}

bool MapLine::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mLineObservations.count(pKF));
}

void MapLine::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnLineVisible+=n;
}

void MapLine::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnLineFound+=n;
}

void MapLine::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;

    map<KeyFrame*,tuple<int,int>> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbLineBad)
            return;
        observations=mLineObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad()){
            tuple<int,int> indexes = mit -> second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if(leftIndex != -1){
                vDescriptors.push_back(pKF->mLineDescriptors.row(leftIndex));
            }
            if(rightIndex != -1){
                vDescriptors.push_back(pKF->mLineDescriptors.row(rightIndex));
            }
        }
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            //int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            int distij = LSDmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);   //to do next...(Line feature descriptor)
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mLineDescriptor = vDescriptors[BestIdx].clone();
    }
}

void MapLine::ComputePluckerLineFromWorldLine()
{
    Eigen::Vector3d ls = mLineWorldPos.head<3>().cast<double>();
    Eigen::Vector3d le = mLineWorldPos.tail<3>().cast<double>();
    if((ls-le).norm() < 1e-8)
    {
        mWorldPlucker.setZero();
        return;
    }
    Eigen::Vector3d plu_n, plu_v;
    Converter::LineSegmentToPlucker(ls, le, plu_n, plu_v);
    mWorldPlucker.head<3>() = plu_n;
    mWorldPlucker.tail<3>() = plu_v;
}

Map* MapLine::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapLine::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

cv::Mat MapLine::GetLineDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mLineDescriptor.clone();
}

MapLine* MapLine::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpLineReplaced;
}

void MapLine::SetLineNormalVector(const Eigen::Vector3f& normal)
{
    unique_lock<mutex> lock3(mMutexPos);
    mLineNormalVector = normal;
}

void MapLine::PrintObservations()
{
    std::cout << "ML_OBS: ML " << mnId << std::endl;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=mLineObservations.begin(), mend=mLineObservations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKFi = mit->first;
        tuple<int,int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        std::cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << std::endl;
    }
}

//to check the bug->done
void MapLine::UpdateNormalAndDepth()
{
    map<KeyFrame*,tuple<int,int> > observations;
    KeyFrame* pRefKF;
    Eigen::Matrix<float,6,1> LineEndPos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbLineBad)
            return;
        observations = mLineObservations;
        pRefKF = mpRefKF;
        LineEndPos = mLineWorldPos;
    }
    //to do next...(Line feature descriptor)
    if(observations.empty())
        return;
    Eigen::Vector3f normal;
    normal.setZero();
    int n=0;
    Eigen::Vector3f Pos = (LineEndPos.head<3>() + LineEndPos.tail<3>()) / 2.0f;
    for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        tuple<int,int> indexes = mit -> second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if(leftIndex != -1){
            Eigen::Vector3f Owi = pKF->GetCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
        if(rightIndex != -1){
            Eigen::Vector3f Owi = pKF->GetRightCameraCenter();
            Eigen::Vector3f normali = Pos - Owi;
            normal = normal + normali / normali.norm();
            n++;
        }
    }
    Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
    const float dist = PC.norm();
    tuple<int ,int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    //std::cerr << "leftIndex: " << leftIndex << std::endl;
    int level;
    if(pRefKF -> NLleft == -1){
        level = pRefKF->mvKeyLinesUn[leftIndex].octave;
    }
    else if(leftIndex != -1){
        level = pRefKF -> mvKeyLines[leftIndex].octave;
    }
    else{
        level = pRefKF -> mvKeyLinesRight[rightIndex - pRefKF -> NLleft].octave;
    }
    //std::cerr << "MapLine: Level: " << level << std::endl;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;
    //std::cerr << "MapLine: nLevels: " << nLevels << std::endl;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mLineNormalVector = normal/n;
    }
}


float MapLine::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapLine::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

float MapLine::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnLineFound)/mnLineVisible;
}

Eigen::Vector3f MapLine::GetProjectedLineABC(KeyFrame *pKF)
{
    // 1) 获取当前直线 Plücker 参数（世界系）
    Eigen::Matrix<double,6,1> Lw = this->GetPluckerLine();

    // 2) 世界 → 相机变换
    Sophus::SE3f Tcw = pKF->GetPose();
    Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    Eigen::Vector3f tcw = Tcw.translation();

    // 3) Plücker 直线世界系 → 相机系
    Eigen::Matrix<float,6,1> Lc;
    {
        Eigen::Vector3f n = Lw.segment<3>(0).cast<float>();
        Eigen::Vector3f v = Lw.segment<3>(3).cast<float>();

        Eigen::Vector3f n_c = Rcw * n + tcw.cross(Rcw * v);
        Eigen::Vector3f v_c = Rcw * v;

        Lc.head<3>() = n_c;
        Lc.tail<3>() = v_c;
    }

    // 4) 投影成 2D 图像线
    //    n_c = [n1, n2, n3], v_c = [v1, v2, v3]
    //    图像线 l = P * Lc
    //    对 pinhole，相机矩阵：
    //    K = [fx 0 cx; 0 fy cy; 0 0 1]

    float fx = pKF->fx;
    float fy = pKF->fy;
    float cx = pKF->cx;
    float cy = pKF->cy;

    Eigen::Vector3f n = Lc.head<3>();
    Eigen::Vector3f v = Lc.tail<3>();

    // 相机平面直线（参考 ORB-SLAM2/3 投影推导）
    Eigen::Vector3f l;
    l[0] = n[0] * fx + n[2] * (cx * fx);     // a
    l[1] = n[1] * fy + n[2] * (cy * fy);     // b
    l[2] = n[2];                              // c  (保持比例即可)

    // 5) 归一化，使 (a^2 + b^2 = 1)
    float norm_ab = std::sqrt(l[0]*l[0] + l[1]*l[1]);
    if(norm_ab > 1e-6)
        l /= norm_ab;

    return l;
}


// 将像素点 (u,v) 在该 keyframe 下反投影成相机坐标系的单位方向向量（未缩放，单位向量）
// 返回：rayDir_world (单位向量), camCenter_world (相机中心在世界系)
void  MapLine::BackprojectPixelToWorldRay(KeyFrame* pKF, const cv::Point2f &uv,
                                       Eigen::Vector3f &camCenter_world,
                                       Eigen::Vector3f &rayDir_world)
{
    // 获取相机内参（假定保存在 KeyFrame）
    float fx = pKF->fx;
    float fy = pKF->fy;
    float cx = pKF->cx;
    float cy = pKF->cy;

    // 相机坐标下的方向（未经畸变校正；若你在 KF 有畸变校正函数，请在这里先做去畸变）
    Eigen::Vector3f p_cam;
    p_cam << (uv.x - cx) / fx, (uv.y - cy) / fy, 1.0f;
    p_cam.normalize(); // 单位方向

    // 相机到世界
    Sophus::SE3f Tcw = pKF->GetPose();
    Sophus::SE3f Twc = Tcw.inverse();

    // 相机中心（世界系）
    camCenter_world = Twc.translation().cast<float>();

    // 将相机坐标系方向变换到世界系
    Eigen::Matrix3f Rwc = Twc.rotationMatrix().cast<float>();
    rayDir_world = (Rwc * p_cam).normalized();
}

// 将 Plücker 参数 L = [n(3); v(3)] 转换为 直线上一点 p0 (world) 和方向 dir (world, 单位向量)
// p0 = (n × v) / |v|^2
void MapLine::PluckerToPointAndDir(const Eigen::Matrix<double,6,1> &L, Eigen::Vector3f &p0, Eigen::Vector3f &dir)
{
    Eigen::Vector3d n_d = L.head<3>();
    Eigen::Vector3d v_d = L.tail<3>();

    double v_norm2 = v_d.squaredNorm();
    if (v_norm2 < 1e-12) {
        // 避免除零；退化时用 v = (1,0,0)
        v_d = Eigen::Vector3d(1,0,0);
        v_norm2 = 1.0;
    }

    Eigen::Vector3d p0_d = n_d.cross(v_d) / v_norm2;
    Eigen::Vector3d dir_d = v_d.normalized();

    p0 = p0_d.cast<float>();
    dir = dir_d.cast<float>();
}

// 计算两条无穷直线（L1: P1 + s * d1, L2: P2 + t * d2）间的最近点
// 返回：点在 L1 上的参数 s，点在 L2 上的参数 t，以及对应的点 p1 = P1 + s*d1, p2 = P2 + t*d2
// 若平行，会选择投影到其中一条上
void MapLine::ClosestPointsBetweenLines(const Eigen::Vector3f &P1, const Eigen::Vector3f &d1,
                                      const Eigen::Vector3f &P2, const Eigen::Vector3f &d2,
                                      float &s_out, float &t_out,
                                      Eigen::Vector3f &p1_out, Eigen::Vector3f &p2_out)
{
    // 使用双精度进行计算以更稳定
    Eigen::Vector3d P1d = P1.cast<double>();
    Eigen::Vector3d d1d = d1.cast<double>();
    Eigen::Vector3d P2d = P2.cast<double>();
    Eigen::Vector3d d2d = d2.cast<double>();

    double a = d1d.dot(d1d);
    double b = d1d.dot(d2d);
    double c = d2d.dot(d2d);
    Eigen::Vector3d r = P1d - P2d;
    double d = d1d.dot(r);
    double e = d2d.dot(r);
    double denom = a*c - b*b;

    double sc, tc;
    if (std::fabs(denom) < 1e-12) {
        // 平行或近似平行 — 取射线 P1 投影到 L2 的参数 tc，sc 设为 0
        sc = 0.0;
        tc = (b>c ? d/b : e/c); // heuristic
    } else {
        sc = (b*e - c*d) / denom;
        tc = (a*e - b*d) / denom;
    }

    Eigen::Vector3d cp1 = P1d + sc * d1d;
    Eigen::Vector3d cp2 = P2d + tc * d2d;

    s_out = static_cast<float>(sc);
    t_out = static_cast<float>(tc);
    p1_out = cp1.cast<float>();
    p2_out = cp2.cast<float>();
}


// 稳健均值（按每个坐标维度取中位数）
// 输入：points (非空)
// 返回：三维中位数向量
Eigen::Vector3f MapLine::RobustMedian3(const vector<Eigen::Vector3f>& pts)
{
    std::vector<float> xs, ys, zs;
    xs.reserve(pts.size());
    ys.reserve(pts.size());
    zs.reserve(pts.size());

    for(auto& p: pts){
        xs.push_back(p.x());
        ys.push_back(p.y());
        zs.push_back(p.z());
    }

    auto med = [](vector<float>& v)->float{
        sort(v.begin(), v.end());
        return v[v.size()/2];
    };

    Eigen::Vector3f m;
    m << med(xs), med(ys), med(zs);
    return m;
}

// 返回 true 表示成功拿到端点 (s,e)
bool MapLine::GetKeyFrameLineEndpoints(KeyFrame* pKF, int lineIdx, cv::Point2f &s_out, cv::Point2f &e_out)
{
    // --------- 这是占位: KeyFrame API ----------
    // 例如： if(pKF->mvLines.size() > lineIdx) { s_out = pKF->mvLines[lineIdx].start; e_out = pKF->mvLines[lineIdx].end; return true; }
    //if(pKf)
    // 我这里返回 false 以强制你实现
    return false;
}

void MapLine::SetPluckerLineNew(const Eigen::Matrix<double,6,1>& plk)
{
    mWorldPlucker = plk;
    // ensure n·v = 0
    Eigen::Vector3d n = mWorldPlucker.head<3>();
    Eigen::Vector3d v = mWorldPlucker.tail<3>();
    double nv = n.dot(v);
    if(std::fabs(nv) > 1e-9){
        Eigen::Vector3d v_norm = v.normalized();
        n = n - nv * v_norm;
        mWorldPlucker.head<3>() = n;
    }
}

void MapLine::UpdateFromPluckerLine()
{
    // ---- 1) 获取 Plücker 参数（优化后） ----
    Eigen::Matrix<double,6,1> Lw = GetPluckerLine();
    Eigen::Vector3f p0_line_world, dir_world;
    PluckerToPointAndDir(Lw, p0_line_world, dir_world);

    // ---- 2) 遍历所有观测（KeyFrame → line idx） ----
    std::map<ORB_SLAM3::KeyFrame*, std::tuple<int, int> > observations = GetLineObservations();

    vector<Eigen::Vector3f> p1_list, p2_list;
    p1_list.reserve(observations.size());
    p2_list.reserve(observations.size());

    for(auto kv : observations)
    {
        KeyFrame* pKF = kv.first;
        if(!pKF || pKF->isBad()) continue;

        int idx = get<0>(kv.second);  // KeyFrame 中的线 index

        // --- 读取 2D 端点（确保你 KeyFrame 有此接口） ---
        cv::Point2f s2d, e2d;
        if(!pKF->GetLineEndpoints(idx, s2d, e2d))
            continue;

        // --- 反投影两个像素端点 ---
        Eigen::Vector3f o1, d1, o2, d2;
        BackprojectPixelToWorldRay(pKF, s2d, o1, d1);
        BackprojectPixelToWorldRay(pKF, e2d, o2, d2);

        d1.normalize();
        d2.normalize();

        // --- 求射线与 Plücker 直线的最近点 ---
        float sp, tp;
        Eigen::Vector3f pr, pl;

        // 端点1
        ClosestPointsBetweenLines(o1, d1, p0_line_world, dir_world,
                                  sp, tp, pr, pl);
        p1_list.push_back(pl);

        // 端点2
        ClosestPointsBetweenLines(o2, d2, p0_line_world, dir_world,
                                  sp, tp, pr, pl);
        p2_list.push_back(pl);
    }

    // ---- 3) 融合各观察（中位数稳健合并） ----
    if(!p1_list.empty() && !p2_list.empty())
    {
        Eigen::Vector3f P1 = RobustMedian3(p1_list);
        Eigen::Vector3f P2 = RobustMedian3(p2_list);

        // ---- 4) 写回 MapLine ----
        SetLineWorldPos(P1, P2);
    }
}

bool  MapLine::UpdatePluckerFromBackProjectLines()
{
    // ---- 2) 遍历所有观测（KeyFrame → line idx） ----
    std::map<ORB_SLAM3::KeyFrame*, std::tuple<int, int> > observations = GetLineObservations();

    return true;
}

#if 0


cv::Mat MapLine::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}




#endif

} //namespace ORB_SLAM
