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
static const float MIN_DEPTH_V = 0.01f; // 1cm
static const float MAX_DEPTH_V = 100.0f; // 100m

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

float MapLine::GetObservationDepth0(KeyFrame* pKf, int idx)
{
    return (pKf->GetObservatonLineLsDepth(idx));
}

float MapLine::GetObservationDepth1(KeyFrame* pKf, int idx)
{
    return (pKf->GetObservatonLineLeDepth(idx));
}

const std::vector<Eigen::Vector3f>& MapLine::GetLineSampledPoints3D()
{
    // TODO: Implement the function to return the 3D points sampled along the line
    return mSampledPoints3D;
}

const std::vector<Eigen::Vector2f>& MapLine::GetLineSampledPoints2D()
{
    // TODO: Implement the function to return the 2D points sampled along the line in the given KeyFrame
    return mSampledPoints2D;
}

const std::vector<cv::Vec3b>& MapLine::GetLineSampledPntsColors()
{
    // TODO: Implement the function to return the colors of the 3D points sampled along the line
    return mSampledPointsColor;
}

void MapLine::SetObservationLineLsDepth(KeyFrame* pKf, int idx, float dv)
{
    pKf->SetObservationLineLsDepth(idx, dv);
}

void MapLine::SetObservationLineLeDepth(KeyFrame* pKf, int idx, float dv)
{
    pKf->SetObservationLineLeDepth(idx, dv);
}

bool MapLine::HasCachedWorldObservationLineEndPoints()
{
    return false;   //TO do Next
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

void MapLine::UpdateWorldEndpointsFromObservationLineDepth()
{
    // --- Validate whole MapLine with PCA using all back-projected points ---
    std::vector<Eigen::Vector3d> all_pts;
    for(const auto &obs : this->GetLineObservations())
    {
        KeyFrame* pKFi = obs.first; int idx = get<0>(obs.second);
        if(!pKFi || pKFi->isBad()) continue;
        float newd0 = GetObservationDepth0(pKFi, idx);
        float newd1 = GetObservationDepth1(pKFi, idx);
        Eigen::Vector2f sl, el; if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
        // --- Validate whole MapLine with PCA using all back-projected points ---
        if(!(newd0>MIN_DEPTH_V && newd0<MAX_DEPTH_V && newd1>MIN_DEPTH_V && newd1<MAX_DEPTH_V)) continue;
        Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1]));
        Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
        g2o::SE3Quat Tcw = g2o::SE3Quat(pKFi->GetPose().unit_quaternion().cast<double>(), pKFi->GetPose().translation().cast<double>());
        g2o::SE3Quat Twc = Tcw.inverse();
        Eigen::Vector3d pw0 = Twc * (double(newd0) * r0);
        Eigen::Vector3d pw1 = Twc * (double(newd1) * r1);
        all_pts.push_back(pw0); all_pts.push_back(pw1);
    }
    // high confidence: recompute Plucker using more stable world points and commit endpoints
    Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
    if(Lw.allFinite())
    {
        SetPluckerLine(Lw); // commit improved plucker
        // Optionally update endpoints (two extreme projections along line)
        UpdateWorldEndpointsFromObservationPntsAndPluckerLine(Lw, all_pts); //use the all points
    }
    // double ratio = Converter::FirstPCVarianceRatio(all_pts);
    // if(all_pts.size() >= 4 && ratio < LINE_COLINEARITY_LOW)
    // {
    //     // Mark line as bad: too inconsistent
    //     pML->SetBadFlag();
    //     // erase all observations to be safe
    //     for(const auto& obs : pML->GetLineObservations())
    //     {
    //         KeyFrame* pKFi = obs.first; if(!pKFi) continue;
    //         pKFi->EraseMapLineMatch(pML);
    //         pML->EraseLineObservation(pKFi);
    //     }
    // }
    // else if(all_pts.size() >= 2 && ratio > LINE_COLINEARITY_HIGH)
    // {   
    // }
}

void MapLine::UpdateWorldEndpointsFromObservationPntsAndPluckerLine(const Eigen::Matrix<double,6,1>& Lw,const std::vector<Eigen::Vector3d>& pnts_3d,double lower_q,double upper_q)
{
    if (pnts_3d.size() < 2)
        return; // cannot update
    // --- 1. Extract Plücker components ---
    Eigen::Vector3d n = Lw.head<3>();
    Eigen::Vector3d v = Lw.tail<3>();
    const double v2 = v.squaredNorm();
    if (v2 < 1e-10)
        return;
    Eigen::Vector3d v_norm = v.normalized();
    // --- 2. Compute a reference point P0 on the line ---
    //     P0 = (n × v) / ||v||²
    Eigen::Vector3d P0 = n.cross(v) / v2;
    // --- 3. Project all points to line (scalar t values) ---
    std::vector<double> ts;
    ts.reserve(pnts_3d.size());
    for (const auto& p : pnts_3d)
        ts.push_back((p - P0).dot(v_norm));
    if (ts.size() < 2)
        return;
    // --- 4. Robust percentile filtering ---
    std::sort(ts.begin(), ts.end());
    auto get_percentile = [&](double q)
    {
        if (ts.empty()) return 0.0;
        double idx = q * (ts.size() - 1);
        size_t lo = (size_t)std::floor(idx);
        size_t hi = (size_t)std::ceil(idx);
        double t = ts[lo];
        if (hi > lo)
            t += (ts[hi] - ts[lo]) * (idx - lo);
        return t;
    };
    double tmin = get_percentile(lower_q);  // e.g. q = 0.05
    double tmax = get_percentile(upper_q);  // e.g. q = 0.95
    if (tmax < tmin)
        std::swap(tmin, tmax);
    // --- 5. Compute endpoints on line ---
    Eigen::Vector3d p1 = P0 + tmin * v_norm;
    Eigen::Vector3d p2 = P0 + tmax * v_norm;
    // --- 6. Store ---
    mLsWorldPos = p1.cast<float>();
    mLeWorldPos   = p2.cast<float>();
    mLineWorldPos.head<3>() = mLsWorldPos;
    mLineWorldPos.tail<3>() = mLeWorldPos;
}


bool  MapLine::UpdatePluckerFromBackProjectLines()
{
    // ---- 2) 遍历所有观测（KeyFrame → line idx） ----
    //std::map<ORB_SLAM3::KeyFrame*, std::tuple<int, int> > observations = GetLineObservations();
    return true;
}

void MapLine::SamplePointsAlongLinesWorld3D_old()
{
    // Sample points along the 3D line
    Eigen::Vector3f line_start = mLineWorldPos.head<3>();
    Eigen::Vector3f line_end = mLineWorldPos.tail<3>();
    Eigen::Vector3f line_dir = (line_end - line_start).normalized();

    // Sample points along the line
    for (float t = 0; t <= 1; t += 0.1)
    {
        Eigen::Vector3f point = line_start + t * (line_end - line_start);
        mSampledPoints3D.push_back(point);
    }
    // TODO: Project 3D points into image To get point color
    for (const auto& point : mSampledPoints3D)
    {
        const auto& obs = GetLineObservations();
        std::vector<cv::Vec3b> sampledLinePoints2DColor;
        for(auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if(!pKFi || pKFi->isBad()) continue;
            cv::Point2f point_2d;
            if(pKFi->ProjectPointToImage(point, point_2d))
            {
                //mSampledPoints2D.push_back(point_2d);
                cv::Vec3b color = pKFi->GetColor(point_2d);
                sampledLinePoints2DColor.push_back(color);
                // TODO: Store or process the sampled line points 2D colors
                cv::Vec3b avg_color = AverageColorIgnoreBlack(sampledLinePoints2DColor);
                // Store or use avg_color as needed
                mSampledPointsColor.push_back(avg_color);
            }
            
        }
    }
}

void MapLine::SamplePointsAlongLinesWorld3D(float sample_step /* e.g. 0.2f */)
{
    mSampledPoints3D.clear();
    mSampledPointsColor.clear();

    Eigen::Vector3f P0 = mLineWorldPos.head<3>();
    Eigen::Vector3f P1 = mLineWorldPos.tail<3>();

    Eigen::Vector3f d = P1 - P0;
    float length = d.norm();
    if (length < 1e-6f) return;

    Eigen::Vector3f dir = d / length;

    int num_samples = std::max(2, static_cast<int>(std::ceil(length / sample_step)));

    for (int i = 0; i <= num_samples; ++i)
    {
        float s = std::min(length, i * sample_step);
        Eigen::Vector3f Pw = P0 + s * dir;
        // ---- color sampling across keyframes ----
        std::vector<cv::Vec3b> colors;
        const auto& obs = GetLineObservations();

        for (auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (!pKFi || pKFi->isBad()) continue;

            cv::Point2f uv;
            if (!pKFi->ProjectPointToImage(Pw, uv)) continue;

            if (!pKFi->IsInImage(uv.x, uv.y)) continue;

            cv::Vec3b c = pKFi->GetColor(uv);
            if (c != cv::Vec3b(0,0,0))
                colors.push_back(c);
        }

        if (!colors.empty())
        {
            cv::Vec3b avg = AverageColorIgnoreBlack(colors);
            mSampledPointsColor.push_back(avg);
            //mSampledPoints2D.push_back(Eigen::Vector2f(uv.x, uv.y));
            mSampledPoints3D.push_back(Pw);
        }
    }
}

void MapLine::SamplePointsByImageLength(KeyFrame* pKF, float pixel_step)
{
    mSampledPoints3D.clear();
    mSampledPointsColor.clear();

    Eigen::Vector3f P0 = mLineWorldPos.head<3>();
    Eigen::Vector3f P1 = mLineWorldPos.tail<3>();

    cv::Point2f uv0, uv1;
    if (!pKF->ProjectPointToImage(P0, uv0) ||
        !pKF->ProjectPointToImage(P1, uv1))
        return;

    float pixel_len = cv::norm(uv1 - uv0);
    int num_samples = std::max(2, static_cast<int>(pixel_len / pixel_step));

    for (int i = 0; i <= num_samples; ++i)
    {
        float t = static_cast<float>(i) / num_samples;
        Eigen::Vector3f Pw = P0 + t * (P1 - P0);

        cv::Point2f uv;
        if (!pKF->ProjectPointToImage(Pw, uv)) continue;
        if (!pKF->IsInImage(uv.x, uv.y)) continue;

        cv::Vec3b c = pKF->GetColor(uv);
        if (c != cv::Vec3b(0,0,0))
        {
            mSampledPoints3D.push_back(Pw);
            //mSampledPoints2D.push_back(Eigen::Vector2f(uv.x, uv.y));
            mSampledPointsColor.push_back(c);
        }
    }
}

//BGR->RGB（后面验证一下）
void MapLine::SamplePointsAlongLine_MultiViewWeighted(
    float sample_step,
    float view_angle_power)
{
    mSampledPoints3D.clear();
    mSampledPointsColor.clear();

    // ---- 1. 世界线段 ----
    Eigen::Vector3f P0 = mLineWorldPos.head<3>();
    Eigen::Vector3f P1 = mLineWorldPos.tail<3>();

    Eigen::Vector3f d = P1 - P0;
    float length = d.norm();
    if (length < 1e-6f) return;

    Eigen::Vector3f line_dir = d / length;

    int num_samples = std::max(2, static_cast<int>(std::ceil(length / sample_step)));

    const auto& obs = GetLineObservations();

    // ---- 2. 沿线采样 ----
    for (int i = 0; i <= num_samples; ++i)
    {
        float s = std::min(length, i * sample_step);
        Eigen::Vector3f Pw = P0 + s * line_dir;

        // 累积加权颜色
        Eigen::Vector3f color_sum(0, 0, 0);
        float weight_sum = 0.0f;

        // ---- 3. 多视角融合 ----
        for (auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (!pKFi || pKFi->isBad()) continue;

            // 投影到图像
            cv::Point2f uv;
            if (!pKFi->ProjectPointToImage(Pw, uv)) continue;
            if (!pKFi->IsInImage(uv.x, uv.y)) continue;

            // 取颜色
            cv::Vec3b c = pKFi->GetColor(uv);
            if (c == cv::Vec3b(0,0,0)) continue;

            // ---- 4. 计算权重 ----

            // (a) 相机中心
            Eigen::Vector3f Cw = pKFi->GetCameraCenter();

            // (b) 视线方向
            Eigen::Vector3f view_dir = (Cw - Pw);
            float depth = view_dir.norm();
            if (depth < 1e-6f) continue;
            view_dir.normalize();

            // (c) 视角权重（线方向 vs 视线）
            float cos_angle = std::fabs(line_dir.dot(view_dir));
            cos_angle = std::max(0.0f, cos_angle);

            float w_angle = std::pow(cos_angle, view_angle_power);

            // (d) 深度权重（近大远小）
            float w_depth = 1.0f / depth;

            float weight = w_angle * w_depth;
            if (!std::isfinite(weight) || weight < 1e-6f) continue;

            // ---- 5. 累积 ----
            Eigen::Vector3f cf(c[2], c[1], c[0]); // BGR → RGB
            color_sum += weight * cf;
            weight_sum += weight;
        }

        // ---- 6. 写回结果 ----
        if (weight_sum > 1e-6f)
        {
            Eigen::Vector3f color = color_sum / weight_sum;
            color = color.cwiseMax(0.0f).cwiseMin(255.0f);

            mSampledPoints3D.push_back(Pw);
            //mSampledPoints2D.push_back(Eigen::Vector2f(uv.x, uv.y));
            mSampledPointsColor.emplace_back(
                static_cast<uchar>(color(0)),
                static_cast<uchar>(color(1)),
                static_cast<uchar>(color(2))
            );
        }
    }
}


void MapLine::SamplePointsAlongLine_MultiViewWeighted_Advanced(
    float sample_step,
    float view_angle_power,
    float sigma_line_pixel,
    int   top_k)
{
    mSampledPoints3D.clear();
    mSampledPointsColor.clear();

    // -------- 1. 世界线段 --------
    Eigen::Vector3f P0 = mLineWorldPos.head<3>();
    Eigen::Vector3f P1 = mLineWorldPos.tail<3>();

    Eigen::Vector3f d = P1 - P0;
    float length = d.norm();
    if (length < 1e-6f) return;

    Eigen::Vector3f line_dir = d / length;

    int num_samples = std::max(
        2, static_cast<int>(std::ceil(length / sample_step)));

    const auto& obs = GetLineObservations();

    // -------- 2. 沿线采样 --------
    for (int i = 0; i <= num_samples; ++i)
    {
        float s = std::min(length, i * sample_step);
        Eigen::Vector3f Pw = P0 + s * line_dir;

        // 每个观测的 (weight, color)
        struct WeightedColor {
            float w;
            Eigen::Vector3f c;
        };
        std::vector<WeightedColor> candidates;

        int visible_count = 0;

        // -------- 3. 多视角观测 --------
        for (auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (!pKFi || pKFi->isBad()) continue;

            // ---- 🌟 防火墙 B 核心开始：深度一致性校验 🌟 ----
            // 1. 计算 Pw 在当前相机坐标系下的深度 (z-depth)
            // 获取相机位姿 Tcw
            Sophus::SE3f Tcw = pKFi->GetPose();
            Eigen::Vector3f Pc = Tcw * Pw; // 世界点转相机点
            float reproj_depth = Pc.z();

            if (reproj_depth <= 0) continue; // 在相机背面

            int idxLine = std::get<0>(mit.second);
            if (idxLine < 0 || idxLine >= (int)pKFi->mvKeyLines.size())
                continue;

            const cv::line_descriptor::KeyLine& kl =
                pKFi->mvKeyLines[idxLine];

            // ---- 投影 ----
            cv::Point2f uv;
            if (!pKFi->ProjectPointToImage(Pw, uv)) continue;
            if (!pKFi->IsInImage(uv.x, uv.y)) continue;

            cv::Vec3b color_bgr = pKFi->GetColor(uv);
            if (color_bgr == cv::Vec3b(0,0,0)) continue;

            if(pKFi->HasDepth())
            {
                // 3. 获取深度图中的测量深度
                // 假设 KeyFrame 有 GetDepth 接口，返回 float 类型的深度值
                float measured_depth = pKFi->GetDepth(uv.x, uv.y); 

                // 4. 深度合法性过滤 (0 表示无效深度)
                if (measured_depth <= 0) continue; 

                // 5. 校验：如果重投影深度与测量深度差值 > 10cm (可调参数)
                // 这是一个非常强力的防火墙，能直接杀掉“悬空线”
                float depth_diff = std::fabs(reproj_depth - measured_depth);
                // 允许 5% 的深度误差
                float depth_threshold = std::max(0.05f, measured_depth * 0.05f); 
                if (depth_diff > depth_threshold) continue;
            }

            // -------- 权重计算 --------
            // (1) 深度 + 视角
            Eigen::Vector3f Cw = pKFi->GetCameraCenter();
            Eigen::Vector3f view_dir = (Cw - Pw);
            float depth = view_dir.norm();
            if (depth < 1e-6f) continue;
            view_dir.normalize();

            float cos_angle = std::fabs(line_dir.dot(view_dir));
            cos_angle = std::max(0.0f, cos_angle);
            float w_angle = std::pow(cos_angle, view_angle_power);
            float w_depth = 1.0f / depth;

            // (2) 图像线一致性（点到 2D 线距离）
            float a, b, c;
            if (!ComputeLineABCFromKeyLine(kl, a, b, c))
                continue;   // 退化线，跳过
            float norm = std::sqrt(a*a + b*b);
            if (norm < 1e-6f) continue;

            float dist_line =
                std::fabs(a * uv.x + b * uv.y + c) / norm;

            float w_line =
                std::exp(-(dist_line * dist_line) /
                         (sigma_line_pixel * sigma_line_pixel));

            float weight = w_angle * w_depth * w_line;
            if (!std::isfinite(weight) || weight < 1e-6f) continue;

            Eigen::Vector3f color_rgb(
                color_bgr[2], color_bgr[1], color_bgr[0]);

            candidates.push_back({weight, color_rgb});

            visible_count++;
        }

        // 如果这个 3D 采样点在少于 2 个视角中被“证实”，则不生成高斯
        //if (visible_count < 2) continue;    //To test next...

        if (candidates.empty()) continue;

        // -------- 4. Top-K 视角选择 --------
        if ((int)candidates.size() > top_k)
        {
            std::nth_element(
                candidates.begin(),
                candidates.begin() + top_k,
                candidates.end(),
                [](const WeightedColor& a, const WeightedColor& b) {
                    return a.w > b.w;
                });
            candidates.resize(top_k);
        }

        // -------- 5. 加权融合 --------
        Eigen::Vector3f color_sum(0,0,0);
        float weight_sum = 0.0f;

        for (const auto& wc : candidates)
        {
            color_sum += wc.w * wc.c;
            weight_sum += wc.w;
        }

        if (weight_sum < 1e-6f) continue;

        Eigen::Vector3f color = color_sum / weight_sum;
        color = color.cwiseMax(0.0f).cwiseMin(255.0f);

        mSampledPoints3D.push_back(Pw);
        //mSampledPoints2D.push_back(Eigen::Vector2f(uv.x, uv.y));
        mSampledPointsColor.emplace_back(
            static_cast<unsigned char>(color(0)),
            static_cast<unsigned char>(color(1)),
            static_cast<unsigned char>(color(2)));
    }
}


bool MapLine::IsValidLineMultiView(float pixel_thresh) {
    auto observations = GetLineObservations();
    if (observations.size() < 3) return false; // 观察帧数太少，置信度低

    float total_error = 0;
    int count = 0;

    for (auto& obs : observations) {
        KeyFrame* pKFi = obs.first;
        int idx = std::get<0>(obs.second);
        
        // 获取该帧检测到的 2D 线 ABC 方程 (ax + by + c = 0)
        float a, b, c;
        const cv::line_descriptor::KeyLine& kl = pKFi->mvKeyLines[idx];
        if (!ComputeLineABCFromKeyLine(kl, a, b, c)) continue;

        // 将 3D 端点投影到该帧
        cv::Point2f uv0, uv1;
        if (!pKFi->ProjectPointToImage(mLsWorldPos, uv0) || 
            !pKFi->ProjectPointToImage(mLeWorldPos, uv1)) continue;

        // 计算 2D 投影点到 2D 检测线的距离
        float d0 = std::fabs(a * uv0.x + b * uv0.y + c) / std::sqrt(a*a + b*b);
        float d1 = std::fabs(a * uv1.x + b * uv1.y + c) / std::sqrt(a*a + b*b);

        total_error += (d0 + d1);
        count += 2;
    }

    if (count == 0) return false;
    return (total_error / count) < pixel_thresh; // pixel_thresh 建议设为 1.5 - 2.0 像素
}

// 在采样循环前增加检查
float MapLine::ComputeMaxParallaxAngle() {
    auto observations = GetLineObservations();
    Eigen::Vector3f mid_p = (mLsWorldPos + mLeWorldPos) * 0.5f;
    float max_cos = -1.0f;

    std::vector<Eigen::Vector3f> view_dirs;
    for (auto& obs : observations) {
        view_dirs.push_back((obs.first->GetCameraCenter() - mid_p).normalized());
    }

    float max_parallax = 0;
    for(size_t i=0; i<view_dirs.size(); ++i) {
        for(size_t j=i+1; j<view_dirs.size(); ++j) {
            float cos_v = view_dirs[i].dot(view_dirs[j]);
            max_parallax = std::max(max_parallax, std::acos(cos_v));
        }
    }
    return max_parallax * 180.0f / M_PI; // 返回角度
}

cv::Vec3b MapLine::AverageColorIgnoreBlack(
    const std::vector<cv::Vec3b>& colors,
    int black_thresh)
{
    if (colors.empty())
        return cv::Vec3b(0, 0, 0);

    int sum_b = 0, sum_g = 0, sum_r = 0;
    int cnt = 0;

    for (const auto& c : colors)
    {
        if (IsBlackZDG(c, black_thresh))
            continue;

        sum_b += c[0];
        sum_g += c[1];
        sum_r += c[2];
        cnt++;
    }

    // 如果全部都是黑色
    if (cnt == 0)
        return cv::Vec3b(0, 0, 0);

    return cv::Vec3b(
        static_cast<uchar>(sum_b / cnt),
        static_cast<uchar>(sum_g / cnt),
        static_cast<uchar>(sum_r / cnt)
    );
}

// 从 KeyLine 端点计算直线 ax + by + c = 0
bool MapLine::ComputeLineABCFromKeyLine(
    const cv::line_descriptor::KeyLine& kl,
    float& a, float& b, float& c)
{
    float x1 = kl.startPointX;
    float y1 = kl.startPointY;
    float x2 = kl.endPointX;
    float y2 = kl.endPointY;

    float dx = x2 - x1;
    float dy = y2 - y1;
    float norm = std::sqrt(dx*dx + dy*dy);
    if (norm < 1e-6f)
        return false;

    // 法向量 (a,b)
    a =  dy / norm;
    b = -dx / norm;
    c = -(a * x1 + b * y1);
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
