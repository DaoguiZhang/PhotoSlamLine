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

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), NL(frame.NL), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvKeyLines(frame.mvKeyLines), mvKeyLinesRight(frame.mvKeyLinesRight), mvKeyLinesUn(frame.mvKeyLinesUn),mvpMapLines(frame.mvpMapLines),
     mvuLineRight(frame.mvuLineRight),mvLineDepth(frame.mvLineDepth), mpLineExtractorLeft(frame.mpLineExtractorLeft), mpLineExtractorRight(frame.mpLineExtractorRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec), mLineDescriptors(frame.mLineDescriptors.clone()),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()), mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mvbLineOutlier(frame.mvbLineOutlier),NLleft (frame.NLleft), NLright(frame.NLright),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mvbOutlierLines(frame.mvbOutlierLines), mnCloseMLs(frame.mnCloseMLs),mLineDescriptorsRight(frame.mLineDescriptorsRight.clone()),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;
    mmProjectLines = frame.mmProjectLines;
    mmMatchedLineInImage = frame.mmMatchedLineInImage;

    imgLeftRGB = frame.imgLeftRGB;
    imgAuxiliary = frame.imgAuxiliary;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const cv::Mat &imRGB, const cv::Mat &imRightRGB, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Save RGB image for Gaussian Mapping
    this->imgLeftRGB = imRGB.clone();
    this->imgAuxiliary = imRightRGB.clone();

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    mmProjectLines.clear();
    mmMatchedLineInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imRGB, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Save RGB image for Gaussian Mapping
    this->imgLeftRGB = imRGB.clone();
    this->imgAuxiliary = imDepth.clone();

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    mmProjectLines.clear();
    mmMatchedLineInImage.clear();

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

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imRGB, const double &timeStamp, ORBextractor* extractor, LSDextractor* lsd_extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),  mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mpLineExtractorLeft(lsd_extractor), mpLineExtractorRight(static_cast<LSDextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Save RGB image for Gaussian Mapping
    this->imgLeftRGB = imRGB.clone();
    this->imgAuxiliary = imDepth.clone();

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

    ExtractORB(0,imGray,0,0);
    ExtractLSD(0, this->imgLeftRGB);   //

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = std::vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    mmProjectLines.clear();
    mmMatchedLineInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    //store line features
    NL = static_cast<int> (mvKeyLines.size());
    //std::cerr <<"NL: " << NL << std::endl;
    if(!mvKeyLines.empty())
    {
        UndistortKeyLines();
        ComputeLineStereoFromRGBD(imDepth);
        mvpMapLines = std::vector<MapLine*>(NL, static_cast<MapLine*>(NULL));

        mvbLineOutlier = std::vector<bool>(NL, false);
    }


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

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;
    monoLineLeft = -1;
    monoLineRight = -1;

    NLleft = -1;
    NLright = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const cv::Mat &imRGB, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Save RGB image for Gaussian Mapping
    this->imgLeftRGB = imRGB.clone();

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();
    mmProjectLines.clear();
    mmMatchedLineInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imRGB, const double &timeStamp, ORBextractor* extractor, LSDextractor* lsd_extractor, ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mpLineExtractorLeft(lsd_extractor), mpLineExtractorRight(static_cast<LSDextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Save RGB image for Gaussian Mapping
    this->imgLeftRGB = imRGB.clone();

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
    ExtractLSD(0, this->imgLeftRGB);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();
    mmProjectLines.clear();
    mmMatchedLineInImage.clear();

    mvbOutlier = vector<bool>(N,false);

     //store line features
    NL = static_cast<int> (mvKeyLines.size());
    if(!mvKeyLines.empty())
    {
        UndistortKeyLines();
        // Set no stereo information
        mvuLineRight.clear();
        mvLineDepth.clear();
        for(int i = 0; i < NL; ++i)
        {
            mvuLineRight.emplace_back(std::make_pair(-1,-1));
            mvLineDepth.emplace_back(std::make_pair(-1,-1));
        }
        mvpMapLines = std::vector<MapLine*>(NL, static_cast<MapLine*>(NULL));

        mvbLineOutlier = std::vector<bool>(NL, false);
    }

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

void Frame::ExtractLSD(int flag, const cv::Mat &im)
{
    // TO DO: Implement line extraction if needed
    vector<int> vLapping = {0,0};
    if(flag == 0)
        (*mpLineExtractorLeft)(im, cv::Mat(), mvKeyLines, mLineDescriptors, vLapping);
    else
        (*mpLineExtractorRight)(im, cv::Mat(), mvKeyLinesRight, mLineDescriptorsRight, vLapping);
}

void Frame::featureSelect(const cv::Mat &im)
{
    // TO DO: Implement feature selection if needed
}

// //Get good line Matchers
// void Frame::lineDescriptorMAD( std::vector<std::vector<cv::DMatch>> matches, double &nn_mad, double &nn12_mad) const
// {
//     // TO DO: Implement line descriptor MAD if needed
// }

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

// bool Frame::isLineInFrustum(MapLine* pML, float minLengthPixels, bool bRight)
// {
//     if(Nleft == -1){
//         pML->mbLineTrackInView = false;
//         pML->mLsTrackProjX = -1;
//         pML->mLsTrackProjY = -1;
//         pML->mLeTrackProjX = -1;
//         pML->mLeTrackProjY = -1;
//         Eigen::Vector3f P1 = pML->GetLineWorldPos().first;
//         Eigen::Vector3f P2 = pML->GetLineWorldPos().second;
//         // 3D in camera coordinates
//         const Eigen::Matrix<float,3,1> Pc1 = mRcw * P1 + mtcw;
//         const Eigen::Matrix<float,3,1> Pc2 = mRcw * P2 + mtcw;
//         // 深度检查
//         if(Pc1(2) <= 0 || Pc2(2) <= 0)
//             return false;
//         // 投影到图像
//         Eigen::Vector2f uv1 = mpCamera->project(Pc1);
//         Eigen::Vector2f uv2 = mpCamera->project(Pc2);
//         // 裁剪到图像边界
//         Eigen::Vector2f uv1_clip = uv1;
//         Eigen::Vector2f uv2_clip = uv2;
//         uv1_clip(0) = std::min(std::max(uv1(0), mnMinX), mnMaxX);
//         uv1_clip(1) = std::min(std::max(uv1(1), mnMinY), mnMaxY);
//         uv2_clip(0) = std::min(std::max(uv2(0), mnMinX), mnMaxX);
//         uv2_clip(1) = std::min(std::max(uv2(1), mnMinY), mnMaxY);
//         // 投影长度判断
//         float lenImg = (uv2_clip - uv1_clip).norm();
//         if(lenImg < minLengthPixels)
//             return false;
//         // 预测尺度
//         Eigen::Vector3f PcCenter = (Pc1 + Pc2) * 0.5f;
//         float dist = (PcCenter - mOw).norm();
//         int nPredictedLevel = pML->PredictScale(dist, this);
//          // Data used by the tracking
//         pML->mbLineTrackInView = true;
//         pML->mLsTrackProjX = uv1(0);
//         pML->mLsTrackProjY = uv1(1);
//         pML->mLeTrackProjX = uv2(0);
//         pML->mLeTrackProjY = uv2(1);
//         //pMP->mTrackProjXR = uv(0) - mbf*invz;
//         pML->mLineTrackDepth = dist;    //MEAN point
//         pML->mnLineTrackScaleLevel= nPredictedLevel;
//         return true;
//     }
//     else
//     {
//         //TO DO next...
//     }
//     return true;
// }


bool Frame::isLineInFrustum(MapLine* pML, float viewingCosLimit)
{
    pML->mbLineTrackInView = false;
    pML->mLsTrackProjX = -1;
    pML->mLsTrackProjY = -1;
    pML->mLeTrackProjX = -1;
    pML->mLeTrackProjY = -1;

    // --- 获取线端点 ---
    Eigen::Vector3f P1w = pML->GetLineWorldPos().first;
    Eigen::Vector3f P2w = pML->GetLineWorldPos().second;

    // --- 转到相机坐标系 ---
    Eigen::Vector3f P1c = mRcw * P1w + mtcw;
    Eigen::Vector3f P2c = mRcw * P2w + mtcw;
    float Z1c = P1c(2), Z2c = P2c(2);

    if (Z1c <= 0.0f && Z2c <= 0.0f)
        return false;

    // --- 投影到像素平面 ---
    Eigen::Vector2f uv1 = mpCamera->project(P1c);
    Eigen::Vector2f uv2 = mpCamera->project(P2c);

    // --- 判断是否部分在图像内 ---
    bool in1 = (uv1(0) >= mnMinX && uv1(0) <= mnMaxX &&
                uv1(1) >= mnMinY && uv1(1) <= mnMaxY);
    bool in2 = (uv2(0) >= mnMinX && uv2(0) <= mnMaxX &&
                uv2(1) >= mnMinY && uv2(1) <= mnMaxY);

    if (!in1 && !in2) {
        // 两端都在外侧，进一步检查是否穿过图像边界
        // 使用线段与矩形相交测试（快速近似）
        float minX = std::min(uv1(0), uv2(0));
        float maxX = std::max(uv1(0), uv2(0));
        float minY = std::min(uv1(1), uv2(1));
        float maxY = std::max(uv1(1), uv2(1));

        if (maxX < mnMinX || minX > mnMaxX ||
            maxY < mnMinY || minY > mnMaxY)
            return false; // 整条线在外侧
    }

    // --- 视角一致性 ---
    Eigen::Vector3f lineDir = P2w - P1w;
    Eigen::Vector3f lineCenter = 0.5f * (P1w + P2w);
    Eigen::Vector3f normal = lineCenter - mOw;
    float dist = normal.norm();
    normal /= dist;
    float viewCos = normal.dot(lineDir.normalized());
    if (std::fabs(viewCos) < viewingCosLimit)
        return false;

    int nPredictedLevel = pML->PredictScale(dist, this);

    // --- 存储结果 ---
    pML->mbLineTrackInView = true;
    pML->mLsTrackProjX = uv1(0);
    pML->mLsTrackProjY = uv1(1);
    pML->mLeTrackProjX = uv2(0);
    pML->mLeTrackProjY = uv2(1);
    pML->mLineTrackDepth = 0.5f * (Z1c + Z2c);
    pML->mLineTrackViewCos = viewCos;
    pML->mnLineTrackScaleLevel = nPredictedLevel;

    return true;
}


// bool Frame::isLineInFrustum(MapLine* pML, float viewingCosLimit)
// {
//     pML->mbLineTrackInView = false;
//     pML->mLsTrackProjX = -1;
//     pML->mLsTrackProjY = -1;
//     pML->mLeTrackProjX = -1;
//     pML->mLeTrackProjY = -1;
//     // 获取线段的两个世界端点
//     Eigen::Vector3f P1w = pML->GetLineWorldPos().first;;
//     Eigen::Vector3f P2w = pML->GetLineWorldPos().second;
//     // ========== [1] 变换到相机坐标系 ==========
//     Eigen::Vector3f P1c = mRcw * P1w + mtcw;
//     Eigen::Vector3f P2c = mRcw * P2w + mtcw;
//     const float Z1c = P1c(2);
//     const float Z2c = P2c(2);
//     // 必须在相机前方
//     if (Z1c <= 0.0f || Z2c <= 0.0f)
//         return false;
//     // ========== [2] 投影到像素坐标 ==========
//     Eigen::Vector2f uv1 = mpCamera->project(P1c);
//     Eigen::Vector2f uv2 = mpCamera->project(P2c);
//     // ========== [3] 检查是否在图像边界 ==========
//     if (uv1(0) < mnMinX || uv1(0) > mnMaxX || uv1(1) < mnMinY || uv1(1) > mnMaxY ||
//         uv2(0) < mnMinX || uv2(0) > mnMaxX || uv2(1) < mnMinY || uv2(1) > mnMaxY)
//         return false;
//     // ========== [5] 检查视角一致性 ==========
//     Eigen::Vector3f lineDir = P2w - P1w; // 世界坐标下的线方向
//     Eigen::Vector3f lineCenter = 0.5f * (P1w + P2w); // 中点
//     Eigen::Vector3f normal = lineCenter - mOw; // 从相机光心指向线中心的向量
//     float dist = normal.norm();
//     normal /= dist;
//     Eigen::Vector3f lineNormal = lineDir / lineDir.norm();
//     float viewCos = normal.dot(lineNormal);
//     // // Check distance is in the scale invariance region of the MapPoint
//     // const float maxDistance = pML->GetMaxDistanceInvariance();
//     // const float minDistance = pML->GetMinDistanceInvariance();
//     // if(dist<minDistance || dist>maxDistance)
//     //         return false;
//     int nPredictedLevel = pML->PredictScale(dist, this);
//     if (std::fabs(viewCos) < viewingCosLimit)
//         return false;
//     // ========== [6] 保存投影信息到 MapLine ==========
//     pML->mbLineTrackInView = true;
//     pML->mLsTrackProjX = uv1(0);
//     pML->mLsTrackProjY = uv1(1);
//     pML->mLeTrackProjX   = uv2(0);
//     pML->mLeTrackProjY   = uv2(1);
//     pML->mLineTrackDepth  = 0.5f * (Z1c + Z2c);
//     pML->mLineTrackViewCos = viewCos;
//     pML->mnLineTrackScaleLevel = nPredictedLevel;
//     return true;
// }


bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

std::vector<size_t> Frame::GetLinesInArea(const float &x1, const float  &y1, const float &x2, const float &y2, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    std::vector<size_t> vIndices;

    std::vector<cv::line_descriptor::KeyLine> vkl = this->mvKeyLinesUn;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

    for (size_t i = 0; i < vkl.size(); i++) {
        cv::line_descriptor::KeyLine keyline = vkl[i];

        // 1. 对比中点距离
        float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                         (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
        if (distance > r * r)
            continue;

        // 2. 比较角度差
        float angle_diff = std::abs(keyline.angle - atan2(y2 - y1, x2 - x1));
        if (angle_diff > r * 0.01)
            continue;

        // 3. 比较金字塔层数
        if (bCheckLevels) {
            if (keyline.octave < minLevel)
                continue;
            if (maxLevel >= 0 && keyline.octave > maxLevel)
                continue;
        }

        vIndices.push_back(i);
    }

    return vIndices;
}

std::vector<size_t> Frame::GetLinesInAreaMean(
        const float &u, const float &v,
        const float r,
        const int minLevel,
        const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(64);

    const float r2 = r * r;
    const float minX = u - r;
    const float maxX = u + r;
    const float minY = v - r;
    const float maxY = v + r;

    int nMinLevel = (minLevel < 0) ? 0 : minLevel;
    int nMaxLevel = (maxLevel >= mnScaleLevels) ? mnScaleLevels - 1 : maxLevel;

    const int N = mvKeyLinesUn.size();
    for (int i = 0; i < N; i++)
    {
        const cv::line_descriptor::KeyLine &kl = mvKeyLinesUn[i];
        const int octave = kl.octave;
        if (octave < nMinLevel || octave > nMaxLevel)
            continue;

        const float u1 = kl.startPointX;
        const float v1 = kl.startPointY;
        const float u2 = kl.endPointX;
        const float v2 = kl.endPointY;

        // 端点落入矩形判断
        bool inArea =
            (u1 >= minX && u1 <= maxX && v1 >= minY && v1 <= maxY) ||
            (u2 >= minX && u2 <= maxX && v2 >= minY && v2 <= maxY);

        if (!inArea)
            continue;

        // 最近点距离判断
        const float du1 = u1 - u;
        const float dv1 = v1 - v;
        const float du2 = u2 - u;
        const float dv2 = v2 - v;

        if ((du1 * du1 + dv1 * dv1 < r2) || (du2 * du2 + dv2 * dv2 < r2))
            vIndices.push_back(i);
    }

    return vIndices;
}


bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::lineDescriptorMAD(const std::vector<std::vector<cv::DMatch>>& matches, double &nn_mad, double &nn12_mad) const
{
    if (matches.empty()) {
        nn_mad = 0;
        nn12_mad = 0;
        return;
    }

    std::vector<std::vector<cv::DMatch>> matches_nn = matches;
    std::vector<std::vector<cv::DMatch>> matches_12 = matches;
    //cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

    // NN距离的MAD
    std::sort(matches_nn.begin(), matches_nn.end(), Frame::compare_descriptor_by_NN_dist);
    double nn_dist_median = matches_nn[matches_nn.size()/2][0].distance;

    for (auto &m : matches_nn)
        m[0].distance = std::fabs(m[0].distance - nn_dist_median);

    std::sort(matches_nn.begin(), matches_nn.end(), Frame::compare_descriptor_by_NN_dist);
    nn_mad = 1.4826 * matches_nn[matches_nn.size()/2][0].distance;

    // NN-12距离差的MAD
    std::sort(matches_12.begin(), matches_12.end(), Frame::compare_descriptor_by_NN_dist);
    double nn12_dist_median = matches_12[matches_12.size()/2][1].distance -
                              matches_12[matches_12.size()/2][0].distance;

    for (auto &m : matches_12)
        m[0].distance = std::fabs((m[1].distance - m[0].distance) - nn12_dist_median);

    std::sort(matches_12.begin(), matches_12.end(), Frame::compare_descriptor_by_NN_dist);
    nn12_mad = 1.4826 * matches_12[matches_12.size()/2][0].distance;
}


// void Frame::lineDescriptorMAD( std::vector<std::vector<cv::DMatch>> matches, double &nn_mad, double &nn12_mad) const
// {
//     // TO DO: Implement line descriptor MAD if needed
//     vector<vector<DMatch>> matches_nn, matches_12;
//     matches_nn = line_matches;
//     matches_12 = line_matches;
// //    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;
//     // estimate the NN's distance standard deviation
//     double nn_dist_median;
//     std::sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
//     nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;
//     for(unsigned int i=0; i<matches_nn.size(); i++)
//         matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
//     std::sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
//     nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;
//     // estimate the NN's 12 distance standard deviation
//     double nn12_dist_median;
//     std::sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
//     nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
//     for (unsigned int j=0; j<matches_12.size(); j++)
//         matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
//     sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
//     nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
// }


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::UndistortKeyLines()
{
    //line not undistorted
    if (true)
    {
        mvKeyLinesUn = mvKeyLines;
        return;
    }
    // if (mDistCoef.at<float>(0) == 0.0)
    // {
    //     mvKeyLinesUn = mvKeyLines;
    //     return;
    // }
    // const int N = mvKeyLines.size();
    // if (N == 0)
    //     return;
    // // 准备 Nx2x2 矩阵（每条线两个端点）
    // cv::Mat mat(2*N, 2, CV_32F);
    // for (int i = 0; i < N; i++)
    // {
    //     mat.at<float>(2*i, 0)   = mvKeyLines[i].startPointX;
    //     mat.at<float>(2*i, 1)   = mvKeyLines[i].startPointY;
    //     mat.at<float>(2*i+1, 0) = mvKeyLines[i].endPointX;
    //     mat.at<float>(2*i+1, 1) = mvKeyLines[i].endPointY;
    // }
    // // reshape为 (N*2,1,2)，便于undistort
    // mat = mat.reshape(2);
    // cv::undistortPoints(mat, mat, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
    // mat = mat.reshape(1);
    // // 填充校正后的线段
    // mvKeyLinesUn.resize(N);
    // for (int i = 0; i < N; i++)
    // {
    //     cv::line_descriptor::KeyLine kl = mvKeyLines[i];
    //     kl.startPointX = mat.at<float>(2*i, 0);
    //     kl.startPointY = mat.at<float>(2*i, 1);
    //     kl.endPointX   = mat.at<float>(2*i+1, 0);
    //     kl.endPointY   = mat.at<float>(2*i+1, 1);
    //     // 更新线段中心点
    //     kl.pt.x = 0.5f * (kl.startPointX + kl.endPointX);
    //     kl.pt.y = 0.5f * (kl.startPointY + kl.endPointY);
    //     mvKeyLinesUn[i] = kl;
    // }
}


void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

void Frame::ComputeStereoLineMatches()
{
    const int NL = mvKeyLines.size();
    mvuLineRight = std::vector<std::pair<float, float>>(NL, {-1.0f, -1.0f});
    mvLineDepth = std::vector<std::pair<float, float>>(NL, {-1.0f, -1.0f});
    std::vector<std::pair<float, float>> mvLineDepthConfidenceEndpoints(NL, {0.0f, 0.0f});
    mvLineDepthConfidence = std::vector<float>(NL, 0.0f);

    if (mvKeyLinesRight.empty() || mvKeyLines.empty())
        return;

    const int thDescDist = (LSDmatcher::TH_HIGH + LSDmatcher::TH_LOW) / 2;
    const float thAngle = 10.0f * CV_PI / 180.0f;
    const float thLengthRatio = 0.5f;

    const float minZ = mb;
    const float minDisp = 0.0f;
    const float maxDisp = mbf / minZ;
    const float thConf = 0.3f; // 置信度阈值

    const int nRows = mpLineExtractorLeft->mvImagePyramid[0].rows;
    std::vector<std::vector<size_t>> vRowIndices(nRows);

    // 按行分桶右图线段
    for (int i = 0; i < (int)mvKeyLinesRight.size(); i++)
    {
        const float yR = 0.5f * (mvKeyLinesRight[i].startPointY + mvKeyLinesRight[i].endPointY);
        const int rmin = std::max(0, (int)floor(yR - 2.0f));
        const int rmax = std::min(nRows - 1, (int)ceil(yR + 2.0f));
        for (int r = rmin; r <= rmax; r++)
            vRowIndices[r].push_back(i);
    }

    for (int iL = 0; iL < NL; iL++)
    {
        const cv::line_descriptor::KeyLine &klL = mvKeyLines[iL];
        const float yL = 0.5f * (klL.startPointY + klL.endPointY);
        if (yL < 0 || yL >= nRows) continue;

        const auto &vCandidates = vRowIndices[(int)yL];
        if (vCandidates.empty()) continue;

        const cv::Mat dL = mLineDescriptors.row(iL);

        float bestScore = 1e9;
        int bestIdxR = -1;

        for (size_t c = 0; c < vCandidates.size(); c++)
        {
            const int iR = vCandidates[c];
            const cv::line_descriptor::KeyLine &klR = mvKeyLinesRight[iR];

            // 方向约束
            float angleDiff = fabs(klL.angle - klR.angle);
            if (angleDiff > thAngle && fabs(angleDiff - CV_PI) > thAngle)
                continue;

            // 长度约束
            float lenL = klL.lineLength;
            float lenR = klR.lineLength;
            if (fabs(lenL - lenR) / lenL > thLengthRatio)
                continue;

            // 端点视差一致性
            float u1L = klL.startPointX, u2L = klL.endPointX;
            float u1R = klR.startPointX, u2R = klR.endPointX;
            float disp1 = u1L - u1R, disp2 = u2L - u2R;

            if (disp1 <= minDisp || disp2 <= minDisp || disp1 > maxDisp || disp2 > maxDisp)
                continue;
            if (fabs(disp1 - disp2) > 2.0f)
                continue;

            // 描述子距离
            const cv::Mat dR = mLineDescriptorsRight.row(iR);
            const int dist = LSDmatcher::DescriptorDistance(dL, dR);

            if (dist < bestScore)
            {
                bestScore = dist;
                bestIdxR = iR;
            }
        }

        if (bestIdxR < 0 || bestScore > thDescDist)
            continue;

        // --- 深度计算（端点法） ---
        const cv::line_descriptor::KeyLine &bestR = mvKeyLinesRight[bestIdxR];
        float u1L = klL.startPointX, u2L = klL.endPointX;
        float u1R = bestR.startPointX, u2R = bestR.endPointX;

        float disp1 = u1L - u1R;
        float disp2 = u2L - u2R;

        float Z1 = mbf / disp1;
        float Z2 = mbf / disp2;

        // --- 每个端点单独置信度 ---
        float conf1 = exp(-0.01f * bestScore) * (disp1 / maxDisp);
        float conf2 = exp(-0.01f * bestScore) * (disp2 / maxDisp);

        // 自动剔除异常端点深度
        if (conf1 < thConf) { Z1 = -1.0f; u1R = -1.0f; conf1 = 0.0f; }
        if (conf2 < thConf) { Z2 = -1.0f; u2R = -1.0f; conf2 = 0.0f; }

        mvLineDepth[iL] = {Z1, Z2};
        mvuLineRight[iL] = {u1R, u2R};
        mvLineDepthConfidenceEndpoints[iL] = {conf1, conf2};

        // 总置信度可以取两端平均
        mvLineDepthConfidence[iL] = 0.5f * (conf1 + conf2);
    }
}

void Frame::ComputeStereoLineMatchesRobustEndpoints()
{
    const int NL = mvKeyLines.size();
    mvuLineRight = std::vector<std::pair<float,float>>(NL, {-1.0f, -1.0f});
    mvLineDepth  = std::vector<std::pair<float,float>>(NL, {-1.0f, -1.0f});
    std::vector<std::pair<float,float>> mvLineDepthConfidenceEndpoints(NL, {0.0f,0.0f});
    mvLineDepthConfidence = std::vector<float>(NL, 0.0f);

    if (mvKeyLinesRight.empty() || mvKeyLines.empty()) return;

    const int thDescDist = (LSDmatcher::TH_HIGH + LSDmatcher::TH_LOW)/2;
    const float minZ = mb;
    const float minDisp = 0.0f;
    const float maxDisp = mbf/minZ;
    const float maxEndpointDispRelDiff = 0.25f;
    const float maxAngleDiffDeg = 20.0f;
    const float minLenRatio = 0.7f;
    const float maxLenRatio = 1.3f;
    const float thConf = 0.3f;

    // 按行分桶右线段
    const int nRows = mpLineExtractorLeft->mvImagePyramid[0].rows;
    std::vector<std::vector<size_t>> vRowIndices(nRows);
    for (size_t i=0; i<mvKeyLinesRight.size(); ++i)
    {
        const auto &klR = mvKeyLinesRight[i];
        const float y = 0.5f*(klR.startPointY + klR.endPointY);
        int ymin = std::max(0,(int)std::floor(y-2.0f));
        int ymax = std::min(nRows-1,(int)std::ceil(y+2.0f));
        for(int r=ymin;r<=ymax;r++) vRowIndices[r].push_back(i);
    }

    std::vector<std::pair<int,int>> vDistIdx; // <descDist, idxL>

    for(int iL=0;iL<NL;iL++)
    {
        const auto &klL = mvKeyLines[iL];
        const float xL_c = 0.5f*(klL.startPointX + klL.endPointX);
        const float yL_c = 0.5f*(klL.startPointY + klL.endPointY);
        if(yL_c<0 || yL_c>=nRows) continue;

        const auto &cands = vRowIndices[(int)std::round(yL_c)];
        if(cands.empty()) continue;

        const float minU = xL_c - maxDisp;
        const float maxU = xL_c - minDisp;
        if(maxU<0) continue;

        int bestDist = LSDmatcher::TH_HIGH;
        int bestIdxR = -1;
        const cv::Mat dL = mLineDescriptors.row(iL);

        for(size_t idx=0; idx<cands.size(); ++idx)
        {
            size_t iR = cands[idx];
            const auto &klR = mvKeyLinesRight[iR];
            const float xR_c = 0.5f*(klR.startPointX + klR.endPointX);
            if(xR_c<minU || xR_c>maxU) continue;
            if(klR.octave < klL.octave-1 || klR.octave > klL.octave+1) continue;

            const cv::Mat dR = mLineDescriptorsRight.row((int)iR);
            int dist = LSDmatcher::DescriptorDistance(dL,dR);
            if(dist<bestDist){ bestDist=dist; bestIdxR=(int)iR; }
        }

        if(bestIdxR<0) continue;
        const auto &klRbest = mvKeyLinesRight[bestIdxR];

        // 端点坐标
        float uL1=klL.startPointX, uL2=klL.endPointX;
        float uR1=klRbest.startPointX, uR2=klRbest.endPointX;
        float d1=uL1-uR1, d2=uL2-uR2;
        if(!(d1>0 && d1<maxDisp && d2>0 && d2<maxDisp)) continue;
        float maxd=std::max(std::abs(d1),std::abs(d2));
        if(maxd<=1e-6f) continue;
        if(std::abs(d1-d2)/maxd>maxEndpointDispRelDiff) continue;

        float angleL=klL.angle, angleR=klRbest.angle;
        float angleDiff=std::fabs(angleL-angleR);
        if(angleDiff>180) angleDiff=std::fmod(angleDiff,180.0f);
        if(angleDiff>maxAngleDiffDeg) continue;

        float lenL=klL.lineLength,lenR=klRbest.lineLength;
        if(lenL<=1e-6f || lenR<=1e-6f) continue;
        float lenRatio = lenR/lenL;
        if(lenRatio<minLenRatio || lenRatio>maxLenRatio) continue;

        // --- 端点法深度 ---
        float Z1 = mbf/d1, Z2 = mbf/d2;
        float conf1 = exp(-0.01f*bestDist)*(d1/maxDisp);
        float conf2 = exp(-0.01f*bestDist)*(d2/maxDisp);

        if(conf1<thConf){ Z1=-1.0f; uR1=-1.0f; conf1=0.0f; }
        if(conf2<thConf){ Z2=-1.0f; uR2=-1.0f; conf2=0.0f; }

        mvLineDepth[iL]={Z1,Z2};
        mvuLineRight[iL]={uR1,uR2};
        mvLineDepthConfidenceEndpoints[iL]={conf1,conf2};
        mvLineDepthConfidence[iL]=0.5f*(conf1+conf2);

        vDistIdx.emplace_back(bestDist,iL);
    }

    // 描述符距离中位数剔除最差匹配
    if(!vDistIdx.empty())
    {
        std::sort(vDistIdx.begin(),vDistIdx.end());
        float median = vDistIdx[vDistIdx.size()/2].first;
        float thDist = 1.4f*1.5f*median;
        for(int i=(int)vDistIdx.size()-1;i>=0;i--)
        {
            if(vDistIdx[i].first<thDist) break;
            int idxL=vDistIdx[i].second;
            mvLineDepth[idxL]={-1.0f,-1.0f};
            mvuLineRight[idxL]={-1.0f,-1.0f};
            mvLineDepthConfidenceEndpoints[idxL]={0.0f,0.0f};
            mvLineDepthConfidence[idxL]=0.0f;
        }
    }
}


// void Frame::ComputeStereoLineMatches()
// {
//     mvuLineRight = std::vector<float>(NL, -1.0f);
//     mvLineDepth = std::vector<float>(NL, -1.0f);
//     const int thDescDist = (LSDmatcher::TH_HIGH + LSDmatcher::TH_LOW) / 2;
//     if (mvKeyLinesRight.empty() || mvKeyLines.empty())
//         return;
//     // Set limits for disparity search
//     const float minZ = mb;
//     const float minDisp = 0.0f;
//     const float maxDisp = mbf / minZ;
//     // Candidate right lines per row
//     const int nRows = mpLineExtractorLeft->mvImagePyramid[0].rows;
//     std::vector<std::vector<std::size_t>> vRowIndices(nRows);
//     for (int i = 0; i < mvKeyLinesRight.size(); i++) {
//         const float y = (mvKeyLinesRight[i].startPointY + mvKeyLinesRight[i].endPointY) * 0.5f;
//         const int rmin = std::max(0, (int)floor(y - 2.0f));
//         const int rmax = std::min(nRows - 1, (int)ceil(y + 2.0f));
//         for (int r = rmin; r <= rmax; r++)
//             vRowIndices[r].push_back(i);
//     }
//     std::vector<std::pair<int,int>> vDistIdx;
//     vDistIdx.reserve(NL);
//     for (int iL = 0; iL < NL; iL++) {
//         const cv::line_descriptor::KeyLine &klL = mvKeyLines[iL];
//         const float yL = (klL.startPointY + klL.endPointY) * 0.5f;
//         const float xL = (klL.startPointX + klL.endPointX) * 0.5f;
//         if (yL < 0 || yL >= nRows) continue;
//         const std::vector<std::size_t> &vCandidates = vRowIndices[(int)yL];
//         if (vCandidates.empty()) continue;
//         const float minU = xL - maxDisp;
//         const float maxU = xL - minDisp;
//         if (maxU < 0) continue;
//         int bestDist = LSDmatcher::TH_HIGH;
//         std::size_t bestIdxR = 0;
//         const cv::Mat dL = mLineDescriptors.row(iL);
//         // Match candidates by descriptor distance
//         for (std::size_t c = 0; c < vCandidates.size(); c++) {
//             const std::size_t iR = vCandidates[c];
//             const cv::line_descriptor::KeyLine &klR = mvKeyLinesRight[iR];
//             const float xR = (klR.startPointX + klR.endPointX) * 0.5f;
//             if (xR >= minU && xR <= maxU) {
//                 const cv::Mat dR = mLineDescriptorsRight.row(iR);
//                 const int dist = LSDmatcher::DescriptorDistance(dL, dR);
//                 if (dist < bestDist) {
//                     bestDist = dist;
//                     bestIdxR = iR;
//                 }
//             }
//         }
//         // Subpixel refinement based on center disparity
//         if (bestDist < thDescDist) {
//             const cv::line_descriptor::KeyLine &bestR = mvKeyLinesRight[bestIdxR];
//             const float xR = (bestR.startPointX + bestR.endPointX) * 0.5f;
//             float disparity = xL - xR;
//             // Sanity check
//             if (disparity < minDisp || disparity > maxDisp)
//                 continue;
//             if (disparity <= 0.01f)
//                 disparity = 0.01f;
//             mvLineDepth[iL] = mbf / disparity;
//             mvuLineRight[iL] = xR;
//             vDistIdx.emplace_back(bestDist, iL);
//         }
//     }
//     // Filter unreliable matches
//     if (vDistIdx.empty()) return;
//     sort(vDistIdx.begin(), vDistIdx.end());
//     const float median = vDistIdx[vDistIdx.size()/2].first;
//     const float thDist = 1.4f * 1.5f * median;
//     for (int i = vDistIdx.size()-1; i >= 0; i--) {
//         if (vDistIdx[i].first < thDist)
//             break;
//         else {
//             mvuLineRight[vDistIdx[i].second] = -1;
//             mvLineDepth[vDistIdx[i].second] = -1;
//         }
//     }
// }



void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

void Frame::ComputeLineStereoFromRGBD(const cv::Mat &imDepth)
{
    const int rows = imDepth.rows;
    const int cols = imDepth.cols;
    // 初始化存储
    mvuLineRight = std::vector<std::pair<float,float>>(NL, {-1.0f,-1.0f});
    mvLineDepth  = std::vector<std::pair<float,float>>(NL, {-1.0f,-1.0f});
    std::vector<std::pair<float,float>> mvLineDepthConfidenceEndpoints(NL, {0.0f,0.0f});
    mvLineDepthConfidence = std::vector<float>(NL, 0.0f);

    const float minDepth = 0.1f;     
    const float maxDepth = 10.0f;    
    const float maxDepthDiffRatio = 0.2f; 
    const float thConf = 0.3f;       

    for (int i = 0; i < NL; i++)
    {
        const cv::line_descriptor::KeyLine &kl = mvKeyLinesUn[i];
        float u1 = kl.startPointX;
        float v1 = kl.startPointY;
        float u2 = kl.endPointX;
        float v2 = kl.endPointY;
        // 边界检查
        if (u1 < 0 || u1 >= cols || v1 < 0 || v1 >= rows ||
            u2 < 0 || u2 >= cols || v2 < 0 || v2 >= rows)
            continue;
        // 读取两端深度
        float d1 = imDepth.at<float>(cvRound(v1), cvRound(u1));
        float d2 = imDepth.at<float>(cvRound(v2), cvRound(u2));
        bool valid1 = (d1 > minDepth && d1 < maxDepth);
        bool valid2 = (d2 > minDepth && d2 < maxDepth);
        if (!valid1 && !valid2) continue; // 两端都无效
        float Z1 = valid1 ? d1 : -1.0f;
        float Z2 = valid2 ? d2 : -1.0f;
        float conf1 = valid1 ? 1.0f : 0.0f;
        float conf2 = valid2 ? 1.0f : 0.0f;
        // --- 深度一致性判断 ---
        if (valid1 && valid2)
        {
            float diff = std::abs(d1 - d2);
            float avg = 0.5f * (d1 + d2);
            if (diff / avg > maxDepthDiffRatio)
            {
                // 不一致时降低置信度
                float ratio = std::max(0.0f, 1.0f - diff/avg);
                conf1 *= ratio;
                conf2 *= ratio;
            }
        }
        // --- 单端点有效时，用该端点近似整条线 ---
        if (!valid1 && valid2)
        {
            Z1 = Z2;
            u1 = u2;
            conf1 = conf2 * 0.8f; // 略低置信度
        }
        else if (valid1 && !valid2)
        {
            Z2 = Z1;
            u2 = u1;
            conf2 = conf1 * 0.8f;
        }
        // --- 根据置信度剔除异常端点 ---
        if(conf1 < thConf){ Z1=-1.0f; u1=-1.0f; conf1=0.0f; }
        if(conf2 < thConf){ Z2=-1.0f; u2=-1.0f; conf2=0.0f; }

        // --- 计算右图端点坐标 ---
        float uR1 = (Z1>0) ? u1 - mbf/Z1 : -1.0f;
        float uR2 = (Z2>0) ? u2 - mbf/Z2 : -1.0f;

        mvLineDepth[i] = {Z1, Z2};
        mvuLineRight[i] = {uR1, uR2};
        mvLineDepthConfidenceEndpoints[i] = {conf1, conf2};
        mvLineDepthConfidence[i] = 0.5f*(conf1 + conf2);
    }
}


// void Frame::ComputeLineStereoFromRGBD(const cv::Mat &imDepth)
// {
//     const int rows = imDepth.rows;
//     const int cols = imDepth.cols;
//     // 初始化存储
//     mvuLineRight = std::vector<float>(NL, -1.0f);
//     mvLineDepth  = std::vector<float>(NL, -1.0f);
//     const float minDepth = 0.1f;     // 深度下限（单位：m）
//     const float maxDepth = 10.0f;    // 深度上限（单位：m）
//     const float maxDepthDiffRatio = 0.2f; // 允许两端深度相差不超过20%
//     for (int i = 0; i < NL; i++)
//     {
//         const cv::line_descriptor::KeyLine &kl = mvKeyLinesUn[i];
//         float u1 = kl.startPointX;
//         float v1 = kl.startPointY;
//         float u2 = kl.endPointX;
//         float v2 = kl.endPointY;
//         // --- 1. 边界检查 ---
//         if (u1 < 0 || u1 >= cols || v1 < 0 || v1 >= rows ||
//             u2 < 0 || u2 >= cols || v2 < 0 || v2 >= rows)
//             continue;
//         // --- 2. 读取两端深度，跳过无效值 ---
//         float d1 = imDepth.at<float>(cvRound(v1), cvRound(u1));
//         float d2 = imDepth.at<float>(cvRound(v2), cvRound(u2));
//         bool valid1 = (d1 > minDepth && d1 < maxDepth);
//         bool valid2 = (d2 > minDepth && d2 < maxDepth);
//         if (!valid1 && !valid2)
//             continue;  // 两端都没深度，跳过
//         float d = -1.0f;
//         // --- 3. 深度一致性判断 ---
//         if (valid1 && valid2)
//         {
//             float diff = std::abs(d1 - d2);
//             float avg = 0.5f * (d1 + d2);
//             if (diff / avg < maxDepthDiffRatio)
//                 d = avg;   // 两端深度接近，取平均
//             else
//                 continue;  // 深度差太大，不可信
//         }
//         else if (valid1)
//             d = d1;
//         else if (valid2)
//             d = d2;
//         // --- 4. 计算右图x坐标 ---
//         if (d > 0)
//         {
//             mvLineDepth[i] = d;
//             // 计算线段中点
//             float u_mid = 0.5f * (u1 + u2);
//             float disparity = mbf / d;
//             mvuLineRight[i] = u_mid - disparity;
//         }
//     }
// }

void Frame::ExportRGBDDepthAndLinesToOBJ(const cv::Mat &imDepth, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ExportRGBDDepthAndLinesToOBJ] Error: cannot open file " 
                  << filename << std::endl;
        return;
    }
    file << "# RGBD depth cloud + line endpoints in world coordinates\n";
    // -------------------------------
    // 相机参数与位姿
    // -------------------------------
    const float minDepth = 0.1f;
    const float maxDepth = 10.0f;
    // 世界坐标变换
    Eigen::Matrix3f Rcw = GetRcw();
    Eigen::Matrix3f Rwc = Rcw.transpose();
    Eigen::Vector3f tcw = Gettcw();
    Eigen::Vector3f twc = -Rwc * tcw; // 相机中心
    int vertexCount = 0;
    int lineCount = 0;
    // -------------------------------
    // 1️⃣ 输出 imDepth 点云
    // -------------------------------
    const int rows = imDepth.rows;
    const int cols = imDepth.cols;

    for (int v = 0; v < rows; v += 2) {         // 可适当下采样（步长 2）
        for (int u = 0; u < cols; u += 2) {
            float d = imDepth.at<float>(v, u);
            if (d <= minDepth || d >= maxDepth || std::isnan(d))
                continue;
            // 相机坐标系下
            Eigen::Vector3f Pc;
            Pc << (u - cx) * d / fx,
                   (v - cy) * d / fy,
                   d;
            // 转到世界坐标
            Eigen::Vector3f Pw = Rwc * Pc + twc;
            file << "v " << Pw.x() << " " << Pw.y() << " " << Pw.z()
                 << " 0.7 0.7 0.7\n";  // 灰色点云
            vertexCount++;
        }
    }
    // -------------------------------
    // 2️⃣ 输出线段端点（mvLineDepth）
    // -------------------------------
    for (size_t i = 0; i < mvKeyLinesUn.size(); ++i)
    {
        float u1 = mvKeyLinesUn[i].startPointX;
        float v1 = mvKeyLinesUn[i].startPointY;
        float u2 = mvKeyLinesUn[i].endPointX;
        float v2 = mvKeyLinesUn[i].endPointY;
        float Z1 = mvLineDepth[i].first;
        float Z2 = mvLineDepth[i].second;
        if (Z1 <= 0 && Z2 <= 0) continue;
        // 反投影两个端点
        Eigen::Vector3f P1c, P2c;
        P1c << (u1 - cx) * Z1 / fx, (v1 - cy) * Z1 / fy, Z1;
        P2c << (u2 - cx) * Z2 / fx, (v2 - cy) * Z2 / fy, Z2;
        // 转到世界坐标
        Eigen::Vector3f P1w = Rwc * P1c + twc;
        Eigen::Vector3f P2w = Rwc * P2c + twc;
        file << "v " << P1w.x() << " " << P1w.y() << " " << P1w.z()
             << " 1 0 0\n";  // 红色端点
        file << "v " << P2w.x() << " " << P2w.y() << " " << P2w.z()
             << " 1 0 0\n";
        file << "l " << vertexCount + 1 << " " << vertexCount + 2 << "\n";
        vertexCount += 2;
        lineCount++;
    }
    // -------------------------------
    // 3️⃣ 相机坐标轴可视化
    // -------------------------------
    float axisLength = 0.2f;
    Eigen::Vector3f X = twc + axisLength * Rwc.col(0);
    Eigen::Vector3f Y = twc + axisLength * Rwc.col(1);
    Eigen::Vector3f Z = twc + axisLength * Rwc.col(2);
    file << "v " << twc.x() << " " << twc.y() << " " << twc.z() << " 0 0 0\n";
    int camIdx = ++vertexCount;
    file << "v " << X.x() << " " << X.y() << " " << X.z() << " 1 0 0\n";
    file << "l " << camIdx << " " << camIdx + 1 << "\n"; vertexCount++;
    file << "v " << Y.x() << " " << Y.y() << " " << Y.z() << " 0 1 0\n";
    file << "l " << camIdx << " " << camIdx + 2 << "\n"; vertexCount++;
    file << "v " << Z.x() << " " << Z.y() << " " << Z.z() << " 0 0 1\n";
    file << "l " << camIdx << " " << camIdx + 3 << "\n"; vertexCount++;
    file.close();
    std::cout << "[ExportRGBDDepthAndLinesToOBJ] Exported " << vertexCount
              << " vertices, " << lineCount << " lines to "
              << filename << std::endl;
}


bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D, Eigen::Vector3f &colorRGB)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;

        const float uOri = mvKeys[i].pt.x;
        const float vOri = mvKeys[i].pt.y;
        const int ui = static_cast<int>(std::round(uOri));
        const int vi = static_cast<int>(std::round(vOri));
        const cv::Vec3f& color = this->imgLeftRGB.at<cv::Vec3f>(vi, ui);
        colorRGB.x() = color[0];
        colorRGB.y() = color[1];
        colorRGB.z() = color[2];

        return true;
    } else
        return false;
}

bool Frame::UnprojectStereoLineSeg(
    const int &i,
    std::pair<Eigen::Vector3f, Eigen::Vector3f> &xLine3D,
    std::pair<Eigen::Vector3f, Eigen::Vector3f> &colorLine3DRGB)
{
    // 获取端点深度
    const float z1 = mvLineDepth[i].first;   // 左端点深度
    const float z2 = mvLineDepth[i].second;  // 右端点深度

    // 深度有效性检查
    if (z1 <= 0 || z2 <= 0)
        return false;

    // ==== 获取未畸变线段两个端点 ====
    const cv::line_descriptor::KeyLine &klUn = mvKeyLinesUn[i];
    cv::Point2f p1u = klUn.getStartPoint();
    cv::Point2f p2u = klUn.getEndPoint();

    // ==== 像素到相机坐标（分别使用端点各自深度） ====
    const float x1 = (p1u.x - cx) * z1 * invfx;
    const float y1 = (p1u.y - cy) * z1 * invfy;
    const float x2 = (p2u.x - cx) * z2 * invfx;
    const float y2 = (p2u.y - cy) * z2 * invfy;

    Eigen::Vector3f Xc1(x1, y1, z1);
    Eigen::Vector3f Xc2(x2, y2, z2);

    // ==== 相机坐标系 -> 世界坐标系 ====
    xLine3D.first  = mRwc * Xc1 + mOw;
    xLine3D.second = mRwc * Xc2 + mOw;

    // ==== 获取颜色信息（原图端点处） ====
    const cv::line_descriptor::KeyLine &kl = mvKeyLines[i];
    cv::Point2f p1 = kl.getStartPoint();
    cv::Point2f p2 = kl.getEndPoint();

    int u1 = static_cast<int>(std::round(p1.x));
    int v1 = static_cast<int>(std::round(p1.y));
    int u2 = static_cast<int>(std::round(p2.x));
    int v2 = static_cast<int>(std::round(p2.y));

    if (u1 < 0 || u1 >= imgLeftRGB.cols || v1 < 0 || v1 >= imgLeftRGB.rows ||
        u2 < 0 || u2 >= imgLeftRGB.cols || v2 < 0 || v2 >= imgLeftRGB.rows)
        return false;

    const cv::Vec3f &color1 = imgLeftRGB.at<cv::Vec3f>(v1, u1);
    const cv::Vec3f &color2 = imgLeftRGB.at<cv::Vec3f>(v2, u2);

    colorLine3DRGB.first  = Eigen::Vector3f(color1[0], color1[1], color1[2]);
    colorLine3DRGB.second = Eigen::Vector3f(color2[0], color2[1], color2[2]);

    return true;
}


bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

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
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    std::vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    std::vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
