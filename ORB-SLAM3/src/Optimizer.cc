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


#include "Optimizer.h"


#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"


// 【关键修正】引入 checkJacobian 函数的定义
#include <Thirdparty/g2o/g2o/core/jacobian_workspace.h>

#include "G2oTypes.h"
#include "Converter.h"
#include <unordered_map>

#include<mutex>



namespace ORB_SLAM3
{

// --- thresholds (tweak if needed) ---
static const double CHI2_MONO_HARD = 7.0; // point chi2 hard threshold (conservative)
static const double CHI2_STEREO_HARD = 6.5; // stereo
static const double CHI2_LINE_HARD = 10.0; // line-edge hard threshold
static const float MIN_DEPTH = 0.01f; // 1cm
static const float MAX_DEPTH = 100.0f; // 100m
static const double LINE_COLINEARITY_HIGH = 0.95; // PCA variance ratio
static const double LINE_COLINEARITY_LOW = 0.70; // if below -> mark bad

bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
{
    return (a.second < b.second);
}

static int GetMaxVertexId(const g2o::SparseOptimizer& opt)
{
    int maxId = -1;
    for(auto it = opt.vertices().begin(); it != opt.vertices().end(); ++it)
        if(it->first > maxId) maxId = it->first;
    return maxId;
}

static inline bool IsFiniteDepth(double d)
{
    return std::isfinite(d) && (d > 1e-6);
}

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::GlobalBundleAdjustemntWithLine(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    std::vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    std::vector<MapLine*> vpML = pMap->GetAllMapLines();
    BundleAdjustmentWithLine(vpKFs,vpMP,vpML,nIterations,pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustmentWithLine(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const std::vector<MapLine *> &vpML,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    Map* pMap = vpKFs[0]->GetMap();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);


    // Set KeyFrame vertices

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if(pKF->mpCamera2){
                int rightIndex = get<1>(mit->second);

                if(rightIndex != -1 && rightIndex < pKF->mvKeysRight.size()){
                    rightIndex -= pKF->NLeft;

                    Eigen::Matrix<double,2,1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl = pKF-> GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }



        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pKF->SetPose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            pKF->mTcwGBA = Sophus::SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            Sophus::SE3f mTwc = pKF->GetPoseInverse();
            Sophus::SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;

                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }

                for(size_t i2=0, iend=vpEdgesStereo.size(); i2<iend;i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i2];
                    MapPoint* pMP = vpMapPointEdgeStereo[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}



void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    Map* pMap = vpKFs[0]->GetMap();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);


    // Set KeyFrame vertices

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)]<0)
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if(pKF->mpCamera2){
                int rightIndex = get<1>(mit->second);

                if(rightIndex != -1 && rightIndex < pKF->mvKeysRight.size()){
                    rightIndex -= pKF->NLeft;

                    Eigen::Matrix<double,2,1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl = pKF-> GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }



        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pKF->SetPose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            pKF->mTcwGBA = Sophus::SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;

            Sophus::SE3f mTwc = pKF->GetPoseInverse();
            Sophus::SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;

                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }

                for(size_t i2=0, iend=vpEdgesStereo.size(); i2<iend;i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i2];
                    MapPoint* pMP = vpMapPointEdgeStereo[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int nNonFixed = 0;

    // Set KeyFrame vertices
    KeyFrame* pIncKF;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        pIncKF=pKFi;
        bool bFixed = false;
        if(bFixLocal)
        {
            bFixed = (pKFi->mnBALocalForKF>=(maxKFid-1)) || (pKFi->mnBAFixedForKF>=(maxKFid-1));
            if(!bFixed)
                nNonFixed++;
            VP->setFixed(bFixed);
        }
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    if(bFixLocal)
    {
        if(nNonFixed<3)
            return;
    }

    // IMU links
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
                    VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
                    VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
                    VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);
                }
                else
                {
                    VG1 = optimizer.vertex(4*maxKFid+2);
                    VA1 = optimizer.vertex(4*maxKFid+3);
                }

                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);

                if (!bInit)
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                        continue;
                    }
                }
                else
                {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                        continue;
                    }
                }

                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);

                if (!bInit)
                {
                    EdgeGyroRW* egr= new EdgeGyroRW();
                    egr->setVertex(0,VG1);
                    egr->setVertex(1,VG2);
                    Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW* ear = new EdgeAccRW();
                    ear->setVertex(0,VA1);
                    ear->setVertex(1,VA2);
                    Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }

    if (bInit)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);

        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMPs.size(),false);

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


        bool bAllFixed = true;

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnId>maxKFid)
                continue;

            if(!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;

                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if(pKFi->mpCamera2){ // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                }
            }
        }

        if(bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;


    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        if(nLoopId==0)
        {
            Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = Sophus::SE3f(VP->estimate().Rcw[0].cast<float>(),VP->estimate().tcw[0].cast<float>());
            pKFi->mnBAGlobalForKF = nLoopId;

        }
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            if(nLoopId==0)
            {
                pKFi->SetVelocity(VV->estimate().cast<float>());
            }
            else
            {
                pKFi->mVwbGBA = VV->estimate().cast<float>();
            }

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopId==0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

        if(nLoopId==0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopId;
        }

    }

    pMap->IncreaseChangeIndex();
}


int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            //Conventional SLAM
            if(!pFrame->mpCamera2){
                // Monocular observation
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
            //SLAM with respect a rigid body
            else{
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {    //Left camera observation
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }
    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(),
            SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

// int Optimizer::PoseOptimizationWithLineEndLine(Frame *pFrame)
// {
//     g2o::SparseOptimizer optimizer;
//     g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
//     linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
//     g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//     optimizer.setAlgorithm(solver);
//     int nInitialCorrespondences=0;
//     // Set Frame vertex
//     g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//     Sophus::SE3<float> Tcw = pFrame->GetPose();
//     vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
//     vSE3->setId(0);
//     vSE3->setFixed(false);
//     optimizer.addVertex(vSE3);
//     // Set MapPoint vertices (existing code)
//     const int N = pFrame->N;
//     vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
//     vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
//     vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
//     vpEdgesMono.reserve(N);
//     vpEdgesMono_FHR.reserve(N);
//     vnIndexEdgeMono.reserve(N);
//     vnIndexEdgeRight.reserve(N);
//     vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
//     vector<size_t> vnIndexEdgeStereo;
//     vpEdgesStereo.reserve(N);
//     vnIndexEdgeStereo.reserve(N);
//     const float deltaMono = sqrt(5.991);
//     const float deltaStereo = sqrt(7.815);
//     // --- NEW: containers for line edges ---
//     vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose*> vpEdgesLineMono;
//     vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody*> vpEdgesLineMono_FHR;
//     vector<g2o::EdgeStereoSE3ProjectLineXYZOnlyPose*> vpEdgesLineStereo;
//     vector<size_t> vnIndexEdgeLineMono;
//     vector<size_t> vnIndexEdgeLineMonoRight;
//     vector<size_t> vnIndexEdgeLineStereo;
//     vpEdgesLineMono.reserve(pFrame->NL);
//     vnIndexEdgeLineMono.reserve(pFrame->NL);
//     vpEdgesLineMono_FHR.reserve(pFrame->NL);
//     vnIndexEdgeLineMonoRight.reserve(pFrame->NL);
//     vpEdgesLineStereo.reserve(pFrame->NL);
//     vnIndexEdgeLineStereo.reserve(pFrame->NL);
//     const float deltaLineMono = sqrt(9.488);   // chi2 for 4 DOF? choose reasonable threshold (p=0.05, df=4 -> ~9.49)
//     const float deltaLineStereo = sqrt(11.345);// if stereo measurement has higher dof; adjust as needed
//     {
//     unique_lock<mutex> lock(MapPoint::mGlobalMutex);
//     // === existing point observation loop ===
//     for(int i=0; i<N; i++)
//     {
//         MapPoint* pMP = pFrame->mvpMapPoints[i];
//         if(pMP)
//         {
//             //Conventional SLAM
//             if(!pFrame->mpCamera2){
//                 // Monocular observation
//                 if(pFrame->mvuRight[i]<0)
//                 {
//                     nInitialCorrespondences++;
//                     pFrame->mvbOutlier[i] = false;
//                     Eigen::Matrix<double,2,1> obs;
//                     const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
//                     obs << kpUn.pt.x, kpUn.pt.y;
//                     ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
//                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
//                     e->setMeasurement(obs);
//                     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
//                     e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
//                     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                     e->setRobustKernel(rk);
//                     rk->setDelta(deltaMono);
//                     e->pCamera = pFrame->mpCamera;
//                     e->Xw = pMP->GetWorldPos().cast<double>();
//                     optimizer.addEdge(e);
//                     vpEdgesMono.push_back(e);
//                     vnIndexEdgeMono.push_back(i);
//                 }
//                 else  // Stereo observation
//                 {
//                     nInitialCorrespondences++;
//                     pFrame->mvbOutlier[i] = false;
//                     Eigen::Matrix<double,3,1> obs;
//                     const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
//                     const float &kp_ur = pFrame->mvuRight[i];
//                     obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
//                     g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
//                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
//                     e->setMeasurement(obs);
//                     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
//                     Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
//                     e->setInformation(Info);
//                     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                     e->setRobustKernel(rk);
//                     rk->setDelta(deltaStereo);
//                     e->fx = pFrame->fx;
//                     e->fy = pFrame->fy;
//                     e->cx = pFrame->cx;
//                     e->cy = pFrame->cy;
//                     e->bf = pFrame->mbf;
//                     e->Xw = pMP->GetWorldPos().cast<double>();
//                     optimizer.addEdge(e);
//                     vpEdgesStereo.push_back(e);
//                     vnIndexEdgeStereo.push_back(i);
//                 }
//             }
//             //SLAM with respect a rigid body
//             else{
//                 nInitialCorrespondences++;
//                 cv::KeyPoint kpUn;
//                 if (i < pFrame->Nleft) {    //Left camera observation
//                     kpUn = pFrame->mvKeys[i];
//                     pFrame->mvbOutlier[i] = false;
//                     Eigen::Matrix<double, 2, 1> obs;
//                     obs << kpUn.pt.x, kpUn.pt.y;
//                     ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
//                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                     e->setMeasurement(obs);
//                     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
//                     e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
//                     g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                     e->setRobustKernel(rk);
//                     rk->setDelta(deltaMono);
//                     e->pCamera = pFrame->mpCamera;
//                     e->Xw = pMP->GetWorldPos().cast<double>();
//                     optimizer.addEdge(e);
//                     vpEdgesMono.push_back(e);
//                     vnIndexEdgeMono.push_back(i);
//                 }
//                 else {
//                     kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];
//                     Eigen::Matrix<double, 2, 1> obs;
//                     obs << kpUn.pt.x, kpUn.pt.y;
//                     pFrame->mvbOutlier[i] = false;
//                     ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();
//                     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                     e->setMeasurement(obs);
//                     const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
//                     e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
//                     g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                     e->setRobustKernel(rk);
//                     rk->setDelta(deltaMono);
//                     e->pCamera = pFrame->mpCamera2;
//                     e->Xw = pMP->GetWorldPos().cast<double>();
//                     e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());
//                     optimizer.addEdge(e);
//                     vpEdgesMono_FHR.push_back(e);
//                     vnIndexEdgeRight.push_back(i);
//                 }
//             }
//         }
//     }
//     // === NEW: 添加线段观测为约束（mono / stereo / FHR） ===
//     // 假设：pFrame->mvpMapLines存储了当前帧各线段对应的 MapLine 指针（可能为 NULL）
//     //        pFrame->mvKeyLinesUn 存储了线段的无畸变像素端点 (startPointX,startPointY,endPointX,endPointY)
//     for (int iL = 0; iL < pFrame->NL; ++iL)
//     {
//         MapLine* pML = pFrame->mvpMapLines[iL];
//         if (!pML) continue;
//         if (pML->isBad()) continue;
//         if (pFrame->mvbLineOutlier[iL]) continue;
//         // 获取当前帧线段的端点像素（无畸变）
//         const cv::line_descriptor::KeyLine &kl = pFrame->mvKeyLinesUn[iL];
//         // Measurement: [x1,y1,x2,y2]^T (double)
//         Eigen::Matrix<double,4,1> obsLine;
//         obsLine << double(kl.startPointX), double(kl.startPointY),
//                    double(kl.endPointX),   double(kl.endPointY);
//         // MapLine 世界坐标端点（假设 MapLine 提供此接口）
//         std::pair<Eigen::Vector3f, Eigen::Vector3f> lineWorld = pML->GetLineWorldPos();
//         Eigen::Vector3d Xw1 = lineWorld.first.cast<double>();
//         Eigen::Vector3d Xw2 = lineWorld.second.cast<double>();
//         // 这里选择单目线段边类：ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose
//         // --- YOU MUST HAVE an edge class with similar interface ---
//         ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose* eLineMono = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose();
//         eLineMono->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
//         eLineMono->setMeasurement(obsLine);
//         // 信息矩阵：4x4 （你可以根据观测不确定性调整）
//         Eigen::Matrix4d InfoLine = Eigen::Matrix4d::Identity();
//         // scaling by keyline octave uncertainty (use the same schema as points)
//         // 假设每个线段也有 octave 信息存在 pFrame->mvKeyLines (or kl.octave)
//         const float invSigma2Line = pFrame->mvInvLevelSigma2[kl.octave];
//         InfoLine *= double(invSigma2Line);
//         eLineMono->setInformation(InfoLine);
//         g2o::RobustKernelHuber* rkLine = new g2o::RobustKernelHuber;
//         eLineMono->setRobustKernel(rkLine);
//         rkLine->setDelta(deltaLineMono);
//         // 需要在 edge 中保存相机内参 / 世界坐标端点 / 其它必要量
//         eLineMono->pCamera = pFrame->mpCamera; // 如果你的边需要相机模型
//         eLineMono->Xw1 = Xw1;
//         eLineMono->Xw2 = Xw2;
//         optimizer.addEdge(eLineMono);
//         vpEdgesLineMono.push_back(eLineMono);
//         vnIndexEdgeLineMono.push_back(iL);
//         ++nInitialCorrespondences;
//         // 如果是 FHR / stereo 场景，你可以在此处创建对应的 ToBody / Stereo line edges，
//         // 类似上面点的处理。下面给出伪代码供参考（如果你实现了这些类可以启用）:
//         // if (pFrame->mpCamera2) { // FHR case
//         //     ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody* eLineFHR = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody();
//         //     // set measurement, information, robust kernel, Xw1,Xw2, pCamera2, mTrl ...
//         //     // optimizer.addEdge(eLineFHR);
//         //     // vpEdgesLineMono_FHR.push_back(eLineFHR);
//         //     // vnIndexEdgeLineMonoRight.push_back(iL);
//         //     // ++nInitialCorrespondences;
//         // }
//         //
//         // if (/* stereo measurement available for lines */) {
//         //     g2o::EdgeStereoSE3ProjectLineXYZOnlyPose* eLineStereo = new g2o::EdgeStereoSE3ProjectLineXYZOnlyPose();
//         //     // set measurement (maybe 6-dim?), information, robust kernel, fx,fy,cx,cy,bf, Xw1, Xw2...
//         //     // optimizer.addEdge(eLineStereo);
//         //     // vpEdgesLineStereo.push_back(eLineStereo);
//         //     // vnIndexEdgeLineStereo.push_back(iL);
//         //     // ++nInitialCorrespondences;
//         // }
//     }
//     } // end lock(MapPoint::mGlobalMutex)
//     if(nInitialCorrespondences<3)
//         return 0;
//     // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
//     // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
//     const float chi2Mono[4]={5.991,5.991,5.991,5.991};
//     const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
//     // New: line chi2 thresholds (for 4D measurement)
//     const float chi2LineMono[4] = {9.488, 9.488, 9.488, 9.488}; // chi2 0.95 for df=4 ~9.488
//     const float chi2LineStereo[4] = {11.345, 11.345, 11.345, 11.345}; // tune as needed
//     const int its[4]={10,10,10,10};
//     int nBad=0;
//     for(size_t it=0; it<4; it++)
//     {
//         Tcw = pFrame->GetPose();
//         vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
//         optimizer.initializeOptimization(0);
//         optimizer.optimize(its[it]);
//         nBad=0;
//         // --- existing point-edge outlier processing ---
//         for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
//         {
//             ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
//             const size_t idx = vnIndexEdgeMono[i];
//             if(pFrame->mvbOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2();
//             if(chi2>chi2Mono[it])
//             {
//                 pFrame->mvbOutlier[idx]=true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 pFrame->mvbOutlier[idx]=false;
//                 e->setLevel(0);
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         for(size_t i=0, iend=vpEdgesMono_FHR.size(); i<iend; i++)
//         {
//             ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];
//             const size_t idx = vnIndexEdgeRight[i];
//             if(pFrame->mvbOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2();
//             if(chi2>chi2Mono[it])
//             {
//                 pFrame->mvbOutlier[idx]=true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 pFrame->mvbOutlier[idx]=false;
//                 e->setLevel(0);
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
//         {
//             g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];
//             const size_t idx = vnIndexEdgeStereo[i];
//             if(pFrame->mvbOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2();
//             if(chi2>chi2Stereo[it])
//             {
//                 pFrame->mvbOutlier[idx]=true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 e->setLevel(0);
//                 pFrame->mvbOutlier[idx]=false;
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         // === NEW: process line monocular edges ===
//         for(size_t i=0, iend=vpEdgesLineMono.size(); i<iend; i++)
//         {
//             ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose* e = vpEdgesLineMono[i];
//             const size_t idx = vnIndexEdgeLineMono[i];
//             if(pFrame->mvbLineOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2(); // requires edge to compute residual over 4-d measurement
//             if(chi2 > chi2LineMono[it])
//             {
//                 pFrame->mvbLineOutlier[idx] = true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 pFrame->mvbLineOutlier[idx] = false;
//                 e->setLevel(0);
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         // === NEW: process line FHR edges ===
//         for(size_t i=0, iend=vpEdgesLineMono_FHR.size(); i<iend; i++)
//         {
//             ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody* e = vpEdgesLineMono_FHR[i];
//             const size_t idx = vnIndexEdgeLineMonoRight[i];
//             if(pFrame->mvbLineOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2();
//             if(chi2 > chi2LineMono[it])
//             {
//                 pFrame->mvbLineOutlier[idx] = true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 pFrame->mvbLineOutlier[idx] = false;
//                 e->setLevel(0);
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         // === NEW: process line stereo edges ===
//         for(size_t i=0, iend=vpEdgesLineStereo.size(); i<iend; i++)
//         {
//             g2o::EdgeStereoSE3ProjectLineXYZOnlyPose* e = vpEdgesLineStereo[i];
//             const size_t idx = vnIndexEdgeLineStereo[i];
//             if(pFrame->mvbLineOutlier[idx])
//             {
//                 e->computeError();
//             }
//             const float chi2 = e->chi2();
//             if(chi2 > chi2LineStereo[it])
//             {
//                 pFrame->mvbLineOutlier[idx] = true;
//                 e->setLevel(1);
//                 nBad++;
//             }
//             else
//             {
//                 pFrame->mvbLineOutlier[idx] = false;
//                 e->setLevel(0);
//             }
//             if(it==2)
//                 e->setRobustKernel(0);
//         }
//         if(optimizer.edges().size()<10)
//             break;
//     }
//     // Recover optimized pose and return number of inliers
//     g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
//     g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
//     Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(),
//             SE3quat_recov.translation().cast<float>());
//     pFrame->SetPose(pose);
//     // Count inliers: combine point and line inliers (optional)
//     int nInliers = 0;
//     nInliers += int(vpEdgesMono.size());
//     for(size_t i=0;i<vpEdgesMono.size();++i){
//         if(!pFrame->mvbOutlier[vnIndexEdgeMono[i]]) ++nInliers;
//     }
//     nInliers += int(vpEdgesLineMono.size());
//     for(size_t i=0;i<vpEdgesLineMono.size();++i){
//         if(!pFrame->mvbLineOutlier[vnIndexEdgeLineMono[i]]) ++nInliers;
//     }
//     // You can refine the return value as you wish. For backward compatibility with your original implementation:
//     return nInitialCorrespondences - nBad;
// }

int Optimizer::PoseOptimizationWithLine(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // === Pose vertex ===
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // === existing point edges (unchanged) ===
    const int N = pFrame->N;

    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;
    std::vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;
    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    std::vector<size_t> vnIndexEdgeStereo;

    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    // === NEW: line edges (Point-to-Line) ===
    std::vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose_PointToLine*> vpEdgesLineMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine*> vpEdgesLineMono_FHR;
    std::vector<ORB_SLAM3::EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine*> vpEdgesLineStereo;
    std::vector<std::size_t> vnIndexEdgeLineMono, vnIndexEdgeLineMonoRight, vnIndexEdgeLineStereo;

    vpEdgesLineMono.reserve(pFrame->NL);
    vpEdgesLineMono_FHR.reserve(pFrame->NL);
    vpEdgesLineStereo.reserve(pFrame->NL);
    vnIndexEdgeLineMono.reserve(pFrame->NL);
    vnIndexEdgeLineMonoRight.reserve(pFrame->NL);
    vnIndexEdgeLineStereo.reserve(pFrame->NL);

    const float deltaLineMono = sqrt(9.488);
    const float deltaLineStereo = sqrt(11.345);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        // === existing MapPoints edges (unchanged) ===
        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (!pMP) continue;

            // --- same logic as original (points) ---
            if (!pFrame->mpCamera2) {
                // Monocular point
                if (pFrame->mvuRight[i] < 0) {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;
                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    auto *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
                    e->setVertex(0, optimizer.vertex(0));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(deltaMono);
                    e->setRobustKernel(rk);
                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    // Stereo point
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                    auto *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                    e->setVertex(0, optimizer.vertex(0));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(deltaStereo);
                    e->setRobustKernel(rk);
                    e->fx = pFrame->fx; e->fy = pFrame->fy;
                    e->cx = pFrame->cx; e->cy = pFrame->cy; e->bf = pFrame->mbf;
                    e->Xw = pMP->GetWorldPos().cast<double>();
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
            else {
                // 双目或FHR情况，（保留原逻辑）
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {    //Left camera observation
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }

        // === NEW: Line edges ===
        for (int iL = 0; iL < pFrame->NL; ++iL)
        {
            MapLine* pML = pFrame->mvpMapLines[iL];
            if (!pML || pML->isBad()) continue;

            const cv::line_descriptor::KeyLine &kl = pFrame->mvKeyLinesUn[iL];
            Eigen::Vector3d Xw1 = pML->GetLineWorldPos().first.cast<double>();
            Eigen::Vector3d Xw2 = pML->GetLineWorldPos().second.cast<double>();

            if (!std::isfinite(Xw1.norm()) || !std::isfinite(Xw2.norm()))
                continue;

            if ((Xw2 - Xw1).norm() < 1e-6)
                continue;

            // === 单目线段边 ===
            auto *eLine = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose_PointToLine();
            eLine->setVertex(0, optimizer.vertex(0));
            eLine->SetObservedLineByEndpoints(kl.startPointX, kl.startPointY,
                                              kl.endPointX, kl.endPointY);
            eLine->SetXw(Xw1, Xw2);
            eLine->SetCameraIntrinsics(pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy);

            const float invSigma2 = pFrame->mvInvLevelSigma2[kl.octave];
            eLine->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * 0.1);   // 线段边的观测信息
            g2o::RobustKernelHuber* rkL = new g2o::RobustKernelHuber;
            rkL->setDelta(deltaLineMono);
            eLine->setRobustKernel(rkL);

            optimizer.addEdge(eLine);
            vpEdgesLineMono.push_back(eLine);
            vnIndexEdgeLineMono.push_back(iL);
            pFrame->mvbLineOutlier[iL] = false;
            ++nInitialCorrespondences;
        }
    } // lock结束

    if (nInitialCorrespondences < 3)
        return 0;

    const float chi2Mono[4] = {5.991,5.991,5.991,5.991};
    const float chi2Stereo[4] = {7.815,7.815,7.815,7.815};
    const float chi2LineMono[4] = {25,25,25,25}; // 你可以根据线段观测的实际误差分布调整这个阈值
    const float chi2LineStereo[4] = {35.345,35.345,35.345,35.345};
    const int its[4] = {10,10,10,10};

    //std::cerr << "PoseOpt Debug: "
    //      << "edges=" << optimizer.edges().size()
    //      << ", vertices=" << optimizer.vertices().size()
    //      << std::endl;

    if (optimizer.edges().empty())
    {
        std::cerr << "[WARN] PoseOptimizationWithLine: no edges, skip." << std::endl;
        return 0;
    }

    int nBad = 0;
    for (int it = 0; it < 4; it++)
    {
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;

        // === 点 outlier 检查 ===
        for (size_t i = 0; i < vpEdgesMono.size(); i++)
        {
            auto *e = vpEdgesMono[i];
            size_t idx = vnIndexEdgeMono[i];
            const float chi2 = e->chi2();
            if (chi2 > chi2Mono[it]) {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }
            if (it==2) e->setRobustKernel(0);
        }

        for (size_t i = 0; i < vpEdgesStereo.size(); i++)
        {
            auto *e = vpEdgesStereo[i];
            size_t idx = vnIndexEdgeStereo[i];
            const float chi2 = e->chi2();
            if (chi2 > chi2Stereo[it]) {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }
            if (it==2) e->setRobustKernel(0);
        }

        // === 线 outlier 检查 ===
        for (size_t i = 0; i < vpEdgesLineMono.size(); i++)
        {
            auto *e = vpEdgesLineMono[i];
            size_t idx = vnIndexEdgeLineMono[i];
            const float chi2 = e->chi2();
            if (chi2 > chi2LineMono[it]) {
                pFrame->mvbLineOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbLineOutlier[idx] = false;
                e->setLevel(0);
            }
            if (it==2) e->setRobustKernel(0);
        }

        if (optimizer.activeEdges().size() < 5)
        {
            std::cerr << "[WARN] too few edges, stop early" << std::endl;
            break;
        }
            
    }

    // === 更新优化后的位姿 ===
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(), SE3quat_recov.translation().cast<float>());
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;
}



void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, MappingOperation& opr)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    set<MapPoint*> sNumObsMP;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {

                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;


    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }
    num_edges = nEdges;

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);

        opr.addKeyFrame(pKFi);
    }

    //Points
    opr.reserveMapPoints(lLocalMapPoints.size());
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();

        if (!pMP->isRetrived()) {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }

    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_edges = nEdges;

    pMap->IncreaseChangeIndex();
}

void Optimizer::TestEdgeSE3ProjectLine_PoseAndPoints()
{
    using namespace g2o;

    std::cout << "\n============================================\n";
    std::cout << " EdgeSE3ProjectLine_PoseAndPoints Jacobian Check\n";
    std::cout << "============================================\n";

    // --------------------------------------------------
    // 构造 edge
    // --------------------------------------------------
    auto* edge = new EdgeSE3ProjectLine_PoseAndPoints();
    edge->SetCameraIntrinsics(500, 500, 320, 240);
    edge->SetObservedLineByEndpoints(100, 100, 400, 200);

    // --------------------------------------------------
    // Pose
    // --------------------------------------------------
    auto* vPose = new VertexSE3Expmap();
    vPose->setId(0);
    vPose->setEstimate(SE3Quat(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(0.1, 0.2, 0.3)));

    // --------------------------------------------------
    // Points
    // --------------------------------------------------
    auto* vP1 = new VertexSBAPointXYZ();
    vP1->setId(1);
    vP1->setEstimate(Eigen::Vector3d(1.0, 0.5, 4.0));

    auto* vP2 = new VertexSBAPointXYZ();
    vP2->setId(2);
    vP2->setEstimate(Eigen::Vector3d(1.2, 0.6, 4.2));

    edge->setVertex(0, vPose);
    edge->setVertex(1, vP1);
    edge->setVertex(2, vP2);

    // --------------------------------------------------
    // 解析 Jacobian
    // --------------------------------------------------
    edge->computeError();
    edge->linearizeOplus();

    Eigen::Matrix<double,2,6> Jpose_ana = edge->JPose();
    Eigen::Matrix<double,2,3> Jp1_ana   = edge->JP1();
    Eigen::Matrix<double,2,3> Jp2_ana   = edge->JP2();

    // --------------------------------------------------
    // 数值 Jacobian：Pose（用 setEstimate 备份恢复，避免 6/7 维 estimateData 坑）
    // --------------------------------------------------
    Eigen::Matrix<double,2,6> Jpose_num;
    Jpose_num.setZero();

    const SE3Quat pose_backup = vPose->estimate();

    for(int i = 0; i < 6; ++i)
    {
        Eigen::Matrix<double,6,1> dx = Eigen::Matrix<double,6,1>::Zero();

        const double eps_i = (i < 3) ? 1e-6 : 1e-4; // rot(rad) / trans(m)
        dx[i] = eps_i;

        // +eps
        vPose->setEstimate(pose_backup);
        vPose->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_plus = edge->error();

        // -eps
        vPose->setEstimate(pose_backup);
        dx[i] = -eps_i;
        vPose->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_minus = edge->error();

        // restore
        vPose->setEstimate(pose_backup);

        Jpose_num.col(i) = (e_plus - e_minus) / (2.0 * eps_i);
    }

    // --------------------------------------------------
    // 数值 Jacobian：Point 1（同样用 setEstimate 备份恢复）
    // --------------------------------------------------
    Eigen::Matrix<double,2,3> Jp1_num;
    Jp1_num.setZero();

    const Eigen::Vector3d p1_backup = vP1->estimate();

    for(int i = 0; i < 3; ++i)
    {
        Eigen::Vector3d dx = Eigen::Vector3d::Zero();
        const double eps_p = 1e-6;
        dx[i] = eps_p;

        // +eps
        vP1->setEstimate(p1_backup);
        vP1->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_plus = edge->error();

        // -eps
        vP1->setEstimate(p1_backup);
        dx[i] = -eps_p;
        vP1->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_minus = edge->error();

        // restore
        vP1->setEstimate(p1_backup);

        Jp1_num.col(i) = (e_plus - e_minus) / (2.0 * eps_p);
    }

    // --------------------------------------------------
    // 数值 Jacobian：Point 2
    // --------------------------------------------------
    Eigen::Matrix<double,2,3> Jp2_num;
    Jp2_num.setZero();

    const Eigen::Vector3d p2_backup = vP2->estimate();

    for(int i = 0; i < 3; ++i)
    {
        Eigen::Vector3d dx = Eigen::Vector3d::Zero();
        const double eps_p = 1e-6;
        dx[i] = eps_p;

        // +eps
        vP2->setEstimate(p2_backup);
        vP2->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_plus = edge->error();

        // -eps
        vP2->setEstimate(p2_backup);
        dx[i] = -eps_p;
        vP2->oplus(dx.data());
        edge->computeError();
        Eigen::Vector2d e_minus = edge->error();

        // restore
        vP2->setEstimate(p2_backup);

        Jp2_num.col(i) = (e_plus - e_minus) / (2.0 * eps_p);
    }

    // --------------------------------------------------
    // 打印对比
    // --------------------------------------------------
    auto print_diff = [](const std::string& name,
                         const auto& A,
                         const auto& B)
    {
        std::cout << "\n==== " << name << " ====\n";
        std::cout << "Analytic:\n" << A << "\n\n";
        std::cout << "Numeric:\n" << B << "\n\n";
        std::cout << "Max |diff| = "
                  << (A - B).cwiseAbs().maxCoeff()
                  << "\n";
    };

    print_diff("Pose Jacobian",   Jpose_ana, Jpose_num);
    print_diff("Point1 Jacobian", Jp1_ana,   Jp1_num);
    print_diff("Point2 Jacobian", Jp2_ana,   Jp2_num);

    // --------------------------------------------------
    // cleanup
    // --------------------------------------------------
    delete edge;
    delete vPose;
    delete vP1;
    delete vP2;
}


void Optimizer::TestEdgeSE3ProjectLineXYZOnlyPose_PointToLineOld()
{
    using namespace g2o;

    std::cout << "\n============================================\n";
    std::cout << "EdgeSE3ProjectLineXYZOnlyPose_PointToLine Jacobian Check (SAFE)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // ---------- pose ----------
    VertexSE3Expmap vPose;
    vPose.setEstimate(SE3Quat(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(0.1, 0.2, 0.3)
    ));

    // ---------- line endpoints ----------
    Eigen::Vector3d Xw1(1.0, 0.0, 5.0);
    Eigen::Vector3d Xw2(1.2, 0.1, 5.1);

    // ---------- camera ----------
    double fx = 500, fy = 500, cx = 320, cy = 240;

    // ---------- observed line (a u + b v + c = 0) ----------
    double a, b, c;
    {
        double u1=100, v1=100, u2=400, v2=200;
        double dx = u2-u1, dy = v2-v1;
        double n = std::sqrt(dx*dx + dy*dy);
        a =  dy / n;
        b = -dx / n;
        c = -(a*u1 + b*v1);
    }

    auto project = [&](const Eigen::Vector3d& Xc){
        return Eigen::Vector2d(
            fx * Xc(0) / Xc(2) + cx,
            fy * Xc(1) / Xc(2) + cy
        );
    };

    auto projectJac = [&](const Eigen::Vector3d& Xc){
        Eigen::Matrix<double,2,3> J;
        double x=Xc(0), y=Xc(1), z=Xc(2), z2=z*z;
        J << fx/z, 0, -fx*x/z2,
             0, fy/z, -fy*y/z2;
        return J;
    };

    // ---------- analytic Jacobian ----------
    Eigen::Matrix<double,2,6> J_ana;
    {
        Eigen::Vector3d Xc1 = vPose.estimate().map(Xw1);
        Eigen::Vector3d Xc2 = vPose.estimate().map(Xw2);

        Eigen::Matrix<double,3,6> dXc1, dXc2;
        dXc1.setZero(); dXc2.setZero();

        auto fill_dXc = [](Eigen::Matrix<double,3,6>& J, const Eigen::Vector3d& X){
            J(0,1)= X(2);  J(0,2)= -X(1);
            J(1,0)= -X(2); J(1,2)=  X(0);
            J(2,0)= X(1);  J(2,1)= -X(0);
            J.block<3,3>(0,3).setIdentity();
        };

        fill_dXc(dXc1, Xc1);
        fill_dXc(dXc2, Xc2);

        Eigen::RowVector2d ab(a,b);

        J_ana.row(0) = ab * projectJac(Xc1) * dXc1;
        J_ana.row(1) = ab * projectJac(Xc2) * dXc2;
    }

    // ---------- numeric Jacobian ----------
    Eigen::Matrix<double,2,6> J_num;
    J_num.setZero();

    auto compute_error = [&](const VertexSE3Expmap& vp){
        Eigen::Vector2d e;
        Eigen::Vector3d Xc1 = vp.estimate().map(Xw1);
        Eigen::Vector3d Xc2 = vp.estimate().map(Xw2);
        e(0) = a*project(Xc1)(0) + b*project(Xc1)(1) + c;
        e(1) = a*project(Xc2)(0) + b*project(Xc2)(1) + c;
        return e;
    };

    SE3Quat T0 = vPose.estimate();

    for(int i=0;i<6;i++)
    {
        Eigen::Matrix<double,6,1> dx = Eigen::Matrix<double,6,1>::Zero();
        dx(i) = eps;

        vPose.setEstimate(T0);
        vPose.oplus(dx.data());
        Eigen::Vector2d ep = compute_error(vPose);

        dx(i) = -eps;
        vPose.setEstimate(T0);
        vPose.oplus(dx.data());
        Eigen::Vector2d em = compute_error(vPose);

        J_num.col(i) = (ep-em)/(2*eps);
    }

    // ---------- print ----------
    std::cout << "\nAnalytic Jacobian:\n" << J_ana << "\n";
    std::cout << "\nNumeric Jacobian:\n" << J_num << "\n";
    std::cout << "\nMax |diff| = "
              << (J_ana - J_num).cwiseAbs().maxCoeff()
              << "\n";
    std::cout << "------ end test ------\n";
}

void Optimizer::TestEdgeSE3ProjectLineXYZOnlyPose_PointToLine()
{
    std::cout << "\n==============================================" << std::endl;
    std::cout << " TEST START: Safe Math Verification (No Crash) " << std::endl;
    std::cout << "==============================================" << std::endl;

    // 1. 准备数据
    double fx = 1000.0, fy = 1000.0, cx = 500.0, cy = 500.0;
    
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY());
    Eigen::Vector3d t(0.5, -0.2, 3.0); 
    g2o::SE3Quat pose_quat(R, t);
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(pose_quat);

    Eigen::Vector3d Xw1(1.0, 1.0, 0.0);
    Eigen::Vector3d Xw2(2.0, 0.5, 0.5);

    auto* edge = new EdgeSE3ProjectLineXYZOnlyPose_PointToLine();
    edge->resize(1); // 必须分配
    edge->setVertex(0, vSE3);
    edge->SetCameraIntrinsics(fx, fy, cx, cy);
    edge->SetXw(Xw1, Xw2);
    edge->SetObservedLineByEndpoints(600, 550, 700, 450); // 计算 a, b, c

    // ---------------------------------------------------------
    // 2. 计算解析雅可比 (手动调用核心函数，避开崩溃变量)
    // ---------------------------------------------------------
    
    // A. 准备中间变量
    Eigen::Vector3d Xc1 = vSE3->estimate().map(Xw1);
    Eigen::Vector3d Xc2 = vSE3->estimate().map(Xw2);
    double a = edge->getA(); // 需在类中添加 getter 或者把 a 设为 public
    double b = edge->getB();
    Eigen::RowVector2d ab(a, b);

    // B. 调用你的推导函数 (假设改为 public 了)
    Eigen::Matrix<double,2,6> Jp1 = edge->projectJacobian(Xc1);
    Eigen::Matrix<double,2,6> Jp2 = edge->projectJacobian(Xc2);

    // C. 组装最终雅可比 (存到我们自己的安全变量里)
    Eigen::Matrix<double, 2, 6> J_analytic;
    J_analytic.row(0) = ab * Jp1;
    J_analytic.row(1) = ab * Jp2;

    // ---------------------------------------------------------
    // 3. 计算数值雅可比 (标准流程)
    // ---------------------------------------------------------
    Eigen::Matrix<double, 2, 6> J_numeric;
    const double eps = 1e-6; 
    g2o::SE3Quat T_backup = vSE3->estimate();

    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
        delta(i) = eps;
        
        // 左扰动 +eps
        vSE3->setEstimate(g2o::SE3Quat::exp(delta) * T_backup);
        edge->computeError();
        Eigen::Vector2d e_plus = edge->error();

        // 左扰动 -eps
        delta(i) = -eps;
        vSE3->setEstimate(g2o::SE3Quat::exp(delta) * T_backup);
        edge->computeError();
        Eigen::Vector2d e_minus = edge->error();

        J_numeric.col(i) = (e_plus - e_minus) / (2.0 * eps);
    }
    vSE3->setEstimate(T_backup); // 恢复

    // ---------------------------------------------------------
    // 4. 对比结果
    // ---------------------------------------------------------
    std::cout.precision(6);
    std::cout << std::scientific; 
    std::cout << "Analytic (Safe):\n" << J_analytic << "\n\n";
    std::cout << "Numeric:\n" << J_numeric << "\n\n";

    double max_err = (J_analytic - J_numeric).cwiseAbs().maxCoeff();
    std::cout << "Max Error: " << max_err << std::endl;

    if (max_err < 1e-5) std::cout << "\n[SUCCESS] Math is correct!" << std::endl;
    else std::cout << "\n[FAIL] Check formulas." << std::endl;

    delete edge;
    delete vSE3;
}


void Optimizer::TestEdgeSE3ProjectXYZOnlyPose()
{
    using namespace g2o;

    std::cout << "\n============================================\n";
    std::cout << "EdgeSE3ProjectXYZOnlyPose Jacobian Check (SAFE)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // ---------------- 1) camera ----------------
    std::vector<float> cam_params = {500.f, 500.f, 320.f, 240.f};
    auto* cam = new ORB_SLAM3::Pinhole(cam_params);

    // ---------------- 2) vertex ----------------
    auto* vPose = new VertexSE3Expmap();
    vPose->setId(0);
    vPose->setEstimate(SE3Quat(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(0.1, 0.2, 0.3)));

    // ---------------- 3) edge ----------------
    auto* edge = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
    edge->setVertex(0, vPose);
    edge->pCamera = cam;
    edge->Xw = Eigen::Vector3d(1.0, 0.5, 4.0);
    edge->setMeasurement(Eigen::Vector2d(350, 260));
    edge->setInformation(Eigen::Matrix2d::Identity());

    // ---------------- 4) compute error (ok) ----------------
    edge->computeError();
    std::cout << "Initial error: " << edge->error().transpose() << "\n";

    // ---------------- 5) Analytic Jacobian (manual, SAME as your linearizeOplus) ----------------
    Eigen::Matrix<double,2,6> J_ana;
    {
        const SE3Quat Tcw = vPose->estimate();
        const Eigen::Vector3d Xc = Tcw.map(edge->Xw);

        // 先单独调用一次，确认 projectJac 本身不崩
        const Eigen::Matrix<double,2,3> Jproj = cam->projectJac(Xc);

        const double x = Xc[0], y = Xc[1], z = Xc[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        // 注意：这里的 se3 增量顺序是 [w, t] = [wx wy wz tx ty tz]
        SE3deriv << 0.0,  z,  -y, 1.0, 0.0, 0.0,
                   -z, 0.0,   x, 0.0, 1.0, 0.0,
                    y,  -x, 0.0, 0.0, 0.0, 1.0;

        // 你原函数： _jacobianOplusXi = -projectJac * SE3deriv;
        J_ana = - Jproj * SE3deriv;
    }

    // ---------------- 6) Numeric Jacobian (central difference) ----------------
    Eigen::Matrix<double,2,6> J_num;
    J_num.setZero();

    const SE3Quat pose_backup = vPose->estimate();

    for(int i = 0; i < 6; ++i)
    {
        Eigen::Matrix<double,6,1> dx = Eigen::Matrix<double,6,1>::Zero();
        dx[i] = eps;

        // +eps
        vPose->setEstimate(pose_backup);
        vPose->oplus(dx.data());          // g2o 的 se3 增量顺序同样是 [w, t]
        edge->computeError();
        const Eigen::Vector2d e_plus = edge->error();

        // -eps
        vPose->setEstimate(pose_backup);
        dx[i] = -eps;
        vPose->oplus(dx.data());
        edge->computeError();
        const Eigen::Vector2d e_minus = edge->error();

        // restore
        vPose->setEstimate(pose_backup);

        J_num.col(i) = (e_plus - e_minus) / (2.0 * eps);
    }

    // ---------------- 7) print ----------------
    std::cout << "\nAnalytic Jacobian (manual):\n" << J_ana << "\n";
    std::cout << "\nNumeric Jacobian:\n"  << J_num << "\n";
    std::cout << "\nMax |diff| = " << (J_ana - J_num).cwiseAbs().maxCoeff() << "\n";
    std::cout << "------ end test ------\n";

    // ---------------- 8) cleanup ----------------
    delete edge;
    delete vPose;
    delete cam;
}

// Local Bundle Adjustment with Line Support, 但是没有优化线段
void Optimizer::LocalBundleAdjustmentWithLine(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_edges,
    int& num_Lines,
    MappingOperation& opr)
{
    // --- 1. collect local keyframes (BFS) ---
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFi : vNeighKFs)
    {
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }
    // --- 2. collect local MapPoints ---
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        if (pKFi->mnId == pMap->GetInitKFid())
            num_fixedKF = 1;
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if (pMP->mnBALocalForKF != pKF->mnId)
                {
                    pMP->mnBALocalForKF = pKF->mnId;
                    lLocalMapPoints.push_back(pMP);
                }
            }
        }
    }
    // --- 3. collect fixed keyframes (that see local map points but are not local) ---
    list<KeyFrame*> lFixedCameras;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        const map<KeyFrame*, tuple<int,int>>& obs = pMP->GetObservations();
        for (auto & mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // --- 4. setup optimizer ---
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);
    unsigned long maxKFid = 0;
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();
    // --- 5. add local keyframe vertices (optimizable) ---
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    // --- 6. add fixed keyframe vertices ---
    for (KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }
    // --- 7. add MapPoint vertices + edges (same as original) ---
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    std::vector<KeyFrame*> vpEdgeKFMono, vpEdgeKFBody, vpEdgeKFStereo;
    std::vector<MapPoint*> vpMapPointEdgeMono, vpMapPointEdgeBody, vpMapPointEdgeStereo;
    vpEdgesMono.reserve(1000);
    vpEdgesBody.reserve(1000);
    vpEdgesStereo.reserve(1000);
    int nEdges = 0;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*, tuple<int,int>>& observations = pMP->GetObservations();
        for (auto & mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            const int leftIndex = get<0>(mit.second);
            if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
            {
                // mono
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,2,1> obs; obs << kpUn.pt.x, kpUn.pt.y;
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                e->pCamera = pKFi->mpCamera;
                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
                nEdges++;
            }
            else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0)
            {
                // stereo
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,3,1> obs; obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberStereo); e->setRobustKernel(rk);
                e->fx = pKFi->fx; e->fy = pKFi->fy; e->cx = pKFi->cx; e->cy = pKFi->cy; e->bf = pKFi->mbf;
                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpMapPointEdgeStereo.push_back(pMP);
                nEdges++;
            }
            // body / right camera observation for systems with mpCamera2
            if (pKFi->mpCamera2)
            {
                int rightIndex = get<1>(mit.second);
                if (rightIndex != -1)
                {
                    rightIndex -= pKFi->NLeft;
                    cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                    Eigen::Matrix<double,2,1> obs; obs << kp.pt.x, kp.pt.y;
                    ORB_SLAM3::EdgeSE3ProjectXYZToBody * e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                    Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());
                    e->pCamera = pKFi->mpCamera2;
                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKFi);
                    vpMapPointEdgeBody.push_back(pMP);
                    nEdges++;
                }
            }
        }
    }
    // --- 8. add MapLine edges (Point-to-Line), using your Edge classes ---
    // collect local maplines observed by local keyframes
    list<MapLine*> lLocalMapLines;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        vector<MapLine*> vpLines = pKFi->GetMapLineMatches();
        for (MapLine* pML : vpLines)
        {
            if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if (pML->mnBALocalForKF != pKF->mnId)
                {
                    pML->mnBALocalForKF = pKF->mnId;
                    lLocalMapLines.push_back(pML);
                }
            }
        }
    }
    // containers for line edges
    std::vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose_PointToLine*> vpEdgesLineMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine*> vpEdgesLineMono_FHR;
    std::vector<ORB_SLAM3::EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine*> vpEdgesLineStereo;
    std::vector<KeyFrame*> vpEdgeKFLineMono, vpEdgeKFLineMono_FHR, vpEdgeKFLineStereo;
    std::vector<MapLine*> vpMapLineEdgeMono, vpMapLineEdgeMono_FHR, vpMapLineEdgeStereo;
    const float deltaLineMono = sqrt(9.488);
    const float deltaLineStereo = sqrt(11.345);
    for (MapLine* pML : lLocalMapLines)
    {
        // get world endpoints
        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3d Xw1 = endpoints.first.cast<double>();
        Eigen::Vector3d Xw2 = endpoints.second.cast<double>();
        // iterate observations of this mapline
        const auto& observations = pML->GetLineObservations();
        for (auto & obs : observations)
        {
            KeyFrame* pKFi = obs.first;
            const int leftIndex = get<0>(obs.second);
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            // measurement: keyline in the KeyFrame
            // you may have stored index in tuple; if GetObservations stores index, adapt accordingly.
            // Here assume obs.second.first is line index in that KF
            int idxLine = get<0>(obs.second); // adapt if your tuple layout different
            if (idxLine < 0 || idxLine >= (int)pKFi->mvKeyLines.size()) continue;
            const cv::line_descriptor::KeyLine &kl = pKFi->mvKeyLines[idxLine];
            // choose edge type by sensor and availability
            if (!pKFi->mpCamera2) // monocular / FHR distinction by system flag
            {
                if (leftIndex != -1 && pKFi->mvuRight[get<0>(obs.second)]<0)   //monocular
                {
                    // mono point-to-line unary edge
                    auto *eLine = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose_PointToLine();
                    eLine->setVertex(0, optimizer.vertex(pKFi->mnId));
                    eLine->SetObservedLineByEndpoints(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY);
                    eLine->SetXw(Xw1, Xw2);
                    eLine->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kl.octave];
                    eLine->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rkL = new g2o::RobustKernelHuber; rkL->setDelta(deltaLineMono); eLine->setRobustKernel(rkL);
                    optimizer.addEdge(eLine);
                    vpEdgesLineMono.push_back(eLine);
                    vpEdgeKFLineMono.push_back(pKFi);
                    vpMapLineEdgeMono.push_back(pML);
                    nEdges++;
                }
                else
                {
                    // treat other sensors that have no mpCamera2 as body-type if needed:
                    auto *eLine = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPose_PointToLine();
                    eLine->setVertex(0, optimizer.vertex(pKFi->mnId));
                    eLine->SetObservedLineByEndpoints(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY);
                    eLine->SetXw(Xw1, Xw2);
                    eLine->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kl.octave];
                    eLine->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rkL = new g2o::RobustKernelHuber; rkL->setDelta(deltaLineMono); eLine->setRobustKernel(rkL);
                    optimizer.addEdge(eLine);
                    vpEdgesLineMono.push_back(eLine);
                    vpEdgeKFLineMono.push_back(pKFi);
                    vpMapLineEdgeMono.push_back(pML);
                    nEdges++;
                }
            }
            else
            {
                // if KF has mpCamera2: we may have body-edge variant or stereo (if right keylines exist)
                if (pKFi->mpCamera2 && pKFi->mvKeyLinesRight.size() == pKFi->NL) // stereo line available
                {
                    const cv::line_descriptor::KeyLine &klR = pKFi->mvKeyLinesRight[idxLine];
                    //Pw1(p1w), Pw2(p2w), mK(K), m_bf(bf)
                    auto *eLine = new ORB_SLAM3::EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine(Xw1, Xw2, pKFi->mpCamera->toK(), pKFi->mbf);
                    eLine->setVertex(0, optimizer.vertex(pKFi->mnId));
                    // use stereo observed endpoints: left & right endpoints - here use API that exists on your class
                    // we use SetObservedLines(...) or SetObservedLineByEndpointsStereo if available
                    // assume your stereo edge has SetObservedLineByEndpointsStereo(...) as in earlier snippet
                    //端点换成直线来处理，后续再做
                    // eLine->SetObservedLines(
                    //     kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY,
                    //     klR.startPointX, klR.startPointY, klR.endPointX, klR.endPointY
                    // );
                    //eLine->SetXw(Xw1, Xw2);
                    // pass intrinsics and bf
                    //cv::Mat K = pKFi->mpCamera->toK();
                    //eLine->mK = K; // if public; else use setter
                    //eLine->m_bf = pKFi->mbf;
                    // if your class uses SetCameraIntrinsics, call it:
                    //eLine->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kl.octave];
                    eLine->setInformation(Eigen::Matrix4d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rkLs = new g2o::RobustKernelHuber; rkLs->setDelta(deltaLineStereo);
                    eLine->setRobustKernel(rkLs);
                    optimizer.addEdge(eLine);
                    vpEdgesLineStereo.push_back(eLine);
                    vpEdgeKFLineStereo.push_back(pKFi);
                    vpMapLineEdgeStereo.push_back(pML);
                    nEdges++;
                }
                else
                {
                    // body-type edge (to body camera transform)
                    auto *eLine = new ORB_SLAM3::EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine();
                    eLine->setVertex(0, optimizer.vertex(pKFi->mnId));
                    eLine->SetObservedLineByEndpoints(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY);
                    eLine->SetXw(Xw1, Xw2);
                    eLine->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                    // set Trl (body->left) if your edge provides setter:
                    Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                    eLine->SetTrl(g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>()));
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kl.octave];
                    eLine->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rkL = new g2o::RobustKernelHuber; rkL->setDelta(deltaLineMono); eLine->setRobustKernel(rkL);
                    optimizer.addEdge(eLine);
                    vpEdgesLineMono_FHR.push_back(eLine);
                    vpEdgeKFLineMono_FHR.push_back(pKFi);
                    vpMapLineEdgeMono_FHR.push_back(pML);
                    nEdges++;
                }
            }
        } // end observations loop of this MapLine
    } // end for each MapLine
    // --- 9. run optimization (you can do robust iterations as desired) ---
    if (pbStopFlag && *pbStopFlag) return;
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    // --- 10. outlier detection (points) ---
    vector<pair<KeyFrame*,MapPoint*>> vToErasePoints;
    vToErasePoints.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());
    for (size_t i = 0; i < vpEdgesMono.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFMono[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesBody.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFBody[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesStereo.size(); ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if (!pMP) continue;
        if (e->chi2() > 7.815 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFStereo[i], pMP);
    }
    // --- 11. outlier detection (lines) ---
    vector<pair<KeyFrame*,MapLine*>> vToEraseLines;
    vToEraseLines.reserve(vpEdgesLineMono.size() + vpEdgesLineMono_FHR.size() + vpEdgesLineStereo.size());
    for (size_t i = 0; i < vpEdgesLineMono.size(); ++i)
    {
        auto *e = vpEdgesLineMono[i];
        MapLine* pML = vpMapLineEdgeMono[i];
        if (!pML) continue;
        if (e->chi2() > (deltaLineMono*deltaLineMono))
            vToEraseLines.emplace_back(vpEdgeKFLineMono[i], pML);
    }
    for (size_t i = 0; i < vpEdgesLineMono_FHR.size(); ++i)
    {
        auto *e = vpEdgesLineMono_FHR[i];
        MapLine* pML = vpMapLineEdgeMono_FHR[i];
        if (!pML) continue;
        if (e->chi2() > (deltaLineMono*deltaLineMono))
            vToEraseLines.emplace_back(vpEdgeKFLineMono_FHR[i], pML);
    }
    for (size_t i = 0; i < vpEdgesLineStereo.size(); ++i)
    {
        auto *e = vpEdgesLineStereo[i];
        MapLine* pML = vpMapLineEdgeStereo[i];
        if (!pML) continue;
        if (e->chi2() > (deltaLineStereo*deltaLineStereo))
            vToEraseLines.emplace_back(vpEdgeKFLineStereo[i], pML);
    }
    // --- 12. apply erasures under map mutex ---
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        for (auto & pr : vToErasePoints)
        {
            KeyFrame* pKFi = pr.first;
            MapPoint* pMPi = pr.second;
            if (pKFi && pMPi)
            {
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        for (auto & pr : vToEraseLines)
        {
            KeyFrame* pKFi = pr.first;
            MapLine* pMLi = pr.second;
            if (pKFi && pMLi)
            {
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
    }
    // --- 13. write back optimized poses and points ---
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
        opr.addKeyFrame(pKFi);
    }
    opr.reserveMapPoints(lLocalMapPoints.size());
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if (!pMP->isRetrived()) { pMP->setRetrived(true); opr.addMapPoint(pMP); }
    }
    // Note: MapLine world endpoints are not modified here because edges are unary (pose-only).
    // If you later add VertexLineXYZ and make edges connect (vertexLine, vertexPose),
    // you would fetch and write optimized endpoints similar to MapPoints.
    opr.reserveMapLines(lLocalMapLines.size());
    for (MapLine* pML : lLocalMapLines)
    {
        // keep mapline as-is; mark retrieved for reporting
        if (!pML->isRetrived()) { pML->setRetrived(true); opr.addMapLine(pML); }
    }
    pMap->IncreaseChangeIndex();
    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_edges = nEdges;
    num_Lines = lLocalMapLines.size();
}

// Local Bundle Adjustment with Line Support, and lines are optimized
void Optimizer::LocalBundleAdjustmentWithLine_Optimization(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_edges,
    int& num_Lines,
    MappingOperation& opr)
{
    // --- 1. collect local keyframes (BFS) ---
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFi : vNeighKFs)
    {
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }
    // --- 2. collect local MapPoints ---
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        if (pKFi->mnId == pMap->GetInitKFid())
            num_fixedKF = 1;
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if (pMP->mnBALocalForKF != pKF->mnId)
                {
                    pMP->mnBALocalForKF = pKF->mnId;
                    lLocalMapPoints.push_back(pMP);
                }
            }
        }
    }
    // --- 3. collect fixed keyframes (that see local map points but are not local) ---
    list<KeyFrame*> lFixedCameras;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        const map<KeyFrame*, tuple<int,int>>& obs = pMP->GetObservations();
        for (auto & mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // --- 4. setup optimizer ---
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
       solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);
    unsigned long maxKFid = 0;
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();
    // --- 5. add local keyframe vertices (optimizable) ---
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    // --- 6. add fixed keyframe vertices ---
    for (KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }
    // --- 7. add MapPoint vertices + edges (same as original) ---
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    std::vector<KeyFrame*> vpEdgeKFMono, vpEdgeKFBody, vpEdgeKFStereo;
    std::vector<MapPoint*> vpMapPointEdgeMono, vpMapPointEdgeBody, vpMapPointEdgeStereo;
    vpEdgesMono.reserve(1000);
    vpEdgesBody.reserve(1000);
    vpEdgesStereo.reserve(1000);
    int nEdges = 0;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*, tuple<int,int>>& observations = pMP->GetObservations();
        for (auto & mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            const int leftIndex = get<0>(mit.second);
            if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
            {
                // mono
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,2,1> obs; obs << kpUn.pt.x, kpUn.pt.y;
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                e->pCamera = pKFi->mpCamera;
                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
                nEdges++;
            }
            else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0)
            {
                // stereo
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,3,1> obs; obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberStereo); e->setRobustKernel(rk);
                e->fx = pKFi->fx; e->fy = pKFi->fy; e->cx = pKFi->cx; e->cy = pKFi->cy; e->bf = pKFi->mbf;
                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpMapPointEdgeStereo.push_back(pMP);
                nEdges++;
            }
            // body / right camera observation for systems with mpCamera2
            if (pKFi->mpCamera2)
            {
                int rightIndex = get<1>(mit.second);
                if (rightIndex != -1)
                {
                    rightIndex -= pKFi->NLeft;
                    cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                    Eigen::Matrix<double,2,1> obs; obs << kp.pt.x, kp.pt.y;
                    ORB_SLAM3::EdgeSE3ProjectXYZToBody * e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                    Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());
                    e->pCamera = pKFi->mpCamera2;
                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKFi);
                    vpMapPointEdgeBody.push_back(pMP);
                    nEdges++;
                }
            }
        }
    }
    // --- 8. add MapLine edges (Point-to-Line), using your Edge classes ---
    // collect local maplines observed by local keyframes
    list<MapLine*> lLocalMapLines;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        vector<MapLine*> vpLines = pKFi->GetMapLineMatches();
        for (MapLine* pML : vpLines)
        {
            if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if (pML->mnBALocalForKF != pKF->mnId)
                {
                    pML->mnBALocalForKF = pKF->mnId;
                    lLocalMapLines.push_back(pML);
                }
            }
        }
    }
    // containers for line edges
    std::vector<ORB_SLAM3::EdgeSE3ProjectPointToLine2D*> vpEdgesLineMono;  // pose + line points
    std::vector<KeyFrame*> vpEdgeKFLineMono;
    std::vector<MapLine*> vpMapLineEdgeMono;
    const float deltaLineMono = sqrt(9.488);
    const float deltaLineStereo = sqrt(11.345);

    // Compute a safe offset for line vertex ids so they do not collide with point ids used above
    int nextVertexId = (int)maxKFid + 1;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        int pid = (int)pMP->mnId + (int)maxKFid + 2;
        if (pid > nextVertexId) nextVertexId = pid;
    }

    // --------- NEW: add MapLine endpoint vertices ----------
    unordered_map<MapLine*, pair<int,int>> mapLineVertexId;
    mapLineVertexId.reserve(lLocalMapLines.size() * 2);

    int numLineVertices = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if (!pML || pML->isBad()) continue;

        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3d Xw1 = endpoints.first.cast<double>();
        Eigen::Vector3d Xw2 = endpoints.second.cast<double>();

        Eigen::Vector3d d = Xw2 - Xw1;
        if (!d.allFinite() || d.squaredNorm() < 1e-12) continue;

        // endpoint 1
        auto* vP1 = new g2o::VertexSBAPointXYZ();
        int id1 = ++nextVertexId;
        vP1->setId(id1);
        vP1->setEstimate(Xw1);
        vP1->setFixed(true);
        vP1->setMarginalized(true);

        // endpoint 2
        auto* vP2 = new g2o::VertexSBAPointXYZ();
        int id2 = ++nextVertexId;
        vP2->setId(id2);
        vP2->setEstimate(Xw2);
        vP2->setFixed(false);
        vP2->setMarginalized(true);

        bool ok1 = optimizer.addVertex(vP1);
        bool ok2 = optimizer.addVertex(vP2);

        if (!ok1 || !ok2)
        {
            if (ok1) optimizer.removeVertex(vP1);
            delete vP1;
            delete vP2;
            continue;
        }

        mapLineVertexId[pML] = {id1, id2};
        numLineVertices += 2;
    }
    // add edges using endpoint vertices
    int addedLineEdges = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        int idP1 = it->second.first;
        int idP2 = it->second.second;

        const auto& obs = pML->GetLineObservations();
        for(auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if(!pKFi || pKFi->isBad() || pKFi->GetMap()!=pCurrentMap) continue;
            int idxLine = get<0>(mit.second);
            if(idxLine < 0 || idxLine >= (int)pKFi->mvKeyLines.size()) continue;
            const cv::line_descriptor::KeyLine& kl = pKFi->mvKeyLines[idxLine];
            // build normalized 2D line: a u + b v + c = 0
            const double u1 = kl.startPointX, v1 = kl.startPointY;
            const double u2 = kl.endPointX,   v2 = kl.endPointY;
            const double dx = u2 - u1;
            const double dy = v2 - v1;
            const double na = dy;
            const double nb = -dx;
            const double nrm = std::sqrt(na*na + nb*nb);
            if (!std::isfinite(nrm) || nrm < 1e-12) continue;
            const double a = na / nrm;
            const double b = nb / nrm;
            const double c = -(a*u1 + b*v1);
            Eigen::Vector3d line_abc(a,b,c);
            // invSigma2 guard
            int octave = kl.octave;
            if (octave < 0 || octave >= (int)pKFi->mvInvLevelSigma2.size()) octave = 0;
            const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
            if (!std::isfinite(invSigma2) || invSigma2 <= 0.0) continue;
            auto* vPose = optimizer.vertex(pKFi->mnId);
            auto* vertex1 = optimizer.vertex(idP1);
            auto* vertex2 = optimizer.vertex(idP2);
            if(!vPose || !vertex1 || !vertex2) continue;
            // Edge for endpoint1
            {
                auto* e1 = new EdgeSE3ProjectPointToLine2D();
                e1->setVertex(0, vertex1);      // point
                e1->setVertex(1, vPose);   // pose
                e1->setMeasurement(line_abc);
                e1->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                e1->setInformation(Eigen::Matrix<double,1,1>::Identity());
                auto* rk = new g2o::RobustKernelHuber;
                rk->setDelta(std::sqrt(3.84)); // 1D chi2 95% ~= 3.84
                e1->setRobustKernel(rk);
                optimizer.addEdge(e1);
                vpEdgesLineMono.push_back(e1);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }

            // Edge for endpoint2
            {
                auto* e2 = new EdgeSE3ProjectPointToLine2D();
                e2->setVertex(0, vertex2);      // point
                e2->setVertex(1, vPose);
                e2->setMeasurement(line_abc);
                e2->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                e2->setInformation(Eigen::Matrix<double,1,1>::Identity());

                auto* rk = new g2o::RobustKernelHuber;
                rk->setDelta(std::sqrt(3.84));
                e2->setRobustKernel(rk);

                optimizer.addEdge(e2);
                vpEdgesLineMono.push_back(e2);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }
        }
    }

    std::cerr << "Added " << addedLineEdges << " (binary) line endpoint edges.\n";

    
    // --- 9. run optimization (you can do robust iterations as desired) ---
    if (pbStopFlag && *pbStopFlag) return;
    optimizer.initializeOptimization();
    std::cerr << "Starting optimization with " << nEdges << " edges." << std::endl;
    std::cerr << "Line edges added: " << vpEdgesLineMono.size() << std::endl;
    optimizer.optimize(10);
    std::cerr << "Optimization done." << std::endl;
    // --- 10. outlier detection (points) ---
    vector<pair<KeyFrame*,MapPoint*>> vToErasePoints;
    vToErasePoints.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());
    for (size_t i = 0; i < vpEdgesMono.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFMono[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesBody.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFBody[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesStereo.size(); ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if (!pMP) continue;
        if (e->chi2() > 7.815 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFStereo[i], pMP);
    }
    // --- 11. outlier detection (lines) ---
    vector<pair<KeyFrame*,MapLine*>> vToEraseLines;
    vToEraseLines.reserve(vpEdgesLineMono.size());
    for (size_t i = 0; i < vpEdgesLineMono.size(); ++i)
    {
        auto *e = vpEdgesLineMono[i];
        MapLine* pML = vpMapLineEdgeMono[i];
        if (!pML) continue;
        if (e->chi2() > 2*3.84)
            vToEraseLines.emplace_back(vpEdgeKFLineMono[i], pML);
    }
    // --- 12. apply erasures under map mutex ---
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        for (auto & pr : vToErasePoints)
        {
            KeyFrame* pKFi = pr.first;
            MapPoint* pMPi = pr.second;
            if (pKFi && pMPi)
            {
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        for (auto & pr : vToEraseLines)
        {
            KeyFrame* pKFi = pr.first;
            MapLine* pMLi = pr.second;
            if (pKFi && pMLi)
            {
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
    }
    // --- 13. write back optimized poses and points ---
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
        opr.addKeyFrame(pKFi);
    }
    opr.reserveMapPoints(lLocalMapPoints.size());
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if (!pMP->isRetrived()) { pMP->setRetrived(true); opr.addMapPoint(pMP); }
    }
    // MapLine endpoints are optimized jointly with pose and written back here.
    // If you later add VertexLineXYZ and make edges connect (vertexLine, vertexPose),
    // you would fetch and write optimized endpoints similar to MapPoints.
    opr.reserveMapLines(lLocalMapLines.size());
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        auto* vP1 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.first));
        auto* vP2 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.second));

        pML->SetLineWorldPos(
            vP1->estimate().cast<float>(),
            vP2->estimate().cast<float>()
        );
        // mark retrieved for reporting
        if (!pML->isRetrived()) { pML->setRetrived(true); opr.addMapLine(pML); }
    }   //end for each MapLine
    // keep mapline; mark retrieved for reporting
    pMap->IncreaseChangeIndex();
    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_edges = nEdges;
    num_Lines = lLocalMapLines.size();
}

#if 1   // 调试模式
// Local Bundle Adjustment with Line Support, and lines are optimized + Regularized terms
void Optimizer::LocalBundleAdjustmentWithLine_Optimization_Reg(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_edges,
    int& num_Lines,
    MappingOperation& opr)
{
    // --- 1. collect local keyframes (BFS) ---
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFi : vNeighKFs)
    {
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }
    // --- 2. collect local MapPoints ---
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        if (pKFi->mnId == pMap->GetInitKFid())
            num_fixedKF = 1;
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if (pMP->mnBALocalForKF != pKF->mnId)
                {
                    pMP->mnBALocalForKF = pKF->mnId;
                    lLocalMapPoints.push_back(pMP);
                }
            }
        }
    }
    // --- 3. collect fixed keyframes (that see local map points but are not local) ---
    list<KeyFrame*> lFixedCameras;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        const map<KeyFrame*, tuple<int,int>>& obs = pMP->GetObservations();
        for (auto & mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // --- 4. setup optimizer ---
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
       solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    //optimizer.setVerbose(false);
    optimizer.setVerbose(true);
    if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);
    unsigned long maxKFid = 0;
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();
    // --- 5. add local keyframe vertices (optimizable) ---
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    // --- 6. add fixed keyframe vertices ---
    for (KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }
    // --- 7. add MapPoint vertices + edges (same as original) ---
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    std::vector<KeyFrame*> vpEdgeKFMono, vpEdgeKFBody, vpEdgeKFStereo;
    std::vector<MapPoint*> vpMapPointEdgeMono, vpMapPointEdgeBody, vpMapPointEdgeStereo;
    vpEdgesMono.reserve(1000);
    vpEdgesBody.reserve(1000);
    vpEdgesStereo.reserve(1000);
    int nEdges = 0;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*, tuple<int,int>>& observations = pMP->GetObservations();
        for (auto & mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            const int leftIndex = get<0>(mit.second);
            if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
            {
                // mono
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,2,1> obs; obs << kpUn.pt.x, kpUn.pt.y;
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                e->pCamera = pKFi->mpCamera;
                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
                nEdges++;
            }
            else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0)
            {
                // stereo
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,3,1> obs; obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberStereo); e->setRobustKernel(rk);
                e->fx = pKFi->fx; e->fy = pKFi->fy; e->cx = pKFi->cx; e->cy = pKFi->cy; e->bf = pKFi->mbf;
                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpMapPointEdgeStereo.push_back(pMP);
                nEdges++;
            }
            // body / right camera observation for systems with mpCamera2
            if (pKFi->mpCamera2)
            {
                int rightIndex = get<1>(mit.second);
                if (rightIndex != -1)
                {
                    rightIndex -= pKFi->NLeft;
                    cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                    Eigen::Matrix<double,2,1> obs; obs << kp.pt.x, kp.pt.y;
                    ORB_SLAM3::EdgeSE3ProjectXYZToBody * e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                    Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());
                    e->pCamera = pKFi->mpCamera2;
                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKFi);
                    vpMapPointEdgeBody.push_back(pMP);
                    nEdges++;
                }
            }
        }
    }
    // --- 8. add MapLine edges (Point-to-Line), using your Edge classes ---
    // collect local maplines observed by local keyframes
    list<MapLine*> lLocalMapLines;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        vector<MapLine*> vpLines = pKFi->GetMapLineMatches();
        for (MapLine* pML : vpLines)
        {
            if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if (pML->mnBALocalForKF != pKF->mnId)
                {
                    pML->mnBALocalForKF = pKF->mnId;
                    lLocalMapLines.push_back(pML);
                }
            }
        }
    }
    // containers for line edges
    std::vector<ORB_SLAM3::EdgeSE3ProjectPointToLine2D*> vpEdgesLineMono;  // pose + line points
    std::vector<KeyFrame*> vpEdgeKFLineMono;
    std::vector<MapLine*> vpMapLineEdgeMono;
    const float deltaLineMono = sqrt(9.488);
    const float deltaLineStereo = sqrt(11.345);

    // Compute a safe offset for line vertex ids so they do not collide with point ids used above
    int nextVertexId = (int)maxKFid + 1;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        int pid = (int)pMP->mnId + (int)maxKFid + 2;
        if (pid > nextVertexId) nextVertexId = pid;
    }

    // --------- NEW: add MapLine endpoint vertices ----------
    unordered_map<MapLine*, pair<int,int>> mapLineVertexId;
    mapLineVertexId.reserve(lLocalMapLines.size() * 2);

    // 阈值设置：比如小于 0.1米 (10cm) 或者像素长度小于 20px
    const double min_3d_length_sq = 0.1 * 0.1;
    //const double min_2d_length_sq = 20.0 * 20.0;
    const double min_pixel_len = 40.0;         // 2D 像素长度阈值 (用于降权)

    int numLineVertices = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if (!pML || pML->isBad()) continue;

        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3d Xw1 = endpoints.first.cast<double>();
        Eigen::Vector3d Xw2 = endpoints.second.cast<double>();

        Eigen::Vector3d d = Xw2 - Xw1;
        if (!d.allFinite() || d.squaredNorm() < 1e-12) continue;

        double len_sq = d.squaredNorm();
        // [策略一] 判断是否是短线段
        bool bIsShortLine = (len_sq < min_3d_length_sq);

        // endpoint 1
        auto* vP1 = new g2o::VertexSBAPointXYZ();
        int id1 = ++nextVertexId;
        vP1->setId(id1);
        vP1->setEstimate(Xw1);
        vP1->setFixed(true);
        vP1->setMarginalized(true);

        // endpoint 2
        auto* vP2 = new g2o::VertexSBAPointXYZ();
        int id2 = ++nextVertexId;
        vP2->setId(id2);
        vP2->setEstimate(Xw2);
        vP2->setFixed(false);
        vP2->setMarginalized(true);

        // // [关键修改] 设置 Fixed 属性
        // if (bIsShortLine) 
        // {
        //     // 如果是短线段，直接锁死，不让优化器动它 (防止乱飞)
        //     vP1->setFixed(true);
        //     vP2->setFixed(true);
        // }
        // else 
        // {
        //     // 如果是正常线段，两个端点都允许移动 (你原来锁了vP1，会导致无法平移)
        //     vP1->setFixed(false);
        //     vP2->setFixed(false);
        // }

        bool ok1 = optimizer.addVertex(vP1);
        bool ok2 = optimizer.addVertex(vP2);

        if (!ok1 || !ok2)
        {
            if (ok1) optimizer.removeVertex(vP1);
            delete vP1;
            delete vP2;
            continue;
        }

        mapLineVertexId[pML] = {id1, id2};
        numLineVertices += 2;
    }

    // // --------- add Line Length Prior edges (ONCE per MapLine) ----------
    ///const double lambdaL = 50.0;   // 1~5
    ///const double lambdaD = 20.0;   // 0.5~2
    ///const double lambdaM = 5.0;   // 0.05~0.5  (不要太大，防止锁死)
    // // --------- add Line Length Prior edges (ONCE per MapLine) ----------
    // [修改] 大幅降低权重，让观测数据说话
    //const double lambdaL = 0.5;   // 原 50.0 -> 改 0.5 (允许长度变化)
    //const double lambdaD = 5.0;   // 原 20.0 -> 改 5.0 (允许方向微调)
    //const double lambdaM = 0.01;  // 原 5.0  -> 改 0.01 (几乎移除中点约束，仅防飞逸)

    // 默认的“松弛”权重 (用于长线段，允许优化)
    const double lambdaL_Loose = 0.5;   
    const double lambdaD_Loose = 5.0;   
    const double lambdaM_Loose = 0.5;

    // “强力”权重 (用于短线段，用于固定)
    const double lambda_Hard = 1000.0;
    
    for (MapLine* pML : lLocalMapLines)
    {
        if (!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if (it == mapLineVertexId.end()) continue;

        int idP1 = it->second.first;
        int idP2 = it->second.second;

        // ... 获取 id1, id2 ...
        //auto endpoints = pML->GetLineWorldPos();
        //Eigen::Vector3d d = endpoints.second.cast<double>() - endpoints.first.cast<double>();
        //double len_sq = d.squaredNorm();

        auto* vP1 = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idP1));
        auto* vP2 = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idP2));
        if (!vP1 || !vP2) continue;

        // initial geometry from current mapline
        auto endpoints_f = pML->GetLineWorldPos();
        Eigen::Vector3d P10 = endpoints_f.first.cast<double>();
        Eigen::Vector3d P20 = endpoints_f.second.cast<double>();
        Eigen::Vector3d d0  = P20 - P10;

        double L0 = d0.norm();
        if (!std::isfinite(L0) || L0 < 1e-6) continue;

        Eigen::Vector3d d0_unit = d0 / L0;
        Eigen::Vector3d m0 = 0.5 * (P10 + P20);

        double wL = lambdaL_Loose;
        double wD = lambdaD_Loose;
        double wM = lambdaM_Loose;

        // Tier 1: "Trash" / Noise (Length < 0.1m)
        // Behavior: LOCK IT DOWN. It's too small to be useful for optimizing pose, 
        // just keep it for visualization or map density.
        if (L0 < 0.1) 
        {
            wL = lambda_Hard; 
            wD = lambda_Hard; 
            wM = lambda_Hard; 
        }
        // Tier 2: "The Danger Zone" (0.1m <= Length < 0.5m) <--- YOUR PROBLEM AREA
        // Behavior: These are the lines on the wall. They are likely unstable.
        // We trust their Direction (mostly), but we DO NOT trust their Depth/Position sliding.
        // We need to constrain them heavily to their initial guess to prevent sliding.
        else if (L0 < 0.5) 
        {
            wL = 2.0;    // Prevent length changing too much
            wD = 20.0;   // Strong direction constraint (keep parallel)
            wM = 10.0;   // Strong midpoint constraint (STOP SLIDING!)
        }
        // Tier 3: "Stable Features" (Length >= 0.5m)
        // Behavior: Long lines (floor edges, door frames). 
        // We trust the image observation more. Let them adjust to minimize reprojection error.
        else 
        {
            wL = lambdaL_Loose;    // Allow length to breathe
            wD = lambdaD_Loose;    // Direction is usually stable due to length
            wM = lambdaM_Loose;    // Weak midpoint constraint (allow slight sliding to fit data)
        }

        // // ========================================================
        // // [关键修改]: 动态权重分配 (Soft Fix Logic)
        // // ========================================================
        // double wL = lambdaL_Loose;
        // double wD = lambdaD_Loose;
        // double wM = lambdaM_Loose;
        // // 如果线段短于 10cm (0.1m)，则认为它极不稳定，施加超强约束
        // if (L0 < 0.1) 
        // {
        //     wL = lambda_Hard; // 锁死长度
        //     wD = lambda_Hard; // 锁死方向
        //     wM = lambda_Hard; // 锁死位置 (核心)
        // }
        // if(L0 < 0.2 && L0 > 0.1)
        // {
        //     //wL = lambda_Hard;
        //     wD = lambdaD_Loose * 3;
        //     //wM = lambda_Hard;
        // }
        // // ========================================================

        // (A) length prior
        {
            auto* eLen = new EdgeLineLengthPrior(L0, wL);
            eLen->setVertex(0, vP1);
            eLen->setVertex(1, vP2);
            eLen->setInformation(Eigen::Matrix<double,1,1>::Identity());
            optimizer.addEdge(eLen);
            nEdges++;
        }

        // (B) direction prior
        {
            auto* eDir = new EdgeLineDirectionPrior(d0_unit, wD);
            eDir->setVertex(0, vP1);
            eDir->setVertex(1, vP2);
            eDir->setInformation(Eigen::Matrix3d::Identity());
            optimizer.addEdge(eDir);
            nEdges++;
        }

        // (C) midpoint prior
        {
            auto* eMid = new EdgeLineMidpointPrior(m0, wM);
            eMid->setVertex(0, vP1);
            eMid->setVertex(1, vP2);
            eMid->setInformation(Eigen::Matrix3d::Identity());
            optimizer.addEdge(eMid);
            nEdges++;
        }
    }

    // add edges using endpoint vertices
    int addedLineEdges = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        int idP1 = it->second.first;
        int idP2 = it->second.second;

        const auto& obs = pML->GetLineObservations();
        for(auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if(!pKFi || pKFi->isBad() || pKFi->GetMap()!=pCurrentMap) continue;
            int idxLine = get<0>(mit.second);
            if(idxLine < 0 || idxLine >= (int)pKFi->mvKeyLines.size()) continue;
            const cv::line_descriptor::KeyLine& kl = pKFi->mvKeyLines[idxLine];
            // build normalized 2D line: a u + b v + c = 0
            const double u1 = kl.startPointX, v1 = kl.startPointY;
            const double u2 = kl.endPointX,   v2 = kl.endPointY;
            const double dx = u2 - u1;
            const double dy = v2 - v1;
            const double na = dy;
            const double nb = -dx;
            const double nrm = std::sqrt(na*na + nb*nb);
            if (!std::isfinite(nrm) || nrm < 1e-12) continue;
            const double a = na / nrm;
            const double b = nb / nrm;
            const double c = -(a*u1 + b*v1);
            Eigen::Vector3d line_abc(a,b,c);

            // [策略二] 根据 2D 长度计算权重
            double length_weight = 1.0;
            if (nrm < min_pixel_len) {
                // 线性下降后平方，例如 20px -> 0.5 -> weight 0.25
                double ratio = nrm / min_pixel_len;
                length_weight = ratio * ratio; 
            }

            // invSigma2 guard
            int octave = kl.octave;
            if (octave < 0 || octave >= (int)pKFi->mvInvLevelSigma2.size()) octave = 0;
            const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
            if (!std::isfinite(invSigma2) || invSigma2 <= 0.0) continue;

            double final_info_val = invSigma2 * length_weight;

            auto* vPose = optimizer.vertex(pKFi->mnId);
            auto* vertex1 = optimizer.vertex(idP1);
            auto* vertex2 = optimizer.vertex(idP2);
            if(!vPose || !vertex1 || !vertex2) continue;
            // Edge for endpoint1
            {
                auto* e1 = new EdgeSE3ProjectPointToLine2D();
                e1->setVertex(0, vertex1);      // point
                e1->setVertex(1, vPose);   // pose
                e1->setMeasurement(line_abc);
                e1->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                //const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
                // 应该应用到 Information 矩阵：
                e1->setInformation(Eigen::Matrix<double,1,1>::Identity() * final_info_val);
                auto* rk = new g2o::RobustKernelHuber;
                //rk->setDelta(std::sqrt(3.84)); // 1D chi2 95% ~= 3.84
                rk->setDelta(3.0); // 宽松一点的 Huber
                e1->setRobustKernel(rk);
                optimizer.addEdge(e1);
                vpEdgesLineMono.push_back(e1);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }

            // Edge for endpoint2
            {
                auto* e2 = new EdgeSE3ProjectPointToLine2D();
                e2->setVertex(0, vertex2);      // point
                e2->setVertex(1, vPose);
                e2->setMeasurement(line_abc);
                e2->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                //const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
                e2->setInformation(Eigen::Matrix<double,1,1>::Identity() * final_info_val);

                auto* rk = new g2o::RobustKernelHuber;
                //rk->setDelta(std::sqrt(3.84));
                rk->setDelta(3.0); // 宽松一点的 Huber
                e2->setRobustKernel(rk);

                optimizer.addEdge(e2);
                vpEdgesLineMono.push_back(e2);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }
        }
    }

    std::cerr << "Added " << addedLineEdges << " (binary) line endpoint edges.\n";
    std::cout << "===== BA DEBUG =====" << std::endl;
    std::cout << "Vertices: " << optimizer.vertices().size() << std::endl;
    std::cout << "Edges: " << optimizer.edges().size() << std::endl;

    int active = 0;
    for (auto& it : optimizer.vertices())
    {
        if (!static_cast<g2o::OptimizableGraph::Vertex*>(it.second)->fixed())
            active++;
    }
    std::cout << "Active vertices: " << active << std::endl;

    if (optimizer.vertices().empty() || optimizer.edges().empty())
    {
        std::cout << "[WARN] Empty graph, skip BA" << std::endl;
        return;
    }

    // --- 9. run optimization (you can do robust iterations as desired) ---
    if (pbStopFlag && *pbStopFlag) return;
    optimizer.initializeOptimization();
    //std::cerr << "Starting optimization with " << nEdges << " edges." << std::endl;
    //std::cerr << "Line edges added: " << vpEdgesLineMono.size() << std::endl;
    optimizer.optimize(10);
    //std::cerr << "Optimization done." << std::endl;
    // --- 10. outlier detection (points) ---
    vector<pair<KeyFrame*,MapPoint*>> vToErasePoints;
    vToErasePoints.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());
    for (size_t i = 0; i < vpEdgesMono.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFMono[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesBody.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFBody[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesStereo.size(); ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if (!pMP) continue;
        if (e->chi2() > 7.815 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFStereo[i], pMP);
    }
    // --- 11. outlier detection (lines) ---
    vector<pair<KeyFrame*,MapLine*>> vToEraseLines;
    vToEraseLines.reserve(vpEdgesLineMono.size());
    for (size_t i = 0; i < vpEdgesLineMono.size(); ++i)
    {
        auto *e = vpEdgesLineMono[i];
        MapLine* pML = vpMapLineEdgeMono[i];
        if (!pML) continue;
        if (e->chi2() > 2*3.84)
            vToEraseLines.emplace_back(vpEdgeKFLineMono[i], pML);
    }
    // --- 12. apply erasures under map mutex ---
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        for (auto & pr : vToErasePoints)
        {
            KeyFrame* pKFi = pr.first;
            MapPoint* pMPi = pr.second;
            if (pKFi && pMPi)
            {
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        for (auto & pr : vToEraseLines)
        {
            KeyFrame* pKFi = pr.first;
            MapLine* pMLi = pr.second;
            if (pKFi && pMLi)
            {
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
    }
    // --- 13. write back optimized poses and points ---
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
        opr.addKeyFrame(pKFi);
    }
    opr.reserveMapPoints(lLocalMapPoints.size());
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if (!pMP->isRetrived()) { pMP->setRetrived(true); opr.addMapPoint(pMP); }
    }
    // MapLine endpoints are optimized jointly with pose and written back here.
    // If you later add VertexLineXYZ and make edges connect (vertexLine, vertexPose),
    // you would fetch and write optimized endpoints similar to MapPoints.
    opr.reserveMapLines(lLocalMapLines.size());
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        auto* vP1 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.first));
        auto* vP2 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.second));

        pML->SetLineWorldPos(
            vP1->estimate().cast<float>(),
            vP2->estimate().cast<float>()
        );
        // mark retrieved for reporting
        if (!pML->isRetrived()) 
        {
            // =========================================================
            // 【关键修改】: 在打包发送给 GS 之前，必须基于"新几何"重新采样！
            // =========================================================
            std::cerr << "Modify MapLine: " << pML->mnId << std::endl;
            // 参数建议放到类成员变量或配置中
            //float sample_step = 0.2f;  // 采样步长，比如 10cm
            //float view_weight = 2.0f; 
            //float sigma = 3.0f;
            //int top_k = 2;

            float sample_step = pKF->getLineSampleStep();
            float view_weight = pKF->getLineViewWeight();
            float sigma = pKF->getLineSigma();
            int top_k = pKF->getLineTopK();

            // 这一步会清空 pML 内部旧的 sampled points，并生成基于 new_p1, new_p2 的新点
            pML->SamplePointsAlongLine_MultiViewWeighted_Advanced(
                sample_step, view_weight, sigma, top_k);

            // 3. 将采样后的数据打包进 MappingOperation
            // (前提：MappingOperation::addMapLine 已经改为了读取 pML->GetLineSampledPoints3D())
            opr.addMapLine(pML);
            pML->setRetrived(true); 
            // 在线段上采样顶点，如果线段是 Combine 两个线段的情况下，怎么处理(后续再处理)
        }
    }   // end for each MapLine
    // keep mapline; mark retrieved for reporting
    pMap->IncreaseChangeIndex();
    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_edges = nEdges;
    num_Lines = lLocalMapLines.size();
}

#else

// 这里是非调试模式下的代码
// Local Bundle Adjustment with Line Support, and lines are optimized + Regularized terms
void Optimizer::LocalBundleAdjustmentWithLine_Optimization_Reg(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_edges,
    int& num_Lines,
    MappingOperation& opr)
{
    // --- 1. collect local keyframes (BFS) ---
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFi : vNeighKFs)
    {
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }
    // --- 2. collect local MapPoints ---
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        if (pKFi->mnId == pMap->GetInitKFid())
            num_fixedKF = 1;
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if (pMP->mnBALocalForKF != pKF->mnId)
                {
                    pMP->mnBALocalForKF = pKF->mnId;
                    lLocalMapPoints.push_back(pMP);
                }
            }
        }
    }
    // --- 3. collect fixed keyframes (that see local map points but are not local) ---
    list<KeyFrame*> lFixedCameras;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        const map<KeyFrame*, tuple<int,int>>& obs = pMP->GetObservations();
        for (auto & mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;
    if (num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // --- 4. setup optimizer ---
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
       solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    //optimizer.setVerbose(false);
    optimizer.setVerbose(true);
    if (pbStopFlag) optimizer.setForceStopFlag(pbStopFlag);
    unsigned long maxKFid = 0;
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();
    // --- 5. add local keyframe vertices (optimizable) ---
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    // --- 6. add fixed keyframe vertices ---
    for (KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > (int)maxKFid) maxKFid = pKFi->mnId;
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }
    // --- 7. add MapPoint vertices + edges (same as original) ---
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    std::vector<KeyFrame*> vpEdgeKFMono, vpEdgeKFBody, vpEdgeKFStereo;
    std::vector<MapPoint*> vpMapPointEdgeMono, vpMapPointEdgeBody, vpMapPointEdgeStereo;
    vpEdgesMono.reserve(1000);
    vpEdgesBody.reserve(1000);
    vpEdgesStereo.reserve(1000);
    int nEdges = 0;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*, tuple<int,int>>& observations = pMP->GetObservations();
        for (auto & mit : observations)
        {
            KeyFrame* pKFi = mit.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap) continue;
            const int leftIndex = get<0>(mit.second);
            if (leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
            {
                // mono
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,2,1> obs; obs << kpUn.pt.x, kpUn.pt.y;
                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                e->pCamera = pKFi->mpCamera;
                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
                nEdges++;
            }
            else if (leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0)
            {
                // stereo
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                Eigen::Matrix<double,3,1> obs; obs << kpUn.pt.x, kpUn.pt.y, pKFi->mvuRight[leftIndex];
                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberStereo); e->setRobustKernel(rk);
                e->fx = pKFi->fx; e->fy = pKFi->fy; e->cx = pKFi->cx; e->cy = pKFi->cy; e->bf = pKFi->mbf;
                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpMapPointEdgeStereo.push_back(pMP);
                nEdges++;
            }
            // body / right camera observation for systems with mpCamera2
            if (pKFi->mpCamera2)
            {
                int rightIndex = get<1>(mit.second);
                if (rightIndex != -1)
                {
                    rightIndex -= pKFi->NLeft;
                    cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                    Eigen::Matrix<double,2,1> obs; obs << kp.pt.x, kp.pt.y;
                    ORB_SLAM3::EdgeSE3ProjectXYZToBody * e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber; rk->setDelta(thHuberMono); e->setRobustKernel(rk);
                    Sophus::SE3f Trl = pKFi->GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());
                    e->pCamera = pKFi->mpCamera2;
                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKFi);
                    vpMapPointEdgeBody.push_back(pMP);
                    nEdges++;
                }
            }
        }
    }
    // --- 8. add MapLine edges (Point-to-Line), using your Edge classes ---
    // collect local maplines observed by local keyframes
    list<MapLine*> lLocalMapLines;
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        vector<MapLine*> vpLines = pKFi->GetMapLineMatches();
        for (MapLine* pML : vpLines)
        {
            if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if (pML->mnBALocalForKF != pKF->mnId)
                {
                    pML->mnBALocalForKF = pKF->mnId;
                    lLocalMapLines.push_back(pML);
                }
            }
        }
    }
    // containers for line edges
    std::vector<ORB_SLAM3::EdgeSE3ProjectPointToLine2D*> vpEdgesLineMono;  // pose + line points
    std::vector<KeyFrame*> vpEdgeKFLineMono;
    std::vector<MapLine*> vpMapLineEdgeMono;
    const float deltaLineMono = sqrt(9.488);
    const float deltaLineStereo = sqrt(11.345);

    // Compute a safe offset for line vertex ids so they do not collide with point ids used above
    int nextVertexId = (int)maxKFid + 1;
    for (MapPoint* pMP : lLocalMapPoints)
    {
        int pid = (int)pMP->mnId + (int)maxKFid + 2;
        if (pid > nextVertexId) nextVertexId = pid;
    }

    // --------- NEW: add MapLine endpoint vertices ----------
    unordered_map<MapLine*, pair<int,int>> mapLineVertexId;
    mapLineVertexId.reserve(lLocalMapLines.size() * 2);

    // 阈值设置：比如小于 0.1米 (10cm) 或者像素长度小于 20px
    const double min_3d_length_sq = 0.1 * 0.1;
    //const double min_2d_length_sq = 20.0 * 20.0;
    const double min_pixel_len = 40.0;         // 2D 像素长度阈值 (用于降权)

    int numLineVertices = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if (!pML || pML->isBad()) continue;

        auto endpoints = pML->GetLineWorldPos();
        Eigen::Vector3d Xw1 = endpoints.first.cast<double>();
        Eigen::Vector3d Xw2 = endpoints.second.cast<double>();

        Eigen::Vector3d d = Xw2 - Xw1;
        if (!d.allFinite() || d.squaredNorm() < 1e-12) continue;

        double len_sq = d.squaredNorm();
        // [策略一] 判断是否是短线段
        bool bIsShortLine = (len_sq < min_3d_length_sq);

        // endpoint 1
        auto* vP1 = new g2o::VertexSBAPointXYZ();
        int id1 = ++nextVertexId;
        vP1->setId(id1);
        vP1->setEstimate(Xw1);
        vP1->setFixed(true);
        vP1->setMarginalized(true);

        // endpoint 2
        auto* vP2 = new g2o::VertexSBAPointXYZ();
        int id2 = ++nextVertexId;
        vP2->setId(id2);
        vP2->setEstimate(Xw2);
        vP2->setFixed(false);
        vP2->setMarginalized(true);

        // // [关键修改] 设置 Fixed 属性
        // if (bIsShortLine) 
        // {
        //     // 如果是短线段，直接锁死，不让优化器动它 (防止乱飞)
        //     vP1->setFixed(true);
        //     vP2->setFixed(true);
        // }
        // else 
        // {
        //     // 如果是正常线段，两个端点都允许移动 (你原来锁了vP1，会导致无法平移)
        //     vP1->setFixed(false);
        //     vP2->setFixed(false);
        // }

        bool ok1 = optimizer.addVertex(vP1);
        bool ok2 = optimizer.addVertex(vP2);

        if (!ok1 || !ok2)
        {
            if (ok1) optimizer.removeVertex(vP1);
            delete vP1;
            delete vP2;
            continue;
        }

        mapLineVertexId[pML] = {id1, id2};
        numLineVertices += 2;
    }

    // // --------- add Line Length Prior edges (ONCE per MapLine) ----------
    ///const double lambdaL = 50.0;   // 1~5
    ///const double lambdaD = 20.0;   // 0.5~2
    ///const double lambdaM = 5.0;   // 0.05~0.5  (不要太大，防止锁死)
    // // --------- add Line Length Prior edges (ONCE per MapLine) ----------
    // [修改] 大幅降低权重，让观测数据说话
    //const double lambdaL = 0.5;   // 原 50.0 -> 改 0.5 (允许长度变化)
    //const double lambdaD = 5.0;   // 原 20.0 -> 改 5.0 (允许方向微调)
    //const double lambdaM = 0.01;  // 原 5.0  -> 改 0.01 (几乎移除中点约束，仅防飞逸)

    // 默认的“松弛”权重 (用于长线段，允许优化)
    const double lambdaL_Loose = 0.5;   
    const double lambdaD_Loose = 5.0;   
    const double lambdaM_Loose = 0.5;

    // “强力”权重 (用于短线段，用于固定)
    const double lambda_Hard = 1000.0;

    
    for (MapLine* pML : lLocalMapLines)
    {
        if (!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if (it == mapLineVertexId.end()) continue;

        int idP1 = it->second.first;
        int idP2 = it->second.second;

        // ... 获取 id1, id2 ...
        //auto endpoints = pML->GetLineWorldPos();
        //Eigen::Vector3d d = endpoints.second.cast<double>() - endpoints.first.cast<double>();
        //double len_sq = d.squaredNorm();

        auto* vP1 = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idP1));
        auto* vP2 = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idP2));
        if (!vP1 || !vP2) continue;

        // initial geometry from current mapline
        auto endpoints_f = pML->GetLineWorldPos();
        Eigen::Vector3d P10 = endpoints_f.first.cast<double>();
        Eigen::Vector3d P20 = endpoints_f.second.cast<double>();
        Eigen::Vector3d d0  = P20 - P10;

        double L0 = d0.norm();
        if (!std::isfinite(L0) || L0 < 1e-6) continue;

        Eigen::Vector3d d0_unit = d0 / L0;
        Eigen::Vector3d m0 = 0.5 * (P10 + P20);

        double wL = lambdaL_Loose;
        double wD = lambdaD_Loose;
        double wM = lambdaM_Loose;

        // Tier 1: "Trash" / Noise (Length < 0.1m)
        // Behavior: LOCK IT DOWN. It's too small to be useful for optimizing pose, 
        // just keep it for visualization or map density.
        if (L0 < 0.1) 
        {
            wL = lambda_Hard; 
            wD = lambda_Hard; 
            wM = lambda_Hard; 
        }
        // Tier 2: "The Danger Zone" (0.1m <= Length < 0.5m) <--- YOUR PROBLEM AREA
        // Behavior: These are the lines on the wall. They are likely unstable.
        // We trust their Direction (mostly), but we DO NOT trust their Depth/Position sliding.
        // We need to constrain them heavily to their initial guess to prevent sliding.
        else if (L0 < 0.5) 
        {
            wL = 2.0;    // Prevent length changing too much
            wD = 20.0;   // Strong direction constraint (keep parallel)
            wM = 10.0;   // Strong midpoint constraint (STOP SLIDING!)
        }
        // Tier 3: "Stable Features" (Length >= 0.5m)
        // Behavior: Long lines (floor edges, door frames). 
        // We trust the image observation more. Let them adjust to minimize reprojection error.
        else 
        {
            wL = lambdaL_Loose;    // Allow length to breathe
            wD = lambdaD_Loose;    // Direction is usually stable due to length
            wM = lambdaM_Loose;    // Weak midpoint constraint (allow slight sliding to fit data)
        }

        // // ========================================================
        // // [关键修改]: 动态权重分配 (Soft Fix Logic)
        // // ========================================================
        // double wL = lambdaL_Loose;
        // double wD = lambdaD_Loose;
        // double wM = lambdaM_Loose;
        // // 如果线段短于 10cm (0.1m)，则认为它极不稳定，施加超强约束
        // if (L0 < 0.1) 
        // {
        //     wL = lambda_Hard; // 锁死长度
        //     wD = lambda_Hard; // 锁死方向
        //     wM = lambda_Hard; // 锁死位置 (核心)
        // }
        // if(L0 < 0.2 && L0 > 0.1)
        // {
        //     //wL = lambda_Hard;
        //     wD = lambdaD_Loose * 3;
        //     //wM = lambda_Hard;
        // }
        // // ========================================================

        // (A) length prior
        {
            auto* eLen = new EdgeLineLengthPrior(L0, wL);
            eLen->setVertex(0, vP1);
            eLen->setVertex(1, vP2);
            eLen->setInformation(Eigen::Matrix<double,1,1>::Identity());
            optimizer.addEdge(eLen);
            nEdges++;
        }

        // (B) direction prior
        {
            auto* eDir = new EdgeLineDirectionPrior(d0_unit, wD);
            eDir->setVertex(0, vP1);
            eDir->setVertex(1, vP2);
            eDir->setInformation(Eigen::Matrix3d::Identity());
            optimizer.addEdge(eDir);
            nEdges++;
        }

        // (C) midpoint prior
        {
            auto* eMid = new EdgeLineMidpointPrior(m0, wM);
            eMid->setVertex(0, vP1);
            eMid->setVertex(1, vP2);
            eMid->setInformation(Eigen::Matrix3d::Identity());
            optimizer.addEdge(eMid);
            nEdges++;
        }
    }

    // add edges using endpoint vertices
    int addedLineEdges = 0;
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        int idP1 = it->second.first;
        int idP2 = it->second.second;

        const auto& obs = pML->GetLineObservations();
        for(auto& mit : obs)
        {
            KeyFrame* pKFi = mit.first;
            if(!pKFi || pKFi->isBad() || pKFi->GetMap()!=pCurrentMap) continue;
            int idxLine = get<0>(mit.second);
            if(idxLine < 0 || idxLine >= (int)pKFi->mvKeyLines.size()) continue;
            const cv::line_descriptor::KeyLine& kl = pKFi->mvKeyLines[idxLine];
            // build normalized 2D line: a u + b v + c = 0
            const double u1 = kl.startPointX, v1 = kl.startPointY;
            const double u2 = kl.endPointX,   v2 = kl.endPointY;
            const double dx = u2 - u1;
            const double dy = v2 - v1;
            const double na = dy;
            const double nb = -dx;
            const double nrm = std::sqrt(na*na + nb*nb);
            if (!std::isfinite(nrm) || nrm < 1e-12) continue;
            const double a = na / nrm;
            const double b = nb / nrm;
            const double c = -(a*u1 + b*v1);
            Eigen::Vector3d line_abc(a,b,c);

            // [策略二] 根据 2D 长度计算权重
            double length_weight = 1.0;
            if (nrm < min_pixel_len) {
                // 线性下降后平方，例如 20px -> 0.5 -> weight 0.25
                double ratio = nrm / min_pixel_len;
                length_weight = ratio * ratio; 
            }

            // invSigma2 guard
            int octave = kl.octave;
            if (octave < 0 || octave >= (int)pKFi->mvInvLevelSigma2.size()) octave = 0;
            const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
            if (!std::isfinite(invSigma2) || invSigma2 <= 0.0) continue;

            double final_info_val = invSigma2 * length_weight;

            auto* vPose = optimizer.vertex(pKFi->mnId);
            auto* vertex1 = optimizer.vertex(idP1);
            auto* vertex2 = optimizer.vertex(idP2);
            if(!vPose || !vertex1 || !vertex2) continue;
            // Edge for endpoint1
            {
                auto* e1 = new EdgeSE3ProjectPointToLine2D();
                e1->setVertex(0, vertex1);      // point
                e1->setVertex(1, vPose);   // pose
                e1->setMeasurement(line_abc);
                e1->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                //const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
                // 应该应用到 Information 矩阵：
                e1->setInformation(Eigen::Matrix<double,1,1>::Identity() * final_info_val);
                auto* rk = new g2o::RobustKernelHuber;
                //rk->setDelta(std::sqrt(3.84)); // 1D chi2 95% ~= 3.84
                rk->setDelta(3.0); // 宽松一点的 Huber
                e1->setRobustKernel(rk);
                optimizer.addEdge(e1);
                vpEdgesLineMono.push_back(e1);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }

            // Edge for endpoint2
            {
                auto* e2 = new EdgeSE3ProjectPointToLine2D();
                e2->setVertex(0, vertex2);      // point
                e2->setVertex(1, vPose);
                e2->setMeasurement(line_abc);
                e2->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
                //const double invSigma2 = (double)pKFi->mvInvLevelSigma2[octave];
                e2->setInformation(Eigen::Matrix<double,1,1>::Identity() * final_info_val);

                auto* rk = new g2o::RobustKernelHuber;
                //rk->setDelta(std::sqrt(3.84));
                rk->setDelta(3.0); // 宽松一点的 Huber
                e2->setRobustKernel(rk);

                optimizer.addEdge(e2);
                vpEdgesLineMono.push_back(e2);
                vpEdgeKFLineMono.push_back(pKFi);
                vpMapLineEdgeMono.push_back(pML);
                addedLineEdges++;
                nEdges++;
            }
        }
    }

    //std::cerr << "Added " << addedLineEdges << " (binary) line endpoint edges.\n";
    //std::cout << "===== BA DEBUG =====" << std::endl;
    //std::cout << "Vertices: " << optimizer.vertices().size() << std::endl;
    //std::cout << "Edges: " << optimizer.edges().size() << std::endl;

    //int active = 0;
    //for (auto& it : optimizer.vertices())
    //{
    //    if (!static_cast<g2o::OptimizableGraph::Vertex*>(it.second)->fixed())
    //        active++;
    //}
    //std::cout << "Active vertices: " << active << std::endl;

    if (optimizer.vertices().empty() || optimizer.edges().empty())
    {
        std::cout << "[WARN] Empty graph, skip BA" << std::endl;
        return;
    }

    // --- 9. run optimization (you can do robust iterations as desired) ---
    if (pbStopFlag && *pbStopFlag) return;
    optimizer.initializeOptimization();
    //std::cerr << "Starting optimization with " << nEdges << " edges." << std::endl;
    //std::cerr << "Line edges added: " << vpEdgesLineMono.size() << std::endl;
    optimizer.optimize(10);
    //std::cerr << "Optimization done." << std::endl;
    // --- 10. outlier detection (points) ---
    vector<pair<KeyFrame*,MapPoint*>> vToErasePoints;
    vToErasePoints.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());
    for (size_t i = 0; i < vpEdgesMono.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFMono[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesBody.size(); ++i)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];
        if (!pMP) continue;
        if (e->chi2() > 5.991 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFBody[i], pMP);
    }
    for (size_t i = 0; i < vpEdgesStereo.size(); ++i)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        if (!pMP) continue;
        if (e->chi2() > 7.815 || !e->isDepthPositive())
            vToErasePoints.emplace_back(vpEdgeKFStereo[i], pMP);
    }
    // --- 11. outlier detection (lines) ---
    vector<pair<KeyFrame*,MapLine*>> vToEraseLines;
    vToEraseLines.reserve(vpEdgesLineMono.size());
    for (size_t i = 0; i < vpEdgesLineMono.size(); ++i)
    {
        auto *e = vpEdgesLineMono[i];
        MapLine* pML = vpMapLineEdgeMono[i];
        if (!pML) continue;
        if (e->chi2() > 2*3.84)
            vToEraseLines.emplace_back(vpEdgeKFLineMono[i], pML);
    }
    // --- 12. apply erasures under map mutex ---
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        for (auto & pr : vToErasePoints)
        {
            KeyFrame* pKFi = pr.first;
            MapPoint* pMPi = pr.second;
            if (pKFi && pMPi)
            {
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        for (auto & pr : vToEraseLines)
        {
            KeyFrame* pKFi = pr.first;
            MapLine* pMLi = pr.second;
            if (pKFi && pMLi)
            {
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
    }
    // --- 13. write back optimized poses and points ---
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for (KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);
        opr.addKeyFrame(pKFi);
    }
    opr.reserveMapPoints(lLocalMapPoints.size());
    for (MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if (!pMP->isRetrived()) { pMP->setRetrived(true); opr.addMapPoint(pMP); }
    }
    // MapLine endpoints are optimized jointly with pose and written back here.
    // If you later add VertexLineXYZ and make edges connect (vertexLine, vertexPose),
    // you would fetch and write optimized endpoints similar to MapPoints.
    opr.reserveMapLines(lLocalMapLines.size());
    for (MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;

        auto it = mapLineVertexId.find(pML);
        if(it == mapLineVertexId.end()) continue;

        auto* vP1 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.first));
        auto* vP2 =
            static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(it->second.second));

        pML->SetLineWorldPos(
            vP1->estimate().cast<float>(),
            vP2->estimate().cast<float>()
        );
        // mark retrieved for reporting
        if (!pML->isRetrived()) 
        {
            // =========================================================
            // 【关键修改】: 在打包发送给 GS 之前，必须基于"新几何"重新采样！
            // =========================================================
            std::cerr << "Modify MapLine: " << pML->mnId << std::endl;
            // 参数建议放到类成员变量或配置中
            //float sample_step = 0.2f;  // 采样步长，比如 10cm
            //float view_weight = 2.0f; 
            //float sigma = 3.0f;
            //int top_k = 2;

            float sample_step = pKF->getLineSampleStep();
            float view_weight = pKF->getLineViewWeight();
            float sigma = pKF->getLineSigma();
            int top_k = pKF->getLineTopK();

            // 这一步会清空 pML 内部旧的 sampled points，并生成基于 new_p1, new_p2 的新点
            pML->SamplePointsAlongLine_MultiViewWeighted_Advanced(
                sample_step, view_weight, sigma, top_k);

            // 3. 将采样后的数据打包进 MappingOperation
            // (前提：MappingOperation::addMapLine 已经改为了读取 pML->GetLineSampledPoints3D())
            opr.addMapLine(pML);
            pML->setRetrived(true); 
            // 在线段上采样顶点，如果线段是 Combine 两个线段的情况下，怎么处理(后续再处理)
        }
    }   // end for each MapLine
    // keep mapline; mark retrieved for reporting
    pMap->IncreaseChangeIndex();
    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_edges = nEdges;
    num_Lines = lLocalMapLines.size();
}

#endif



/*
 * 现在的写法 if (!pML->isRetrived()) 会导致旧的线段永远不会被更新。（这个非常重要改进，后续测试一下）

 一旦一个线段在某次 LBA 中被发送给了 Gaussian Mapper（设置了 isRetrived(true)），即使后续的 LBA 把它优化得更准、更长，它也无法再次进入 if 内部，导致 Gaussian Mapper 永远拿不到它最新的几何信息。
为什么不能简单删掉这个 if？
如果你直接删掉 if (!pML->isRetrived())，让所有局部线段每次 LBA 都发送：
    后果：LBA 运行频率很高（每秒几次），你会对同一条线段疯狂重复采样，导致 GaussianMapper 里的点数爆炸式增长，显存瞬间撑爆。
解决方案：基于“几何变化”的条件更新

我们需要一个机制：只有当线段发生显著变化（比如变长了、位置大改了）时，才再次发送，否则保持现状。
 // ... Inside loop over lLocalMapLines ...
    // 新增：记录上次发送给 GS 时的长度
    float mLastSentLength = 0.0f;

    pML->SetLineWorldPos(vP1->estimate().cast<float>(), vP2->estimate().cast<float>());
    
    // 计算当前新长度
    float currentLen = (vP1->estimate() - vP2->estimate()).norm();

    // 逻辑：
    // 1. 如果从未发送过 (!isRetrived) -> 发送
    // 2. 如果发送过，但长度变化超过一定阈值 (例如 10% 或 0.1m) -> 发送补充点
    //    注意：如果是变短了，不需要发送（旧的高斯球会被 Prune 掉）
    //    主要是处理变长（Merge 或 延伸）的情况
    bool needUpdate = false;
    
    if (!pML->isRetrived()) {
        needUpdate = true;
    } 
    else {
        // 如果长度增加了 20% 或者 超过 0.2m，认为发生了显著变化（例如合并）
        if (currentLen > pML->mLastSentLength * 1.2f || (currentLen - pML->mLastSentLength) > 0.2f) {
            needUpdate = true;
        }
    }

    if (needUpdate) 
    { 
        pML->setRetrived(true); 
        pML->mLastSentLength = currentLen; // 更新记录

        // =========================================================
        // 采样并打包
        // =========================================================
        float sample_step = 0.1f;
        float view_weight = 2.0f; 
        float sigma = 3.0f;
        int top_k = 3;

        // 【关键】：这里会清空 pML 内部的 buffer 并重新采样
        // 即使是旧线段，采样也是基于“最新”的端点进行的
        pML->SamplePointsAlongLine_MultiViewWeighted_Advanced(
            sample_step, view_weight, sigma, top_k);

        // 打包发送
        // 这里的 opr.addMapLine 会读取最新的采样点
        // GS 线程收到后，会把这些点作为"新点"加入显存
        // 旧位置的点因为不再被刷新，后续会被 densifyAndPrune 中的 Prune 逻辑干掉
        opr.addMapLine(pML); 
    }
 */

void Optimizer::TestEdgeSE3ProjectPointToLine2D_Jacobian()
{
    using namespace g2o;

    std::cout << "\n============================================\n";
    std::cout << "Jacobian Check: EdgeSE3ProjectPointToLine2D \n";
    std::cout << "  (Rotation cols 0-2, Translation cols 3-5)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // 1. Setup Data
    VertexSE3Expmap vPose;
    // Use a non-identity pose to properly test rotation effects
    SE3Quat T0(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d(1,0,0))), 
        Eigen::Vector3d(0.1, -0.2, 0.3)
    );
    vPose.setEstimate(T0);

    Eigen::Vector3d Xw(1.2, 0.5, 4.0);

    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;

    // Normalized Line (a, b, c)
    double u1 = 100, v1 = 120, u2 = 420, v2 = 260;
    double dx = u2 - u1, dy = v2 - v1;
    double n = std::sqrt(dx*dx + dy*dy);
    double a = dy / n, b = -dx / n, c = -(a*u1 + b*v1);

    // Helpers
    auto project = [&](const Eigen::Vector3d& Xc){
        return Eigen::Vector2d(fx * Xc(0) / Xc(2) + cx, fy * Xc(1) / Xc(2) + cy);
    };

    auto compute_error = [&](const SE3Quat& T, const Eigen::Vector3d& P){
        Eigen::Vector3d Xc = T.map(P);
        Eigen::Vector2d uv = project(Xc);
        return a * uv(0) + b * uv(1) + c;
    };

    // ============================================================
    // 7. Analytic Jacobian
    // ============================================================
    Eigen::Matrix<double,1,3> Ji_ana;
    Eigen::Matrix<double,1,6> Jj_ana;

    {
        Eigen::Vector3d Xc = T0.map(Xw);
        Eigen::Matrix3d R = T0.rotation().toRotationMatrix();
        double x = Xc(0), y = Xc(1), z = Xc(2);

        // Projection Jacobian (1x3)
        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z, 0, -fx*x/(z*z), 0, fy/z, -fy*y/(z*z);
        Eigen::RowVector2d ab(a,b);
        Eigen::RowVector3d Jimg = ab * Jpi;

        // Ji: w.r.t Point (1x3) = Jimg * R
        Ji_ana = Jimg * R;

        // Jj: w.r.t Pose (1x6)
        // Order based on YOUR SE3Quat.h: [ Rotation (0-2) | Translation (3-5) ]
        Eigen::Matrix<double,3,6> Jse3;
        Jse3.setZero();

        // Cols 0-2: Rotation -> -[Xc]^
        Jse3(0,1) =  z; Jse3(0,2) = -y;
        Jse3(1,0) = -z; Jse3(1,2) =  x;
        Jse3(2,0) =  y; Jse3(2,1) = -x;

        // Cols 3-5: Translation -> Identity
        Jse3.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        Jj_ana = Jimg * Jse3;
    }

    // ============================================================
    // 8. Numerical Jacobian
    // ============================================================
    Eigen::Matrix<double,1,3> Ji_num;
    Eigen::Matrix<double,1,6> Jj_num;

    // ---- A. w.r.t Point ----
    for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        dp(k) = eps;
        double rp = compute_error(T0, Xw + dp);
        double rm = compute_error(T0, Xw - dp);
        Ji_num(0,k) = (rp - rm) / (2*eps);
    }

    // ---- B. w.r.t Pose (Left Update) ----
    for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<double,6,1> delta = Eigen::Matrix<double,6,1>::Zero();
        delta(k) = eps;

        VertexSE3Expmap vp; vp.setEstimate(T0); vp.oplus(delta.data());
        double rp = compute_error(vp.estimate(), Xw);

        delta(k) = -eps;
        VertexSE3Expmap vm; vm.setEstimate(T0); vm.oplus(delta.data());
        double rm = compute_error(vm.estimate(), Xw);

        Jj_num(0,k) = (rp - rm) / (2*eps);
    }

    // ============================================================
    // 9. Report
    // ============================================================
    std::cout << "Analytic Jj: " << Jj_ana << "\n";
    std::cout << "Numeric  Jj: " << Jj_num << "\n";
    std::cout << "Diff     Jj: " << (Jj_ana - Jj_num).norm() << "\n";

    if ((Jj_ana - Jj_num).norm() < 1e-4)
        std::cout << "[SUCCESS] Jacobian Verified!\n";
    else
        std::cout << "[FAILURE] Jacobian Mismatch!\n";
}

 #if 0
void Optimizer::TestEdgeSE3ProjectPointToLine2D_Jacobian()
{
    using namespace g2o;

    std::cout << "\n============================================\n";
    std::cout << "Jacobian Check: EdgeSE3ProjectPointToLine2D (SAFE)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // ----------------------------
    // 1. 构造 pose
    // ----------------------------
    VertexSE3Expmap vPose;
    vPose.setEstimate(SE3Quat(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(0.1, -0.2, 0.3)
    ));
    SE3Quat T0 = vPose.estimate();

    // ----------------------------
    // 2. 构造 point
    // ----------------------------
    Eigen::Vector3d Xw(1.2, 0.5, 4.0);

    // ----------------------------
    // 3. 相机内参
    // ----------------------------
    const double fx = 500.0;
    const double fy = 500.0;
    const double cx = 320.0;
    const double cy = 240.0;

    // ----------------------------
    // 4. 构造观测直线 ax + by + c = 0（单位化）
    // ----------------------------
    double a, b, c;
    {
        double u1 = 100, v1 = 120;
        double u2 = 420, v2 = 260;
        double dx = u2 - u1;
        double dy = v2 - v1;
        double n = std::sqrt(dx*dx + dy*dy);
        a =  dy / n;
        b = -dx / n;
        c = -(a*u1 + b*v1);
    }

    // ----------------------------
    // 5. 投影函数
    // ----------------------------
    auto project = [&](const Eigen::Vector3d& Xc){
        return Eigen::Vector2d(
            fx * Xc(0) / Xc(2) + cx,
            fy * Xc(1) / Xc(2) + cy
        );
    };

    auto projectJac = [&](const Eigen::Vector3d& Xc){
        Eigen::Matrix<double,2,3> J;
        double x = Xc(0), y = Xc(1), z = Xc(2), z2 = z*z;
        J << fx/z,   0,    -fx*x/z2,
              0,    fy/z, -fy*y/z2;
        return J;
    };

    // ----------------------------
    // 6. 误差函数 r = a*u + b*v + c
    // ----------------------------
    auto compute_error = [&](const SE3Quat& T){
        Eigen::Vector3d Xc = T.map(Xw);
        Eigen::Vector2d uv = project(Xc);
        return a * uv(0) + b * uv(1) + c;
    };

    // ============================================================
    // 7. 解析 Jacobian
    // ============================================================
    Eigen::Matrix<double,1,3> Ji_ana;
    Eigen::Matrix<double,1,6> Jj_ana;

    {
        Eigen::Vector3d Xc = T0.map(Xw);
        Eigen::Matrix3d R = T0.rotation().toRotationMatrix();

        Eigen::Matrix<double,2,3> Jpi = projectJac(Xc);
        Eigen::RowVector2d ab(a,b);
        Eigen::RowVector3d Jimg = ab * Jpi;   // dr/dXc

        // wrt point
        Ji_ana = Jimg * R;

        // wrt pose: dXc/dxi = [-skew(Xc) | I]
        Eigen::Matrix<double,3,6> Jse3;
        Jse3.setZero();
        Jse3(0,1)= Xc(2);  Jse3(0,2)= -Xc(1);
        Jse3(1,0)= -Xc(2); Jse3(1,2)=  Xc(0);
        Jse3(2,0)= Xc(1);  Jse3(2,1)= -Xc(0);
        Jse3.block<3,3>(0,3).setIdentity();

        Jj_ana = Jimg * Jse3;
    }

    // ============================================================
    // 8. 数值 Jacobian
    // ============================================================
    Eigen::Matrix<double,1,3> Ji_num;
    Eigen::Matrix<double,1,6> Jj_num;

    // ---- wrt point ----
    for (int k = 0; k < 3; ++k)
    {
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        dp(k) = eps;

        double rp = compute_error(T0);
        double rm = compute_error(T0);

        rp = compute_error(SE3Quat(T0.rotation(), T0.translation()));
        rm = compute_error(SE3Quat(T0.rotation(), T0.translation()));

        rp = compute_error(SE3Quat(T0.rotation(), T0.translation()));
        rm = compute_error(SE3Quat(T0.rotation(), T0.translation()));

        rp = compute_error(SE3Quat(T0.rotation(), T0.translation()));

        // 手动 perturb point
        Eigen::Vector3d Xp = Xw + dp;
        Eigen::Vector3d Xm = Xw - dp;

        rp = a * project(T0.map(Xp))(0) + b * project(T0.map(Xp))(1) + c;
        rm = a * project(T0.map(Xm))(0) + b * project(T0.map(Xm))(1) + c;

        Ji_num(0,k) = (rp - rm) / (2*eps);
    }

    // ---- wrt pose ----
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Matrix<double,6,1> dx = Eigen::Matrix<double,6,1>::Zero();
        dx(k) = eps;

        VertexSE3Expmap vp, vm;
        vp.setEstimate(T0);
        vm.setEstimate(T0);

        vp.oplus(dx.data());
        dx(k) = -eps;
        vm.oplus(dx.data());

        double rp = compute_error(vp.estimate());
        double rm = compute_error(vm.estimate());

        Jj_num(0,k) = (rp - rm) / (2*eps);
    }

    // ============================================================
    // 9. 输出结果
    // ============================================================
    std::cout << "\nAnalytic Ji:\n" << Ji_ana << "\n";
    std::cout << "Numeric  Ji:\n" << Ji_num << "\n";
    std::cout << "Diff     Ji:\n" << (Ji_ana - Ji_num) << "\n";

    std::cout << "\nAnalytic Jj:\n" << Jj_ana << "\n";
    std::cout << "Numeric  Jj:\n" << Jj_num << "\n";
    std::cout << "Diff     Jj:\n" << (Jj_ana - Jj_num) << "\n";

    std::cout << "\nMax |diff| = "
              << std::max(
                    (Ji_ana - Ji_num).cwiseAbs().maxCoeff(),
                    (Jj_ana - Jj_num).cwiseAbs().maxCoeff()
                 )
              << "\n";

    std::cout << "====== Jacobian Check Done ======\n";
}

#endif

void Optimizer::TestEdgeLineLengthPrior_Jacobian_SAFE()
{
    std::cout << "\n============================================\n";
    std::cout << "Jacobian Check: Line Length Prior (SAFE)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // ---- test data ----
    Eigen::Vector3d p1(1.0, 0.2, 3.5);
    Eigen::Vector3d p2(1.4, 0.1, 3.7);

    const double lambdaL = 2.0;

    // L0 用初始长度（也可以给固定值）
    const double L0 = (p2 - p1).norm();

    auto compute_r = [&](const Eigen::Vector3d& _p1, const Eigen::Vector3d& _p2)->double{
        Eigen::Vector3d s = _p2 - _p1;
        double l = s.norm();
        if(!std::isfinite(l) || l < 1e-12) return 1e6;
        return lambdaL * (l - L0);
    };

    // ---- analytic jacobians ----
    Eigen::RowVector3d Jp1_ana, Jp2_ana;
    {
        Eigen::Vector3d s = p2 - p1;
        double l = s.norm();
        Eigen::Vector3d shat = s / l;
        Jp2_ana = lambdaL * shat.transpose();
        Jp1_ana = -Jp2_ana;
    }

    // ---- numeric jacobians ----
    Eigen::RowVector3d Jp1_num, Jp2_num;

    for(int k=0;k<3;k++)
    {
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        dp(k) = eps;

        // p1
        double rp = compute_r(p1 + dp, p2);
        double rm = compute_r(p1 - dp, p2);
        Jp1_num(k) = (rp - rm) / (2*eps);

        // p2
        rp = compute_r(p1, p2 + dp);
        rm = compute_r(p1, p2 - dp);
        Jp2_num(k) = (rp - rm) / (2*eps);
    }

    std::cout << "\nJp1 analytic:\n" << Jp1_ana << "\n";
    std::cout << "Jp1 numeric :\n" << Jp1_num << "\n";
    std::cout << "diff        :\n" << (Jp1_ana - Jp1_num) << "\n";

    std::cout << "\nJp2 analytic:\n" << Jp2_ana << "\n";
    std::cout << "Jp2 numeric :\n" << Jp2_num << "\n";
    std::cout << "diff        :\n" << (Jp2_ana - Jp2_num) << "\n";

    double maxDiff = std::max((Jp1_ana - Jp1_num).cwiseAbs().maxCoeff(),
                             (Jp2_ana - Jp2_num).cwiseAbs().maxCoeff());

    std::cout << "\nMax |diff| = " << maxDiff << "\n";
    std::cout << "------ end test ------\n";
}

void Optimizer::TestEdgeLineDirectionPrior_Jacobian_SAFE()
{
    std::cout << "\n============================================\n";
    std::cout << "Jacobian Check: Line Direction Prior (SAFE, 3D residual)\n";
    std::cout << "============================================\n";

    constexpr double eps = 1e-6;

    // ---- test data ----
    Eigen::Vector3d p1(1.0, 0.2, 3.5);
    Eigen::Vector3d p2(1.4, 0.1, 3.7);

    const double lambdaD = 2.0;

    // target direction (must be unit)
    Eigen::Vector3d d0(1.0, 0.1, 0.0);
    d0.normalize();

    auto compute_r = [&](const Eigen::Vector3d& _p1, const Eigen::Vector3d& _p2)->Eigen::Vector3d{
        Eigen::Vector3d s = _p2 - _p1;
        Eigen::Vector3d d;
        double l;
        if(!UtilSlam::SafeNormalize_ZDG(s, d, l)) return Eigen::Vector3d(1e3,1e3,1e3);
        return lambdaD * (d.cross(d0)); // 3D residual
    };

    // ---- analytic jacobians (3x3 for p1 and p2) ----
    Eigen::Matrix3d Jp1_ana, Jp2_ana;
    {
        Eigen::Vector3d s = p2 - p1;
        Eigen::Vector3d d;
        double l;
        if(!UtilSlam::SafeNormalize_ZDG(s, d, l))
        {
            std::cout << "[ERROR] degenerate line\n";
            return;
        }

        // dd/ds = (I - d d^T)/l
        Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d*d.transpose();
        Eigen::Matrix3d dd_ds = P / l;

        // r = lambda * (d x d0) = -lambda * [d0]_x d
        Eigen::Matrix3d dr_dd = -lambdaD * Skew(d0);

        Eigen::Matrix3d dr_ds = dr_dd * dd_ds;

        Jp2_ana = dr_ds;
        Jp1_ana = -dr_ds;
    }

    // ---- numeric jacobians ----
    Eigen::Matrix3d Jp1_num, Jp2_num;
    Jp1_num.setZero(); Jp2_num.setZero();

    for(int k=0;k<3;k++)
    {
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        dp(k) = eps;

        // p1
        Eigen::Vector3d rp = compute_r(p1 + dp, p2);
        Eigen::Vector3d rm = compute_r(p1 - dp, p2);
        Jp1_num.col(k) = (rp - rm) / (2*eps);

        // p2
        rp = compute_r(p1, p2 + dp);
        rm = compute_r(p1, p2 - dp);
        Jp2_num.col(k) = (rp - rm) / (2*eps);
    }

    std::cout << "\nJp1 analytic:\n" << Jp1_ana << "\n";
    std::cout << "\nJp1 numeric:\n" << Jp1_num << "\n";
    std::cout << "\nJp1 diff:\n" << (Jp1_ana - Jp1_num) << "\n";

    std::cout << "\nJp2 analytic:\n" << Jp2_ana << "\n";
    std::cout << "\nJp2 numeric:\n" << Jp2_num << "\n";
    std::cout << "\nJp2 diff:\n" << (Jp2_ana - Jp2_num) << "\n";

    double maxDiff = std::max((Jp1_ana - Jp1_num).cwiseAbs().maxCoeff(),
                             (Jp2_ana - Jp2_num).cwiseAbs().maxCoeff());

    std::cout << "\nMax |diff| = " << maxDiff << "\n";
    std::cout << "------ end test ------\n";
}


g2o::SE3Quat Optimizer::se3Plus_ZDG(const g2o::SE3Quat& T, const Eigen::Matrix<double,6,1>& dx)
{
    Eigen::Vector3d omega = dx.head<3>();
    Eigen::Vector3d upsilon = dx.tail<3>();
    Sophus::SE3d dT(Sophus::SO3d::exp(omega), upsilon);
    Sophus::SE3d Tnew = dT * Sophus::SE3d(T.rotation(), T.translation());
    return g2o::SE3Quat(Tnew.rotationMatrix(), Tnew.translation());
}

// Local Bundle Adjustment with Line Support, Old Version (Plucker Coordinates)
void Optimizer::LocalBundleAdjustmentWithLinesPluckerOld(
    KeyFrame *pKF, 
    bool* pbStopFlag, 
    Map* pMap, 
    int& num_fixedKF, 
    int& num_OptKF, 
    int& num_MPs, 
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // -----------------------------
    // Step 1: 局部关键帧 BFS
    // -----------------------------
    std::list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(KeyFrame* pKFi : vNeighKFs)
    {
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    // -----------------------------
    // Step 2: 局部 MapPoint
    // -----------------------------
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        for(MapPoint* pMP : pKFi->GetMapPointMatches())
        {
            if(pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if(pMP->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapPoints.push_back(pMP);
                    pMP->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }
    // -----------------------------
    // Step 3: 局部 MapLine
    // -----------------------------
    list<MapLine*> lLocalMapLines;
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        for(MapLine* pML : pKFi->GetMapLineMatches())
        {
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }
    // -----------------------------
    // Step 4: Fixed KeyFrames
    // -----------------------------
    list<KeyFrame*> lFixedCameras;
    for(MapPoint* pMP : lLocalMapPoints)
    {
        for(auto& obs : pMP->GetObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size();
    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LBA: No fixed keyframes, abort", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // -----------------------------
    // Step 5: 构建优化器
    // -----------------------------
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if(pMap->IsInertial())
        solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // -----------------------------
    // Step 6: 添加关键帧顶点
    // -----------------------------
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3f Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid) maxKFid = pKFi->mnId;
    }
    for(KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3f Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid) maxKFid = pKFi->mnId;
    }

    // -----------------------------
    // Step 7: MapPoint 顶点 + 边
    // -----------------------------
    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vector<KeyFrame*> vpEdgeKFMono;
    vector<MapPoint*> vpMapPointEdgeMono;
    int nEdges = 0;

    for(MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        vPoint->setId(pMP->mnId + maxKFid + 1);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        for(auto& obs : pMP->GetObservations())
        {
            //mono
            KeyFrame* pKFi = obs.first;
            if(pKFi->isBad()) continue;
            const int idx = get<0>(obs.second);
            if(idx < 0) continue;
            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[idx];
            Eigen::Vector2d meas(kpUn.pt.x, kpUn.pt.y);
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
            e->setVertex(0, vPoint);
            e->setVertex(1, optimizer.vertex(pKFi->mnId));
            e->setMeasurement(meas);
            // 必须加上这个，否则 computeError 会崩溃 
            e->pCamera = pKFi->mpCamera;
            e->setInformation(Eigen::Matrix2d::Identity() * pKFi->mvInvLevelSigma2[kpUn.octave]);
            //e->setInformation(Eigen::Matrix2d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));
            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vpEdgeKFMono.push_back(pKFi);
            vpMapPointEdgeMono.push_back(pMP);
            nEdges++;
        }
    }

    // -----------------------------
    // Step 8: MapLine 顶点 + EdgeSE3ProjectPluckerLine_PoseAndLine 边
    // -----------------------------
    // std::vector<EdgeSE3ProjectPluckerLine_PoseAndLine*> vpEdgesLine;
    // std::vector<KeyFrame*> vpEdgeKFLine;
    // std::vector<MapLine*> vpMapLineEdge;
    // for(MapLine* pML : lLocalMapLines)
    // {
    //     VertexLinePlucker* vLine = new VertexLinePlucker();
    //     vLine->setEstimate(pML->GetPluckerLine());
    //     vLine->setId(pML->mnId + maxKFid + 1 + lLocalMapPoints.size());
    //     vLine->setMarginalized(true);
    //     optimizer.addVertex(vLine);
    //     for(auto& obs : pML->GetLineObservations())
    //     {
    //         KeyFrame* pKFi = obs.first;
    //         if(pKFi->isBad()) continue;
    //         EdgeSE3ProjectPluckerLine_PoseAndLine* e = new EdgeSE3ProjectPluckerLine_PoseAndLine();
    //         e->setVertex(0, optimizer.vertex(pKFi->mnId));
    //         e->setVertex(1, vLine);
    //         e->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
    //         Eigen::Vector3f ob_line_projected_pnt = pML->GetProjectedLineABC(pKFi);
    //         e->SetObservedLineABC(ob_line_projected_pnt[0],ob_line_projected_pnt[1], ob_line_projected_pnt[2]);
    //         g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
    //         e->setRobustKernel(rk);
    //         rk->setDelta(3.0);
    //         optimizer.addEdge(e);
    //         vpEdgesLine.push_back(e);
    //         vpEdgeKFLine.push_back(pKFi);
    //         vpMapLineEdge.push_back(pML);
    //         nEdges++;
    //     }
    // }
    num_MPs = lLocalMapPoints.size();
    num_lines = lLocalMapLines.size();
    num_edges = nEdges;

    // -----------------------------
    // Step 9: 优化
    // -----------------------------
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // -----------------------------
    // Step 10: 剔除 MapPoint + MapLine Outlier
    // -----------------------------
    for(size_t i=0; i<vpEdgesMono.size(); i++)
    {
        auto e = vpEdgesMono[i];
        auto pMP = vpMapPointEdgeMono[i];
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            pKFi->EraseMapPointMatch(pMP);
            pMP->EraseObservation(pKFi);
        }
    }

    // for(size_t i=0; i<vpEdgesLine.size(); i++)
    // {
    //     auto e = vpEdgesLine[i];
    //     auto pML = vpMapLineEdge[i];
    //     if(e->chi2()>9.0)
    //     {
    //         KeyFrame* pKFi = vpEdgeKFLine[i];
    //         pKFi->EraseMapLineMatch(pML);
    //         pML->EraseLineObservation(pKFi);
    //     }
    // }
    // -----------------------------
    // Step 11: 更新关键帧 + MapPoint + MapLine
    // -----------------------------
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(vSE3->estimate().rotation().cast<float>(), vSE3->estimate().translation().cast<float>());
        pKFi->SetPose(Tcw);
        opr.addKeyFrame(pKFi);
    }
    for(MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if(!pMP->isRetrived())
        {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }
    // for(MapLine* pML : lLocalMapLines)
    // {
    //     VertexLinePlucker* vLine = static_cast<VertexLinePlucker*>(optimizer.vertex(pML->mnId + maxKFid + 1 + lLocalMapPoints.size()));
    //     // 1) 写回 Plücker
    //     pML->SetPluckerLine(vLine->estimate());
    //     // 2) **用优化后的 Plücker + 各 KeyFrame 观测反投影来重建/更新世界端点**
    //     //    这个函数在 MapLine 类内部实现（你之前实现的 UpdateFromPluckerLine / UpdateFromPlucker）
    //     //    请确保 MapLine 中实现了这个函数并且线程安全（会读取 Plücker、遍历观测、写入端点）
    //     pML->UpdateFromPluckerLine();  
    //     // 3) 更新描述子（基于新的端点/Plücker）
    //     pML->ComputeDistinctiveDescriptors();
    //     pML->UpdateNormalAndDepth();
    //     if(!pML->isRetrived())
    //     {
    //         pML->setRetrived(true);
    //         opr.addMapLine(pML);
    //     }
    // }
    pMap->IncreaseChangeIndex();
}

void Optimizer::LocalBundleAdjustmentWithLinesPlucker(
    KeyFrame *pKF, 
    bool* pbStopFlag, 
    Map* pMap, 
    int& num_fixedKF, 
    int& num_OptKF, 
    int& num_MPs, 
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // --- This function is adapted from the original LocalBundleAdjustment ---
    // Goal: keep all existing KeyFrame and MapPoint vertices exactly as before
    // and *only add MapLine (Plucker) vertices + corresponding line projection edges*.
    // The original pose and point vertices are left unchanged.

    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local Keyframes
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }

    // Local MapLines seen in Local Keyframes
    list<MapLine*> lLocalMapLines;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        vector<MapLine*> vpMLs = pKFi->GetMapLineMatches(); // <-- assumes KeyFrame::GetMapLineMatches() exists
        for(size_t i=0;i<vpMLs.size();i++){
            MapLine* pML = vpMLs[i];
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF!=pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints/MapLines but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // // Also consider MapLine observations when building fixed cameras
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Setup optimizer (same as original)
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();
    pCurrentMap->msFixedKFs.clear();

    // Set Local KeyFrame vertices (unchanged)
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed KeyFrame vertices (unchanged)
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }

    // Set MapPoint vertices (unchanged)
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        //Set edges (unchanged)
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->pCamera = pKFi->mpCamera;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0)// Stereo observation
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }

    //std::cerr << "maxKFid=" << maxKFid << ", lastPointId=" << (maxMapPointId + maxKFid + 1)
    //      << ", nextLineStart=" << lineIdOffset << ", nPoints=" << nPoints << ", nLines=" << nLines << "\n";

    // --- NEW SECTION: add MapLine (Plucker) vertices and edges ---
    // Keep all the existing vertices unchanged. We only add new vertices for MapLines
    //Prepare containers for line edges
    //-----------------------------
    //Step 8: MapLine 顶点 + EdgeSE3ProjectPluckerLine_PoseAndLine 边
    //-----------------------------
    std::vector<EdgeSE3ProjectPluckerLine_PoseAndLine*> vpEdgesLine;
    std::vector<KeyFrame*> vpEdgeKFLine;
    std::vector<MapLine*> vpMapLineEdge;
    // Compute a safe offset for line vertex ids so they do not collide with point ids used above
    int maxMapPointId = 0;
    for(list<MapPoint*>::iterator mit=lLocalMapPoints.begin(), mend=lLocalMapPoints.end(); mit!=mend; mit++){
        if((*mit)->mnId > maxMapPointId) maxMapPointId = (*mit)->mnId;
    }
    int lineIdOffset = static_cast<int>(maxKFid) + maxMapPointId + 2; // +2 safety
    int nLines = 0;

    for(MapLine* pML : lLocalMapLines)
    {
        ORB_SLAM3::VertexLinePlucker* vLine = new ORB_SLAM3::VertexLinePlucker();
        Eigen::Matrix<double,6,1> Lw =  pML->GetPluckerLine();
        std::cerr << "vLine->Lw: " << Lw.transpose() << std::endl;
        vLine->setEstimate(pML->GetPluckerLine());  //获取plucker的公式 //这个很重要
        //打印出来，用于处理数据
        int idLine = pML->mnId + lineIdOffset;
        vLine->setId(idLine);
        optimizer.addVertex(vLine);
        nLines++;
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(!pKFi||pKFi->isBad()) continue;
            EdgeSE3ProjectPluckerLine_PoseAndLine* e = new EdgeSE3ProjectPluckerLine_PoseAndLine();
            e->setVertex(0, optimizer.vertex(pKFi->mnId));
            e->setVertex(1, optimizer.vertex(idLine));
            e->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
            int obline_idx = get<0>(obs.second);
            Eigen::Vector2f sl, el; //end points
            if(!pKFi->GetLineEndPointEigen(obline_idx, sl, el))
            {
                continue;
            }
            Eigen::Vector3f abc_fn = Converter::getLineFromSegment2D(sl, el);
            float abc_n = std::sqrt(abc_fn[0]*abc_fn[0] + abc_fn[1]*abc_fn[1]);
            abc_fn /= abc_n;
            e->SetObservedLineABC(abc_fn[0],abc_fn[1], abc_fn[2]);
            //std::cerr <<"abc_fn:  " << abc_fn.transpose() << std::endl; //debug
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            e->setRobustKernel(rk);
            rk->setDelta(3.0);
            optimizer.addEdge(e);
            vpEdgesLine.push_back(e);
            vpEdgeKFLine.push_back(pKFi);
            vpMapLineEdge.push_back(pML);
            nEdges++;
        }
    }

    //std::cerr << "maxKFid=" << maxKFid << ", lastPointId=" << (maxMapPointId + maxKFid + 1)
    //      << ", nextLineStart=" << lineIdOffset << ", nPoints=" << nPoints << ", nLines=" << nLines << "\n";
    //---------------------end add new vertices for MapLines---------------------------------//

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

    // Check inlier observations for points (unchanged)
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Check inlier observations for lines
    vector<pair<KeyFrame*,MapLine*>> vLineToErase;
    for(size_t i=0, iend=vpMapLineEdge.size(); i<iend; i++){
        ORB_SLAM3::EdgeSE3ProjectPluckerLine_PoseAndLine* e = vpEdgesLine[i];
        MapLine* pML = vpMapLineEdge[i];
        if(pML->isBad()) continue;
        // chi2 threshold for line edges may differ; we use 5.991 as default
        if(e->chi2()>5.991)
        {
            KeyFrame* pKFi = vpEdgeKFLine[i];
            vLineToErase.push_back(make_pair(pKFi,pML));
        }
    }
    // Get Map Mutex
    //unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    if(!vLineToErase.empty()){
        for(size_t i=0;i<vLineToErase.size();i++){
            KeyFrame* pKFi = vLineToErase[i].first;
            MapLine* pMLi = vLineToErase[i].second;
            pKFi->EraseMapLineMatch(pMLi); // 
            pMLi->EraseLineObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    opr.reserveKeyFrames(lLocalKeyFrames.size());
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->SetPose(Tiw);

        opr.addKeyFrame(pKFi);
    }

    //Points
    opr.reserveMapPoints(lLocalMapPoints.size());
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();

        if (!pMP->isRetrived()) {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }

    //Lines: recover optimized plucker and update MapLine world representation
    for(MapLine* pML : lLocalMapLines)
    {
        int idLine = pML->mnId + lineIdOffset;
        ORB_SLAM3::VertexLinePlucker* vLine = static_cast<ORB_SLAM3::VertexLinePlucker*>(optimizer.vertex(idLine));
        if(vLine)
        {
            // 1) 写回 Plücker
            pML->SetPluckerLine(vLine->estimate());
            // 2) **用优化后的 Plücker + 各 KeyFrame 观测反投影来重建/更新世界端点**
            //    这个函数在 MapLine 类内部实现（你之前实现的 UpdateFromPluckerLine / UpdateFromPlucker）
            //    请确保 MapLine 中实现了这个函数并且线程安全（会读取 Plücker、遍历观测、写入端点）
            pML->UpdateFromPluckerLine();  
            // 3) 更新描述子（基于新的端点/Plücker）
            pML->ComputeDistinctiveDescriptors();
            pML->UpdateNormalAndDepth();
            if(!pML->isRetrived())
            {
                pML->setRetrived(true);
                opr.addMapLine(pML);
            }
        }
    }

    // //Lines: recover optimized plucker and update MapLine world representation
    // opr.reserveMapLines(lLocalMapLines.size());
    // for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    // {
    //     MapLine* pML = *lit;
    //     int idLine = pML->mnId + lineIdOffset;
    //     ORB_SLAM3::VertexLinePlucker* vLine = static_cast<ORB_SLAM3::VertexLinePlucker*>(optimizer.vertex(idLine));
    //     if(vLine){
    //         Eigen::Matrix<double,6,1> pluckerOpt = vLine->estimate();
    //         pML->SetPlucker(pluckerOpt.cast<float>());
    //         pML->UpdateDirectionAndDepth(); // <-- assumed helper to refresh cached values
    //         if (!pML->isRetrived()) {
    //             pML->setRetrived(true);
    //             opr.addMapLine(pML);
    //         }
    //     }
    // }

    // --- 14. statistics output ---
    num_OptKF = lLocalKeyFrames.size();
    num_MPs = lLocalMapPoints.size();
    num_lines = nLines;
    num_edges = nEdges;

    pMap->IncreaseChangeIndex();
}

Eigen::Matrix<double,6,1> Optimizer::NormalizePluckerLine(const Eigen::Matrix<double,6,1>& L)
{
    Eigen::Vector3d n = L.head<3>();
    Eigen::Vector3d v = L.tail<3>();

    // 如果 n 为 0 向量，不能归一化
    if (n.norm() < 1e-10) return L;

    // 强制正交： v' = v - proj_n(v)
    Eigen::Vector3d v_proj = v - n * (n.dot(v) / n.squaredNorm());

    Eigen::Matrix<double,6,1> Ln;
    Ln.head<3>() = n;
    Ln.tail<3>() = v_proj;
    // 👉 打印正交约束是否满足
    std::cout << "n·v = " << n.dot(v_proj) << std::endl;
    return Ln;
}

void Optimizer::TestPluckerLineEdgeJacobian() {

    cout << "[Main] Start\n";

    // --- 1. 创建顶点 ---
    cout << "[Main] Creating pose vertex\n";
    auto* vPose = new g2o::VertexSE3Expmap();
    vPose->setId(0);
    vPose->setEstimate(g2o::SE3Quat()); // Identity pose

    cout << "[Main] Creating line vertex\n";
    auto* vLine = new VertexLinePlucker();
    vLine->setId(1);
    Eigen::Matrix<double,6,1> Lw;
    Lw << 0.0, 1.0, 0.0,    // direction
          0.0, 0.0, 1.0;    // moment
    vLine->setEstimate(Lw);

    // --- 2. 构建边 ---
    cout << "[Main] Creating edge\n";
    auto* edge = new EdgeSE3ProjectPluckerLine_PoseAndLine();
    edge->resize(2); // VERY IMPORTANT
    edge->setVertex(0, vPose);
    edge->setVertex(1, vLine);
    edge->SetCameraIntrinsics(500, 500, 320, 240);
    edge->SetObservedLineABC(1.0, 0.0, -320);

    // --- 3. 计算误差和 Jacobian ---
    cout << "[Main] computeError + linearizeOplus\n";
    edge->computeError();
    edge->linearizeOplus();

    auto J_pose_analytic = edge->getJacobianPose();
    auto J_line_analytic = edge->getJacobianLine();

    // --- 4. 数值 Jacobian ---
    double eps = 1e-6;

    Eigen::Matrix<double,2,6> J_pose_numeric;
    for (int i = 0; i < 6; ++i)
    {
        g2o::VertexSE3Expmap tempPose = *vPose;

        Eigen::Matrix<double,6,1> epsVec = Eigen::Matrix<double,6,1>::Zero();
        epsVec[i] = eps;

        g2o::SE3Quat dT = g2o::SE3Quat::exp(epsVec);
        tempPose.setEstimate(dT * vPose->estimate());

        edge->setVertex(0, &tempPose);
        edge->computeError();
        Eigen::Vector2d err_plus = edge->error();

        epsVec[i] = -eps;
        dT = g2o::SE3Quat::exp(epsVec);
        tempPose.setEstimate(dT * vPose->estimate());

        edge->setVertex(0, &tempPose);
        edge->computeError();
        Eigen::Vector2d err_minus = edge->error();

        J_pose_numeric.col(i) = (err_plus - err_minus) / (2 * eps);
    }

    Eigen::Matrix<double,2,6> J_line_numeric;
    for (int i = 0; i < 6; ++i)
    {
        VertexLinePlucker tempLine = *vLine;
        Eigen::Matrix<double,6,1> L = vLine->estimate();
        Eigen::Matrix<double,6,1> dL = Eigen::Matrix<double,6,1>::Zero();
        dL[i] = eps;
        tempLine.setEstimate(NormalizePluckerLine(L + dL));
        edge->setVertex(1, &tempLine);
        edge->computeError();
        Eigen::Vector2d err_plus = edge->error();

        tempLine.setEstimate(NormalizePluckerLine(L - dL));
        edge->setVertex(1, &tempLine);
        edge->computeError();
        Eigen::Vector2d err_minus = edge->error();

        J_line_numeric.col(i) = (err_plus - err_minus) / (2 * eps);
    }

    // --- 5. 打印结果 ---
    cout << "\n=== Pose Jacobian ===" << endl;
    cout << "Analytic:\n" << J_pose_analytic << endl;
    cout << "Numeric:\n" << J_pose_numeric << endl;
    cout << "Diff:\n" << J_pose_analytic - J_pose_numeric << endl;

    cout << "\n=== Line Jacobian ===" << endl;
    cout << "Analytic:\n" << J_line_analytic << endl;
    cout << "Numeric:\n" << J_line_numeric << endl;
    cout << "Diff:\n" << J_line_analytic - J_line_numeric << endl;

    // === 检查每一列差异，定位最大误差的列 ===
    std::cout << "\n=== Per-column Difference for Line Jacobian ===" << std::endl;

    double max_error = 0.0;
    int max_col = -1;

    for (int i = 0; i < 6; ++i)
    {
        Eigen::Vector2d diff = J_line_analytic.col(i) - J_line_numeric.col(i);
        double norm = diff.norm();
        std::cout << "Column " << i << " diff norm = " << norm
                  << ", diff = " << diff.transpose() << std::endl;

        if (norm > max_error)
        {
            max_error = norm;
            max_col = i;
        }

    }

    std::cout << ">> Max error at column " << max_col << ", norm = " << max_error << std::endl;

    cout << "[Main] Done\n";

    // cout << "[Main] Start\n";

    // // 创建顶点
    // cout << "[Main] Creating pose vertex\n";
    // auto* vPose = new g2o::VertexSE3Expmap();
    // vPose->setId(0);
    // vPose->setEstimate(g2o::SE3Quat()); // identity pose

    // cout << "[Main] Creating line vertex\n";
    // auto* vLine = new VertexLinePlucker();
    // vLine->setId(1);
    // Eigen::Matrix<double,6,1> Lw;
    // Lw << 0.0, 1.0, 0.0,    // direction
    //       0.0, 0.0, 1.0;    // moment
    // vLine->setEstimate(Lw);

    // cout << "[Main] Creating edge\n";
    // auto* edge = new EdgeSE3ProjectPluckerLine_PoseAndLine();
    // edge->resize(2);  // VERY IMPORTANT
    // edge->setVertex(0, vPose);
    // edge->setVertex(1, vLine);
    // edge->SetCameraIntrinsics(500, 500, 320, 240);
    // edge->SetObservedLineABC(1.0, 0.0, -320);

    // cout << "[Main] computeError + linearizeOplus\n";
    // edge->computeError();
    // edge->linearizeOplus();

    // cout << "[Main] Extracting analytic Jacobians\n";
    // //std::cout << "Jacobian vector size: " << edge->jacobians().size() << std::endl;
    // auto J_pose_analytic = edge->getJacobianPose();
    // std::cerr << "J_pose_analytic: \n" << J_pose_analytic << std::endl;
    // auto J_line_analytic = edge->getJacobianLine();

    // cout << "[Main] Computing numerical Jacobians\n";
    // auto J_pose_numeric = ComputeNumericalJacobianPose(edge, vPose);
    // auto J_line_numeric = ComputeNumericalJacobianLine(edge, vLine);

    // cout << "\n=== Pose Jacobian ===" << endl;
    // cout << "Analytic:\n" << J_pose_analytic << endl;
    // cout << "Numeric:\n" << J_pose_numeric << endl;
    // cout << "Diff:\n" << J_pose_analytic - J_pose_numeric << endl;

    // cout << "\n=== Line Jacobian ===" << endl;
    // cout << "Analytic:\n" << J_line_analytic << endl;
    // cout << "Numeric:\n" << J_line_numeric << endl;
    // cout << "Diff:\n" << J_line_analytic - J_line_numeric << endl;

    // cout << "[Main] Done\n";
}

Eigen::Matrix<double,2,6> Optimizer::ComputeNumericalJacobianPose(
    ORB_SLAM3::EdgeSE3ProjectPluckerLine_PoseAndLine* edge,
    g2o::VertexSE3Expmap* vPose)
{
    cout << "[NumJacPose] Start\n";
    Eigen::Matrix<double,2,6> Jnum;
    const double eps = 1e-6;
    g2o::SE3Quat T0 = vPose->estimate();

    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double,6,1> d;
        d.setZero(); d[i] = eps;

        g2o::SE3Quat T_plus = g2o::SE3Quat::exp(d) * T0;
        g2o::SE3Quat T_minus = g2o::SE3Quat::exp(-d) * T0;

        vPose->setEstimate(T_plus);
        edge->computeError();
        Eigen::Vector2d err_plus = edge->error();

        vPose->setEstimate(T_minus);
        edge->computeError();
        Eigen::Vector2d err_minus = edge->error();

        if (!err_plus.allFinite() || !err_minus.allFinite()) {
            cerr << "[WARN] NaN in pose diff i=" << i << endl;
            Jnum.col(i).setZero();
        } else {
            Jnum.col(i) = (err_plus - err_minus) / (2.0 * eps);
        }
    }

    vPose->setEstimate(T0);
    edge->computeError();
    cout << "[NumJacPose] Done\n";
    return Jnum;
}

Eigen::Matrix<double,2,6> Optimizer::ComputeNumericalJacobianLine(
    ORB_SLAM3::EdgeSE3ProjectPluckerLine_PoseAndLine* edge,
    ORB_SLAM3::VertexLinePlucker* vLine)
{
    cout << "[NumJacLine] Start\n";
    Eigen::Matrix<double,2,6> Jnum;
    const double eps = 1e-6;
    Eigen::Matrix<double,6,1> L0 = vLine->estimate();

    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double,6,1> Lp = L0;
        Eigen::Matrix<double,6,1> Lm = L0;
        Lp(i) += eps;
        Lm(i) -= eps;

        vLine->setEstimate(Lp);
        edge->computeError();
        Eigen::Vector2d err_plus = edge->error();

        vLine->setEstimate(Lm);
        edge->computeError();
        Eigen::Vector2d err_minus = edge->error();

        if (!err_plus.allFinite() || !err_minus.allFinite()) {
            cerr << "[WARN] NaN in line diff i=" << i << endl;
            Jnum.col(i).setZero();
        } else {
            Jnum.col(i) = (err_plus - err_minus) / (2.0 * eps);
        }
    }

    vLine->setEstimate(L0);
    edge->computeError();
    cout << "[NumJacLine] Done\n";
    return Jnum;
}

void Optimizer::TestPluckerLinesBundleEdge()
{
    // // solver
    // g2o::SparseOptimizer optimizer;
    // optimizer.setVerbose(true);
    // // --- solver ---
    // using BlockSolverType = g2o::BlockSolver< g2o::BlockSolverTraits<6,1> >;
    // auto* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();
    // auto* solver = new g2o::OptimizationAlgorithmLevenberg(new BlockSolverType(linearSolver));
    // optimizer.setAlgorithm(solver);
    //  // -------- Vertex SE3 ----------
    // g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    // vSE3->setId(0);
    // vSE3->setEstimate(g2o::SE3Quat());
    // optimizer.addVertex(vSE3);
    // // -------- Generate multiple lines and depths ----------
    // const int numLines = 5;
    // std::vector<VertexDepth*> depthVertices;
    // std::vector<EdgePointToPluckerLinePoseAndDepth*> edges;
    // Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    // for (int i = 0; i < numLines; ++i) {
    //     Eigen::Matrix<double,6,1> plucker;
    //     plucker << i+1, i+2, i+3, i+4, i+5, i+6;
    //     // 两个 depth 顶点 per line
    //     VertexDepth* vD0 = new VertexDepth();
    //     vD0->setId(1 + i*2);
    //     vD0->setEstimate(2.0 + i);
    //     optimizer.addVertex(vD0);
    //     depthVertices.push_back(vD0);
    //     VertexDepth* vD1 = new VertexDepth();
    //     vD1->setId(2 + i*2);
    //     vD1->setEstimate(3.0 + i);
    //     optimizer.addVertex(vD1);
    //     depthVertices.push_back(vD1);
    //     // 两个边
    //     EdgePointToPluckerLinePoseAndDepth* e0 =
    //         new EdgePointToPluckerLinePoseAndDepth(Eigen::Vector2d(100+i*10,200+i*5), Kinv, plucker);
    //     e0->setVertex(0, vSE3);
    //     e0->setVertex(1, vD0);
    //     e0->setMeasurement(Eigen::Vector3d::Zero());
    //     e0->setInformation(Eigen::Matrix3d::Identity());
    //     optimizer.addEdge(e0);
    //     edges.push_back(e0);
    //     EdgePointToPluckerLinePoseAndDepth* e1 =
    //         new EdgePointToPluckerLinePoseAndDepth(Eigen::Vector2d(150+i*8,250+i*7), Kinv, plucker);
    //     e1->setVertex(0, vSE3);
    //     e1->setVertex(1, vD1);
    //     e1->setMeasurement(Eigen::Vector3d::Zero());
    //     e1->setInformation(Eigen::Matrix3d::Identity());
    //     optimizer.addEdge(e1);
    //     edges.push_back(e1);
    // }
    // // -------- Optimize ----------
    // optimizer.initializeOptimization();
    // optimizer.optimize(10);
    // std::cout << "Optimization finished. Depth estimates:\n";
    // for (size_t i = 0; i < depthVertices.size(); ++i) {
    //     std::cout << "vD" << i << " = " << depthVertices[i]->estimate() << "\n";
    // }

    // // -------- Vertex SE3 ----------
    // g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    // vSE3->setId(0);
    // vSE3->setEstimate(g2o::SE3Quat());
    // optimizer.addVertex(vSE3);
    // // -------- VertexDepth ----------
    // VertexDepth* vD0 = new VertexDepth();
    // vD0->setId(1);
    // vD0->setEstimate(2.5);
    // optimizer.addVertex(vD0);
    // VertexDepth* vD1 = new VertexDepth();
    // vD1->setId(2);
    // vD1->setEstimate(3.5);
    // optimizer.addVertex(vD1);
    // // -------- Edges ----------
    // Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    // Eigen::Matrix<double,6,1> plucker;
    // plucker << 1,2,3,4,5,6;
    // EdgePointToPluckerLinePoseAndDepth* e0 =
    //     new EdgePointToPluckerLinePoseAndDepth(Eigen::Vector2d(100,200), Kinv, plucker);
    // e0->setVertex(0, vSE3);
    // e0->setVertex(1, vD0);
    // e0->setMeasurement(Eigen::Vector3d::Zero());
    // e0->setInformation(Eigen::Matrix3d::Identity());
    // optimizer.addEdge(e0);
    // EdgePointToPluckerLinePoseAndDepth* e1 =
    //     new EdgePointToPluckerLinePoseAndDepth(Eigen::Vector2d(150,250), Kinv, plucker);
    // e1->setVertex(0, vSE3);
    // e1->setVertex(1, vD1);
    // e1->setMeasurement(Eigen::Vector3d::Zero());
    // e1->setInformation(Eigen::Matrix3d::Identity());
    // optimizer.addEdge(e1);
    // // -------- Optimize ----------
    // optimizer.initializeOptimization();
    // optimizer.optimize(5);
    // std::cout << "Optimization finished. Depth estimates: "
    //           << vD0->estimate() << ", " << vD1->estimate() << std::endl;

    // g2o::SparseOptimizer optimizer;
    // optimizer.setVerbose(true);
    // using BlockSolverType = g2o::BlockSolver< g2o::BlockSolverTraits<6,1> >;
    // auto* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();
    // auto* solver = new g2o::OptimizationAlgorithmLevenberg(
    //                     new BlockSolverType(linearSolver));
    // optimizer.setAlgorithm(solver);
    // // --- create pose vertex ---
    // auto* vPose = new g2o::VertexSE3Expmap();
    // vPose->setId(0);
    // vPose->setEstimate(g2o::SE3Quat(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0,0,0)));
    // optimizer.addVertex(vPose);
    // // --- create depth vertex ---
    // auto* vDepth = new VertexDepth();
    // vDepth->setId(1);
    // vDepth->setEstimate(1.0);
    // optimizer.addVertex(vDepth);
    // // --- Plücker line ---
    // Eigen::Matrix<double,6,1> Lw;
    // Lw << 0.1,0.2,0.3, 1,0,0; // n nonzero, v nonzero
    // std::cerr << "Lw: " <<  Lw.transpose() << std::endl;
    // // --- one edge ---
    // Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    // Eigen::Vector2d px(0.5,0.5);
    // auto* e = new EdgePointToPluckerLinePoseAndDepthNew(px,Kinv,Lw);
    // e->setVertex(0, vPose);
    // e->setVertex(1, vDepth);
    // e->setMeasurement(Eigen::Vector3d::Zero());
    // e->setInformation(Eigen::Matrix3d::Identity());
    // optimizer.addEdge(e);
    // std::cerr << "1111111111" << std::endl;
    // // --- optimize ---
    // optimizer.initializeOptimization();
    // optimizer.optimize(10);
    // std::cout << "Optimized depth: " << vDepth->estimate() << std::endl;
    // std::cout << "Optimized depth: " << vDepth->estimate() << std::endl;
    // std::cout << "Optimized depth: " << vDepth->estimate() << std::endl;
    // std::cout << "Optimized depth: " << vDepth->estimate() << std::endl;

// std::cout << "==== Minimal Jacobian Test (CORRECT) ====" << std::endl;
// // --- vertices ---
// auto* vPose = new g2o::VertexSE3Expmap();
// vPose->setId(0);
// vPose->setEstimate(g2o::SE3Quat());
// auto* vDepth = new VertexDepth();
// vDepth->setId(1);
// vDepth->setEstimate(2.0);
// // --- edge ---
// auto* edge = new EdgeSimpleSE3Depth();
// edge->setVertex(0, vPose);
// edge->setVertex(1, vDepth);
// edge->setMeasurement(Eigen::Vector3d::Zero());
// edge->setInformation(Eigen::Matrix3d::Identity());
// // --- optimizer (BlockSolverX) ---
// g2o::SparseOptimizer opt;
// using BlockSolverX =
//     g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
// using LinearSolverX =
//     g2o::LinearSolverEigen<BlockSolverX::PoseMatrixType>;
// auto* linearSolver = new LinearSolverX();
// auto* blockSolver  = new BlockSolverX(linearSolver);
// auto* solver =
//     new g2o::OptimizationAlgorithmLevenberg(blockSolver);
// opt.setAlgorithm(solver);
// opt.setVerbose(false);
// // --- add graph ---
// opt.addVertex(vPose);
// opt.addVertex(vDepth);
// opt.addEdge(edge);
// // --- 关键步骤（缺一不可） ---
// opt.initializeOptimization();
// opt.optimize(0);   // ⭐ 这一行是“生死线”
// // ❌ 不要再手动调用：
// // edge->computeError();
// // edge->linearizeOplus();
// std::cout << "OK, no segfault." << std::endl;

    std::cout << "==== Test EdgePointToPluckerLine ====" << std::endl;

    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    Eigen::Vector2d px(0.1, 0.2);

    Eigen::Matrix<double,6,1> Lw;
    Lw << 0.1, 0.2, 0.3, 1.0, 0.2, 0.1;

    auto* vPose = new g2o::VertexSE3Expmap();
    vPose->setId(0);
    vPose->setEstimate(g2o::SE3Quat());

    auto* vDepth = new VertexDepth();
    vDepth->setId(1);
    vDepth->setEstimate(2.0);

    auto* edge =
        new EdgePointToPluckerLinePoseAndDepthNew(px, Kinv, Lw);
    edge->setVertex(0, vPose);
    edge->setVertex(1, vDepth);
    edge->setMeasurement(Eigen::Vector3d::Zero());
    edge->setInformation(Eigen::Matrix3d::Identity());

    g2o::SparseOptimizer opt;

    using BlockSolverX =
        g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    using LinearSolverX =
        g2o::LinearSolverEigen<BlockSolverX::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        new BlockSolverX(new LinearSolverX()));

    opt.setAlgorithm(solver);
    opt.setVerbose(false);

    opt.addVertex(vPose);
    opt.addVertex(vDepth);
    opt.addEdge(edge);

    opt.initializeOptimization();
    opt.optimize(0);   // ⭐ 关键

    std::cout << "OK: no segfault, Jacobians allocated." << std::endl;


}

void Optimizer::CheckJacobianNumerical()
{
std::cout << "==== Numerical Jacobian Check (Scheme A: right-multiplicative) ===="
              << std::endl;

    constexpr double eps = 1e-6;

    // =====================================================
    // Test data (fixed, reproducible)
    // =====================================================
    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    Eigen::Vector2d px(0.15, -0.1);

    Eigen::Matrix<double,6,1> Lw;
    Lw << 0.2, -0.1, 0.3,   1.0, 0.3, -0.2;

    // =====================================================
    // Phase A: Analytic Jacobian (via optimizer)
    // =====================================================
    Eigen::Matrix<double,3,6> J_pose_ana;
    Eigen::Matrix<double,3,1> J_depth_ana;
    Eigen::Vector3d e0;

    {
        // ---------- vertices ----------
        auto* vPoseA = new g2o::VertexSE3Expmap();
        vPoseA->setId(0);
        vPoseA->setEstimate(g2o::SE3Quat());   // identity
        vPoseA->setFixed(false);

        auto* vDepthA = new VertexDepth();
        vDepthA->setId(1);
        vDepthA->setEstimate(2.5);
        vDepthA->setFixed(false);

        // ---------- edge ----------
        auto* edgeA =
            new EdgePointToPluckerLinePoseAndDepthNew(px, Kinv, Lw);
        edgeA->setId(0);
        edgeA->setVertex(0, vPoseA);
        edgeA->setVertex(1, vDepthA);
        edgeA->setMeasurement(Eigen::Vector3d::Zero());
        edgeA->setInformation(Eigen::Matrix3d::Identity());
        edgeA->setLevel(0);

        // ---------- optimizer ----------
        g2o::SparseOptimizer opt;

        using BlockSolverX =
            g2o::BlockSolver<g2o::BlockSolverTraits<-1,-1>>;
        using LinearSolverX =
            g2o::LinearSolverEigen<BlockSolverX::PoseMatrixType>;

        opt.setAlgorithm(
            new g2o::OptimizationAlgorithmLevenberg(
                new BlockSolverX(new LinearSolverX())));
        opt.setVerbose(false);

        opt.addVertex(vPoseA);
        opt.addVertex(vDepthA);
        opt.addEdge(edgeA);

        opt.initializeOptimization();

        // ⭐ 必须 optimize >= 1 才会调用 linearizeOplus
        edgeA->dbg_jac_updated = false;
        opt.optimize(1);

        if(!edgeA->dbg_jac_updated)
        {
            std::cerr << "[CHK] ERROR: linearizeOplus() was NOT called."
                      << std::endl;
            return;
        }

        // ---------- fetch analytic Jacobians ----------
        J_pose_ana  = edgeA->dbg_J_pose;
        J_depth_ana = edgeA->dbg_J_depth;

        // ---------- base error ----------
        edgeA->computeError();
        e0 = edgeA->error();

        std::cout << "[CHK] Analytic Jacobians computed." << std::endl;
    }
    // Phase A objects are deleted by optimizer destructor

    // =====================================================
    // Phase B: Numerical Jacobian (NO optimizer)
    // =====================================================
    Eigen::Matrix<double,3,6> J_pose_num;
    Eigen::Matrix<double,3,1> J_depth_num;
    J_pose_num.setZero();
    J_depth_num.setZero();

    // ---------- fresh vertices ----------
    auto* vPoseB = new g2o::VertexSE3Expmap();
    vPoseB->setEstimate(g2o::SE3Quat());   // same base point

    auto* vDepthB = new VertexDepth();
    vDepthB->setEstimate(2.5);

    // ---------- fresh edge ----------
    auto* edgeB =
        new EdgePointToPluckerLinePoseAndDepthNew(px, Kinv, Lw);
    edgeB->setVertex(0, vPoseB);
    edgeB->setVertex(1, vDepthB);
    edgeB->setMeasurement(Eigen::Vector3d::Zero());
    edgeB->setInformation(Eigen::Matrix3d::Identity());

    // ---------- base error ----------
    edgeB->computeError();
    Eigen::Vector3d e_base = edgeB->error();

    // ---------- pose numeric Jacobian ----------
    const g2o::SE3Quat T0 = vPoseB->estimate();

    for(int i = 0; i < 6; ++i)
    {
        Eigen::Matrix<double,6,1> dx =
            Eigen::Matrix<double,6,1>::Zero();
        dx[i] = eps;

        // Scheme A: RIGHT-multiplicative perturbation
        vPoseB->setEstimate(T0 * g2o::SE3Quat::exp(dx));
        edgeB->computeError();

        J_pose_num.col(i) = (edgeB->error() - e_base) / eps;
    }
    vPoseB->setEstimate(T0);

    // ---------- depth numeric Jacobian ----------
    const double d0 = vDepthB->estimate();
    vDepthB->setEstimate(d0 + eps);
    edgeB->computeError();
    J_depth_num.col(0) = (edgeB->error() - e_base) / eps;
    vDepthB->setEstimate(d0);

    // =====================================================
    // Print comparison
    // =====================================================
    std::cout << "\n[J_pose analytic]\n" << J_pose_ana << std::endl;
    std::cout << "\n[J_pose numeric]\n"  << J_pose_num << std::endl;
    std::cout << "\n[J_pose diff]\n"     << (J_pose_ana - J_pose_num) << std::endl;

    std::cout << "\n[J_depth analytic]\n"
              << J_depth_ana.transpose() << std::endl;
    std::cout << "\n[J_depth numeric]\n"
              << J_depth_num.transpose() << std::endl;
    std::cout << "\n[J_depth diff]\n"
              << (J_depth_ana - J_depth_num).transpose() << std::endl;

    std::cout << "==== End Numerical Jacobian Check (Scheme A) ===="
              << std::endl;

    // ---------- cleanup ----------
    delete edgeB;
    delete vPoseB;
    delete vDepthB;
}


void Optimizer::TestNumericalJacobian_PointToPlucker()
{
    // std::cout << "==== Numerical Jacobian Check ====" << std::endl;

    // // ---------- 构造测试数据 ----------
    // Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    // Eigen::Vector2d px(0.15, -0.1);

    // Eigen::Matrix<double,6,1> Lw;
    // Lw << 0.2, -0.1, 0.3,   1.0, 0.3, -0.2;

    // auto* vPose = new g2o::VertexSE3Expmap();
    // vPose->setId(0);
    // vPose->setEstimate(g2o::SE3Quat());   // identity

    // auto* vDepth = new VertexDepth();
    // vDepth->setId(1);
    // vDepth->setEstimate(2.5);

    // ORB_SLAM3::EdgePointToPluckerLinePoseAndDepthNew* edge =
    //     new EdgePointToPluckerLinePoseAndDepthNew(px, Kinv, Lw);
    // edge->setId(0);          // ⭐ 必须
    // edge->setVertex(0, vPose);
    // edge->setVertex(1, vDepth);
    // edge->setMeasurement(Eigen::Vector3d::Zero());
    // edge->setInformation(Eigen::Matrix3d::Identity());

    // ---------- 调用数值 Jacobian Checker ----------
    CheckJacobianNumerical();

    //std::cout << "==== End Numerical Jacobian Check ====" << std::endl;
}

void Optimizer::CheckDuplicateVertexID(g2o::SparseOptimizer& optimizer)
{
    std::unordered_map<int,int> idCount;

    // 遍历 optimizer 所有 vertex
    for (auto it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
    {
        int id = it->first; // vertex ID
        idCount[id]++;
    }

    // 输出重复的 ID
    bool hasDup = false;
    for (const auto& kv : idCount)
    {
        if(kv.second > 1)
        {
            std::cerr << "[Duplicate Vertex ID] ID=" << kv.first 
                      << " occurs " << kv.second << " times." << std::endl;
            hasDup = true;
        }
    }

    if(!hasDup)
    {
        std::cout << "No duplicate vertex ID found." << std::endl;
    }
}

void Optimizer::LocalBundleAdjustmentWithLinesPlucker_Alternating(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // --- This function is adapted from the original LocalBundleAdjustment ---
    // Goal: keep all existing KeyFrame and MapPoint vertices exactly as before
    // and *only add MapLine (Plucker) vertices + corresponding line projection edges*.
    // The original pose and point vertices are left unchanged.

    // Local KeyFrames: First Breath Search from Current Keyframe
    std::list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }
    // Local MapPoints seen in Local Keyframes
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        std::vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }
    // Local MapLines seen in Local Keyframes
    list<MapLine*> lLocalMapLines;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        vector<MapLine*> vpMLs = pKFi->GetMapLineMatches(); // <-- assumes KeyFrame::GetMapLineMatches() exists
        for(size_t i=0;i<vpMLs.size();i++){
            MapLine* pML = vpMLs[i];
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF!=pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints/MapLines but that are not Local Keyframes
    std::list<KeyFrame*> lFixedCameras;
    for(std::list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        std::map<KeyFrame*,std::tuple<int,int>> observations = (*lit)->GetObservations();
        for(std::map<KeyFrame*,std::tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // // Also consider MapLine observations when building fixed cameras
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // ---------------------------
    // Alternating loop parameters
    // ---------------------------
    double value_scale = 1.0;
    
    const int ALT_ITER = 1; // 2~4 typical
    // Outer alternating loop
    bool initial_iter_plucker_flag = true;
    for(int alt = 0; alt < ALT_ITER; alt++)
    {
        if(alt == 0)      value_scale = 3.0;
        else if(alt == 1) value_scale = 2.0;
        else              value_scale = 1.0;  // alt>=2
        // ----------------------------
        // (A) Fit Plücker lines using current poses + endpoints
        // ----------------------------
        if(initial_iter_plucker_flag)
        {
            for(MapLine* pML : lLocalMapLines)
            {
                std::vector<Eigen::Vector3d> pts_w;
                // Preferred: if MapLine stores world endpoints or cached 3D endpoints, use them:
                // Assume MapLine::GetAllWorldEndPoints() returns vector<Eigen::Vector3d> of world pts (all obs endpoints)
                if(pML->HasCachedWorldObservationLineEndPoints()) {
                    //pts_w = pML->GetAllWorldEndPoints(); // <-- you should implement next...
                } else {
                    // Fallback: for each observation, backproject endpoints using KeyFrame pose and stored per-observation depths
                    // Assumes MapLine stores per-observation endpoint depths: pML->GetObservationData(pKFi) -> {d0,d1}
                    for(auto& obs : pML->GetLineObservations())
                    {
                        KeyFrame* pKFi = obs.first;
                        int line_idx = get<0>(obs.second);
                        // Get pixel endpoints in that KF
                        Eigen::Vector2f sl, el;
                        if(!pKFi->GetLineEndPointEigen(line_idx, sl, el))
                            continue;
                        // Obtain per-observation depths or initial depths (you need to provide or compute these)
                        float d0 = pML->GetObservationDepth0(pKFi, line_idx); // <- implement or store initial depth
                        float d1 = pML->GetObservationDepth1(pKFi, line_idx);
                        // Backproject using intrinsics
                        Eigen::Vector3d ray0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1])); // implement or use K^-1 * [u,v,1]
                        Eigen::Vector3d ray1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
                        // World point = Twc * (d * ray)
                        g2o::SE3Quat Tcw = g2o::SE3Quat(pKFi->GetPose().unit_quaternion().cast<double>(), pKFi->GetPose().translation().cast<double>());
                        g2o::SE3Quat Twc = Tcw.inverse();
                        Eigen::Vector3d pw0 = Twc * (d0 * ray0);
                        Eigen::Vector3d pw1 = Twc * (d1 * ray1);
                        pts_w.push_back(pw0);
                        pts_w.push_back(pw1);
                    }
                }
                // If we have at least 2 points, fit
                if(pts_w.size() >= 2)
                {
                    Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(pts_w);
                    // Write back into MapLine temporary plucker estimate (do not yet commit endpoints)
                    pML->SetPluckerLine(Lw);
                }
            } // end for each mapline
            initial_iter_plucker_flag = false;
        }
            
        // ----------------------------
        // (B) Build g2o optimizer with MapPoints + Poses + Line vertices (line vertices setFixed(true))
        // and edges: original point edges + line projection edges (using observed image line abc).
        // Then optimize (poses and points will change; line vertices fixed).
        // ----------------------------
        g2o::SparseOptimizer optimizer;
        //optimizer.setVerbose(true);
        optimizer.clear();
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        if (pMap->IsInertial())
            solver->setUserLambdaInit(100.0);
        optimizer.setAlgorithm(solver);
        //optimizer.setVerbose(false);
        optimizer.setVerbose(true);
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Add Local KeyFrame vertices (same as original)
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }
        num_OptKF = lLocalKeyFrames.size();

        // Fixed Keyframes (same as original)
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Add MapPoint vertices (unchanged)
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();
        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;    //To do next
        vpEdgesBody.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nPoints = 0;
        int nEdges = 0;

        // Add point vertices and edges (copy from original LBA code)
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            if(!pMP)
            {
                continue;
            }
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;
            const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
            for(auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                KeyFrame* pKFi = mit->first;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);
                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;
                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        e->pCamera = pKFi->mpCamera;
                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        nEdges++;
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0) // Stereo
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;
                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }

         //Step 8: MapLine 顶点 + EdgePointToPluckerLine 边
        //-----------------------------
        // =====================================================
        // Step X: MapLine Depth Vertices + Point-to-Plücker Edges
        // (ORB-SLAM3 style, SAFE, Scheme A)
        // =====================================================

        // ---- containers (与 MapPoint LBA 对齐) ----
        std::vector<EdgePointToPluckerLinePoseAndDepthNew*> vpEdgesLine;
        vpEdgesLine.reserve(lLocalMapLines.size() * 10);
        std::vector<KeyFrame*> vpEdgeKFLine;
        vpEdgeKFLine.reserve(lLocalMapLines.size() * 10);
        std::vector<MapLine*>  vpMapLineEdge;
        vpMapLineEdge.reserve(lLocalMapLines.size() * 10);

        // 每一个 (MapLine*, KeyFrame*, endpointIndex) 对应一个 VertexDepth
        std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*> depthVertexMap;
        
        // ---- 取得安全起始 vertex id（不要用 rbegin；ORB-SLAM3 这版 g2o 是 tr1 unordered_map）----
        int nextVid = GetMaxVertexId(optimizer) + 1;

        // ---- 可选：给 line depth vertex 留大间隔，避免与你 MapPoint vertex 发生任何潜在冲突 ----
        if(nextVid < (int)(maxKFid + 100000)) nextVid = (int)(maxKFid + 100000);

        // -------------------------------------------------
        // 遍历 Local MapLines
        // -------------------------------------------------
        // 在进入 lLocalMapLines 循环前做：
        // int nextObId = GetMaxVertexId(optimizer) + 1;

        for (MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;
            Eigen::Matrix<double,6,1> Lw = pML->GetPluckerLine().cast<double>();
            if(!Lw.allFinite()) continue;
            if(Lw.tail<3>().norm() < 1e-9) continue;
            const auto& obsList = pML->GetLineObservations();
            for(const auto& obsPair : obsList)
            {
                KeyFrame* pKFi = obsPair.first;
                if(!pKFi || pKFi->isBad()) continue;
                int idx = std::get<0>(obsPair.second);
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
                if(!sl.allFinite() || !el.allFinite()) continue;
                // depth init
                const float d0f = pML->GetObservationDepth0(pKFi, idx);
                const float d1f = pML->GetObservationDepth1(pKFi, idx);
                const double d0 = (double)d0f;
                const double d1 = (double)d1f;
                if(!IsFiniteDepth(d0) || !IsFiniteDepth(d1)) continue;
                // pose vertex must exist
                auto* vPoseBase = optimizer.vertex(pKFi->mnId);
                if(!vPoseBase) continue;
                // ✅ 不用 dynamic_cast（ORB-SLAM3 常见关闭 RTTI）
                auto* vPose = static_cast<g2o::VertexSE3Expmap*>(vPoseBase);
                Eigen::Matrix3d Kinv = pKFi->GetCamKinv();
                // ----------------------------
                // Create two depth vertices (FIXED)
                // ----------------------------
                VertexDepth* vD0 = new VertexDepth();
                vD0->setId(nextVid++);
                vD0->setEstimate(d0);
                vD0->setFixed(true);                 // ✅ Scheme A: FIXED
                if(!optimizer.addVertex(vD0))
                {
                    delete vD0;
                    continue;
                }

                VertexDepth* vD1 = new VertexDepth();
                vD1->setId(nextVid++);
                vD1->setEstimate(d1);
                vD1->setFixed(true);                 // ✅ Scheme A: FIXED
                if(!optimizer.addVertex(vD1))
                {
                    // 成对回滚：vD0 已经加入图，必须 remove 再 delete
                    optimizer.removeVertex(vD0);
                    delete vD0;
                    delete vD1;
                    continue;
                }
                // ---- edge endpoint 0 ----
                {
                    auto* e0 = new EdgePointToPluckerLinePoseAndDepthNew(
                        Eigen::Vector2d((double)sl[0], (double)sl[1]),
                        Kinv, Lw);
                    e0->setVertex(0, vPose);
                    e0->setVertex(1, vD0);
                    e0->setMeasurement(Eigen::Vector3d::Zero());
                    e0->setInformation(Eigen::Matrix3d::Identity());
                    e0->setLevel(0);
                    if(!optimizer.addEdge(e0))
                    {
                        delete e0;
                    }
                    else
                    {
                        vpEdgesLine.push_back(e0);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
                // ---- edge endpoint 1 ----
                {
                    auto* e1 = new EdgePointToPluckerLinePoseAndDepthNew(
                        Eigen::Vector2d((double)el[0], (double)el[1]),
                        Kinv, Lw);
                    e1->setVertex(0, vPose);
                    e1->setVertex(1, vD1);
                    e1->setMeasurement(Eigen::Vector3d::Zero());
                    e1->setInformation(Eigen::Matrix3d::Identity());
                    e1->setLevel(0);
                    if(!optimizer.addEdge(e1))
                    {
                        delete e1;
                    }
                    else
                    {
                        vpEdgesLine.push_back(e1);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
            } // end for each observation
        } // end for each mapline

        num_lines = lLocalMapLines.size();
        num_edges = nEdges + vpEdgesLine.size();
        
        CheckDuplicateVertexID(optimizer);
        // ----------------------------
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        double chi2_line_sum = 0;
        for(auto* e : vpEdgesLine)
            chi2_line_sum += e->chi2();
        std::cout << "[Line LBA] numEdges=" << vpEdgesLine.size()
          << " chi2_sum=" << chi2_line_sum << std::endl;

        std::cerr << "------------11111111111111111111---------------------" << std::endl;

        std::vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

        const double chi2_mono_thr   = CHI2_MONO_HARD   * value_scale;
        const double chi2_stereo_thr = CHI2_STEREO_HARD * value_scale;
        const double chi2_body_thr   = CHI2_MONO_HARD   * value_scale; // body 用 mono 阈值

        for(size_t i=0; i<vpEdgesMono.size(); ++i)
        {
            auto* e = vpEdgesMono[i]; auto* pMP = vpMapPointEdgeMono[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive());
            if(bad) vToErase.emplace_back(vpEdgeKFMono[i], pMP);
        }
        for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            //MapPoint* pMP = vpMapPointEdgeBody[i];
            // if(pMP->isBad())
            //     continue;
            // if(e->chi2()>chi2_body_thr || !e->isDepthPositive())
            // {
            //     KeyFrame* pKFi = vpEdgeKFBody[i];
            //     vToErase.push_back(make_pair(pKFi,pMP));
            // }
        }
        for(size_t i=0; i<vpEdgesStereo.size(); ++i)
        {
            auto* e = vpEdgesStereo[i]; auto* pMP = vpMapPointEdgeStereo[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) ;
            if(bad) vToErase.emplace_back(vpEdgeKFStereo[i], pMP);
        }
        std::cerr << "------------2222222222222222---------------------" << std::endl;
        // Check inlier observations for lines
        // --- Evaluate line endpoint observation outliers (conservative) ---
        vector<pair<KeyFrame*, MapLine*>> vLineObsToErase;
        for(size_t i=0; i<vpEdgesLine.size(); ++i)
        {
            auto* e = vpEdgesLine[i];
            if(e->chi2() > CHI2_LINE_HARD)
            {
                KeyFrame* pKFi = vpEdgeKFLine[i];
                MapLine*  pML  = vpMapLineEdge[i];
                if(pKFi && pML)
                {
                    pKFi->EraseMapLineMatch(pML);
                    pML->EraseLineObservation(pKFi);
                }       
            }
        }
        // for(size_t i = 0; i < vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     bool bad = false;
        //     // ---- Check depth (if depth vertex exists) ----
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d > MIN_DEPTH && d < MAX_DEPTH))
        //             bad = true;
        //     }
        //     // ---- chi2 check (2D error) ----
        //     if(e->chi2() > CHI2_LINE_HARD)
        //         bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }

        // for(size_t i=0; i<vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLine* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) continue;
        //     bool bad = false;
        //     // try to access depth associated to this VertexDepth
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d>MIN_DEPTH && d<MAX_DEPTH)) bad = true;
        //     }
        //     // chi2 check
        //     if(e->chi2() > CHI2_LINE_HARD) bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }
        //std::cerr << "------------333333333333333333333333---------------------" << std::endl;
        // Get Map Mutex
        //unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        // Lines: erase the offending observation only
        if(!vLineObsToErase.empty())
        {
            for(auto &pr : vLineObsToErase)
            {
                KeyFrame* pKFi = pr.first;
                MapLine* pMLi = pr.second;
                if(!pKFi || !pMLi) continue;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
            pKFi->SetPose(Tiw);
        }
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            if(vPoint)
            {
                pMP->SetWorldPos(vPoint->estimate().cast<float>());
                pMP->UpdateNormalAndDepth();
            }
        }
        // //update the optimization depth
        for(MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;
            std::vector<Eigen::Vector3d> all_pts;
            for(const auto& obs : pML->GetLineObservations())
            {
                KeyFrame* pKFi = obs.first;
                int idx = std::get<0>(obs.second);
                if(!pKFi || pKFi->isBad()) continue;
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
                float d0 = pML->GetObservationDepth0(pKFi, idx);
                float d1 = pML->GetObservationDepth1(pKFi, idx);
                if(!(d0>MIN_DEPTH && d1>MIN_DEPTH)) continue;
                Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(
                                Eigen::Vector2d(sl[0], sl[1]));
                Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(
                                Eigen::Vector2d(el[0], el[1]));
                Sophus::SE3f Tcw = pKFi->GetPose();
                // 1. 先算相机坐标系下的点
                Eigen::Vector3f pc0 = (float)d0 * r0.cast<float>();
                // 2. 再用 SE3 变换点
                Eigen::Vector3f pw0_f = Tcw.inverse() * pc0;
                Eigen::Vector3d pw0 = pw0_f.cast<double>();
                Eigen::Vector3f pc1 = (float)d1 * r1.cast<float>();
                Eigen::Vector3f pw1_f = Tcw.inverse() * pc1;
                Eigen::Vector3d pw1 = pw1_f.cast<double>();
                all_pts.push_back(pw0);
                all_pts.push_back(pw1);
            }

            if(all_pts.size() >= 2)
            {
                Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
                if(Lw.allFinite())
                    pML->SetPluckerLine(Lw);
            }
        }

        // // ====== 更新 MapLine 的深度 + 结构一致性检查 ======
        // for(MapLine* pML : lLocalMapLines)
        // {
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     for(const auto &obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad()) 
        //             continue;
        //         // --- 查 depth vertex ID ---
        //         size_t key = UtilSlam::MakeDepthKey(pML, pKFi, idx);
        //         auto it = depthIdMap.find(key);
        //         if(it == depthIdMap.end())
        //             continue;
        //         int idD0 = it->second.first;
        //         int idD1 = it->second.second;
        //         VertexDepth* vD0 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD0));
        //         VertexDepth* vD1 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD1));
        //         if(!vD0 || !vD1) 
        //             continue;
        //         double newd0 = vD0->estimate();
        //         double newd1 = vD1->estimate();
        //         // ---- 基础范围检查 ----
        //         bool accept0 = (newd0 > MIN_DEPTH && newd0 < MAX_DEPTH);
        //         bool accept1 = (newd1 > MIN_DEPTH && newd1 < MAX_DEPTH);
        //         // ---- chi2 检查（你的 edge 是 2 维） ----
        //         for(size_t ei=0; ei < vpEdgesLine.size(); ++ei)
        //         {
        //             if(vpEdgeKFLine[ei] != pKFi) continue;
        //             if(vpMapLineEdge[ei] != pML) continue;
        //             EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[ei];
        //             // edge 的 depth vertex 是 vertex(1) 或 vertex(2)
        //             VertexDepth* vD = nullptr;
        //             // 你只有一个 depth？
        //             // ——如果是双 depth，这里要判断 vertex(1) / vertex(2)
        //             vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //             if(vD)
        //             {
        //                 if(vD == vD0 && e->chi2() > CHI2_LINE_HARD)
        //                     accept0 = false;
        //                 if(vD == vD1 && e->chi2() > CHI2_LINE_HARD)
        //                     accept1 = false;
        //             }
        //         }
        //         // ---- 更新到 MapLine observation ----
        //         if(accept0)
        //             pML->SetObservationLineLsDepth(pKFi, idx, float(newd0));
        //         if(accept1)
        //             pML->SetObservationLineLeDepth(pKFi, idx, float(newd1));
        //     }
        //     // ------------------------------------------------------------------
        //     // (Ⅱ) 使用全部回投点检查线的结构一致性（PCA）
        //     // ------------------------------------------------------------------
        //     std::vector<Eigen::Vector3d> all_pts;
        //     all_pts.reserve(pML->GetLineObservations().size() * 2);
        //     for(const auto& obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad())
        //             continue;
        //         Eigen::Vector2f sl, el;
        //         if(!pKFi->GetLineEndPointEigen(idx, sl, el))
        //             continue;
        //         float d0 = pML->GetObservationDepth0(pKFi, idx);
        //         float d1 = pML->GetObservationDepth1(pKFi, idx);
        //         if(!(d0>MIN_DEPTH && d0<MAX_DEPTH && d1>MIN_DEPTH && d1<MAX_DEPTH))
        //             continue;
        //         Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1]));
        //         Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
        //         g2o::SE3Quat Tcw(pKFi->GetPose().unit_quaternion().cast<double>(),
        //                  pKFi->GetPose().translation().cast<double>());
        //         Eigen::Vector3d pw0 = Tcw.inverse() * (double(d0) * r0);
        //         Eigen::Vector3d pw1 = Tcw.inverse() * (double(d1) * r1);
        //         all_pts.push_back(pw0);
        //         all_pts.push_back(pw1);
        //     }
        //     double ratio = Converter::FirstPCVarianceRatio(all_pts);
        //     // // ------------------------------------------------------------------
        //     // // (Ⅲ) If inconsistent → remove the whole line
        //     // // ------------------------------------------------------------------
        //     // if(all_pts.size() >= 4 && ratio < LINE_COLINEARITY_LOW)
        //     // {
        //     //     std::vector<KeyFrame*> toErase;
        //     //     toErase.reserve(pML->GetLineObservations().size());
        //     //     for(const auto& obs : pML->GetLineObservations())
        //     //         toErase.push_back(obs.first);
        //     //     for(KeyFrame* kf : toErase)
        //     //     {
        //     //         if(kf)
        //     //         {
        //     //             kf->EraseMapLineMatch(pML);
        //     //             pML->EraseLineObservation(kf);
        //     //         }                       
        //     //     }
        //     //     pML->SetBadFlag();
        //     //     continue;
        //     // }
        //     // ------------------------------------------------------------------
        //     // (Ⅳ) 若线一致性高 → 重估 Plücker + 更新端点
        //     // ------------------------------------------------------------------
        //     if(all_pts.size() >= 2 && ratio > LINE_COLINEARITY_HIGH)
        //     {
        //         Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
        //         if(Lw.allFinite())
        //         {
        //             pML->SetPluckerLine(Lw);
        //             pML->UpdateWorldEndpointsFromObservationPntsAndPluckerLine(Lw, all_pts);
        //         }
        //     }
        // }

    }
    //update into Opr
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        opr.addKeyFrame((*lit));
    }
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        if(!(*lit) || (*lit)->isBad()) continue;
        MapPoint* pMP = *lit;
        if(!pMP->isRetrived())
        {
            pMP->setRetrived(true);
            opr.addMapPoint(*lit);
        }
        //else
        //replaceMapPoint(To do Next)
    }
    for(MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;
        if(!pML->isRetrived())
        {
            pML->setRetrived(true);
            opr.addMapLine(pML);
        }
        //else replaceMapLine(To do Next)
    }

}

//fixed clashing vertex id issue and the depth is optimized(but some depth vertices are too large/small, need to fix the issue, 
//(convert the depth and plucker line to two 3D endpoints and re-compute the plucker line from the two endpoints)? the process may be wrong)
void Optimizer::LocalBundleAdjustmentWithLinesPlucker_Depth_Alternating(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // --- This function is adapted from the original LocalBundleAdjustment ---
    // Goal: keep all existing KeyFrame and MapPoint vertices exactly as before
    // and *only add MapLine (Plucker) vertices + corresponding line projection edges*.
    // The original pose and point vertices are left unchanged.

    // Local KeyFrames: First Breath Search from Current Keyframe
    std::list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }
    // Local MapPoints seen in Local Keyframes
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        std::vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }
    // Local MapLines seen in Local Keyframes
    list<MapLine*> lLocalMapLines;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        vector<MapLine*> vpMLs = pKFi->GetMapLineMatches(); // <-- assumes KeyFrame::GetMapLineMatches() exists
        for(size_t i=0;i<vpMLs.size();i++){
            MapLine* pML = vpMLs[i];
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF!=pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints/MapLines but that are not Local Keyframes
    std::list<KeyFrame*> lFixedCameras;
    for(std::list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        std::map<KeyFrame*,std::tuple<int,int>> observations = (*lit)->GetObservations();
        for(std::map<KeyFrame*,std::tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // // Also consider MapLine observations when building fixed cameras
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // ---------------------------
    // Alternating loop parameters
    // ---------------------------
    double value_scale = 1.0;
    
    const int ALT_ITER = 1; // 2~4 typical
    // Outer alternating loop
    bool initial_iter_plucker_flag = true;
    for(int alt = 0; alt < ALT_ITER; alt++)
    {
        if(alt == 0)      value_scale = 3.0;
        else if(alt == 1) value_scale = 2.0;
        else              value_scale = 1.0;  // alt>=2
        // ----------------------------
        // (A) Fit Plücker lines using current poses + endpoints
        // ----------------------------
        if(initial_iter_plucker_flag)
        {
            for(MapLine* pML : lLocalMapLines)
            {
                std::vector<Eigen::Vector3d> pts_w;
                // Preferred: if MapLine stores world endpoints or cached 3D endpoints, use them:
                // Assume MapLine::GetAllWorldEndPoints() returns vector<Eigen::Vector3d> of world pts (all obs endpoints)
                if(pML->HasCachedWorldObservationLineEndPoints()) {
                    //pts_w = pML->GetAllWorldEndPoints(); // <-- you should implement next...
                } else {
                    // Fallback: for each observation, backproject endpoints using KeyFrame pose and stored per-observation depths
                    // Assumes MapLine stores per-observation endpoint depths: pML->GetObservationData(pKFi) -> {d0,d1}
                    for(auto& obs : pML->GetLineObservations())
                    {
                        KeyFrame* pKFi = obs.first;
                        int line_idx = get<0>(obs.second);
                        // Get pixel endpoints in that KF
                        Eigen::Vector2f sl, el;
                        if(!pKFi->GetLineEndPointEigen(line_idx, sl, el))
                            continue;
                        // Obtain per-observation depths or initial depths (you need to provide or compute these)
                        float d0 = pML->GetObservationDepth0(pKFi, line_idx); // <- implement or store initial depth
                        float d1 = pML->GetObservationDepth1(pKFi, line_idx);
                        // Backproject using intrinsics
                        Eigen::Vector3d ray0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1])); // implement or use K^-1 * [u,v,1]
                        Eigen::Vector3d ray1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
                        // World point = Twc * (d * ray)
                        g2o::SE3Quat Tcw = g2o::SE3Quat(pKFi->GetPose().unit_quaternion().cast<double>(), pKFi->GetPose().translation().cast<double>());
                        g2o::SE3Quat Twc = Tcw.inverse();
                        Eigen::Vector3d pw0 = Twc * (d0 * ray0);
                        Eigen::Vector3d pw1 = Twc * (d1 * ray1);
                        pts_w.push_back(pw0);
                        pts_w.push_back(pw1);
                    }
                }
                // If we have at least 2 points, fit
                if(pts_w.size() >= 2)
                {
                    Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(pts_w);
                    // Write back into MapLine temporary plucker estimate (do not yet commit endpoints)
                    pML->SetPluckerLine(Lw);
                }
            } // end for each mapline
            initial_iter_plucker_flag = false;
        }
            
        // ----------------------------
        // (B) Build g2o optimizer with MapPoints + Poses + Line vertices (line vertices setFixed(true))
        // and edges: original point edges + line projection edges (using observed image line abc).
        // Then optimize (poses and points will change; line vertices fixed).
        // ----------------------------
        g2o::SparseOptimizer optimizer;
        //optimizer.setVerbose(true);
        optimizer.clear();
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        if (pMap->IsInertial())
            solver->setUserLambdaInit(100.0);
        optimizer.setAlgorithm(solver);
        //optimizer.setVerbose(false);
        optimizer.setVerbose(true);
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Add Local KeyFrame vertices (same as original)
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }
        num_OptKF = lLocalKeyFrames.size();

        // Fixed Keyframes (same as original)
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Add MapPoint vertices (unchanged)
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();
        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;    //To do next
        vpEdgesBody.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nPoints = 0;
        int nEdges = 0;

        // Add point vertices and edges (copy from original LBA code)
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            if(!pMP)
            {
                continue;
            }
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;
            const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
            for(auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                KeyFrame* pKFi = mit->first;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);
                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;
                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        e->pCamera = pKFi->mpCamera;
                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        nEdges++;
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0) // Stereo
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;
                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }

         //Step 8: MapLine 顶点 + EdgePointToPluckerLine 边
        //-----------------------------
        // =====================================================
        // Step X: MapLine Depth Vertices + Point-to-Plücker Edges
        // (ORB-SLAM3 style, SAFE, Scheme A)
        // =====================================================

        // ---- containers (与 MapPoint LBA 对齐) ----
        std::vector<EdgePointToPluckerLinePoseAndDepthNew*> vpEdgesLine;
        vpEdgesLine.reserve(lLocalMapLines.size() * 20);
        std::vector<KeyFrame*> vpEdgeKFLine;
        vpEdgeKFLine.reserve(lLocalMapLines.size() * 20);
        std::vector<MapLine*>  vpMapLineEdge;
        vpMapLineEdge.reserve(lLocalMapLines.size() * 20);

        // 每一个 (MapLine*, KeyFrame*, endpointIndex) 对应一个 VertexDepth (idx 不是 line 的 index（idx），而是 endpoint。)
        std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*> depthVertexMap;

        // 外部 depth 更新要用到
        std::vector<VertexDepth*> vpDepthVerts;
        vpDepthVerts.reserve(lLocalMapLines.size()*20);
        std::unordered_map<VertexDepth*, double> depthPrior;
        depthPrior.reserve(lLocalMapLines.size()*20);
        //std::map<std::tuple<MapLine*, KeyFrame*, int, int>, VertexDepth*> depthVertexMap;
        // ---- 取得安全起始 vertex id（不要用 rbegin；ORB-SLAM3 这版 g2o 是 tr1 unordered_map）----
        int nextVid = GetMaxVertexId(optimizer) + 1;

        // ---- 可选：给 line depth vertex 留大间隔，避免与你 MapPoint vertex 发生任何潜在冲突 ----
        if(nextVid < (int)(maxKFid + 100000)) nextVid = (int)(maxKFid + 100000);

        // -------------------------------------------------
        // 遍历 Local MapLines
        // -------------------------------------------------
        // 在进入 lLocalMapLines 循环前做：
        // int nextObId = GetMaxVertexId(optimizer) + 1;

        for (MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;
            Eigen::Matrix<double,6,1> Lw = pML->GetPluckerLine().cast<double>();
            if(!Lw.allFinite()) continue;
            if(Lw.tail<3>().norm() < 1e-9) continue;
            const auto& obsList = pML->GetLineObservations();
            for(const auto& obsPair : obsList)
            {
                KeyFrame* pKFi = obsPair.first;
                if(!pKFi || pKFi->isBad()) continue;
                int idx = std::get<0>(obsPair.second);
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
                if(!sl.allFinite() || !el.allFinite()) continue;
                // depth init
                const float d0f = pML->GetObservationDepth0(pKFi, idx);
                const float d1f = pML->GetObservationDepth1(pKFi, idx);
                const double d0 = (double)d0f;
                const double d1 = (double)d1f;
                if(!IsFiniteDepth(d0) || !IsFiniteDepth(d1)) continue;
                // pose vertex must exist
                auto* vPoseBase = optimizer.vertex(pKFi->mnId);
                if(!vPoseBase) continue;
                // ✅ 不用 dynamic_cast（ORB-SLAM3 常见关闭 RTTI）
                auto* vPose = static_cast<g2o::VertexSE3Expmap*>(vPoseBase);
                Eigen::Matrix3d Kinv = pKFi->GetCamKinv();
                // ----------------------------
                // Create two depth vertices (FIXED)
                // ----------------------------
                // ---------- endpoint 0 ----------
                VertexDepth* vD0 = nullptr;
                auto key0 = std::make_tuple(pML, pKFi, 0);
                auto it0 = depthVertexMap.find(key0);
                if(it0 == depthVertexMap.end())
                {
                    vD0 = new VertexDepth();
                    vD0->setId(nextVid++);
                    vD0->setEstimate(d0);
                    vD0->setFixed(true);                 // ✅ 固定，避免进 block solver
                    if(!optimizer.addVertex(vD0)) { delete vD0; continue; }
                    depthVertexMap[key0] = vD0;
                    vpDepthVerts.push_back(vD0);
                    depthPrior[vD0] = d0;
                }
                else
                {
                    vD0 = it0->second;
                }

                // ---------- endpoint 1 ----------
                VertexDepth* vD1 = nullptr;
                auto key1 = std::make_tuple(pML, pKFi, 1);
                auto it1 = depthVertexMap.find(key1);
                if(it1 == depthVertexMap.end())
                {
                    vD1 = new VertexDepth();
                    vD1->setId(nextVid++);
                    vD1->setEstimate(d1);
                    vD1->setFixed(true);                 // ✅ 固定
                    if(!optimizer.addVertex(vD1)) { delete vD1; continue; }
                    depthVertexMap[key1] = vD1;
                    vpDepthVerts.push_back(vD1);
                    depthPrior[vD1] = d1;
                }
                else
                {
                    vD1 = it1->second;
                }
                // ----------------------------
                // ---- edge endpoint 0 ----
                {
                    auto* e0 = new EdgePointToPluckerLinePoseAndDepthNew(
                        Eigen::Vector2d((double)sl[0], (double)sl[1]),
                        Kinv, Lw);
                    e0->setVertex(0, vPose);
                    e0->setVertex(1, vD0);
                    e0->setMeasurement(Eigen::Vector3d::Zero());
                    e0->setInformation(Eigen::Matrix3d::Identity());
                    g2o::RobustKernelHuber* rk0 = new g2o::RobustKernelHuber;
                    e0->setRobustKernel(rk0);
                    rk0->setDelta(sqrt(CHI2_LINE_HARD));
                    e0->setLevel(0);
                    if(!optimizer.addEdge(e0))
                    {
                        delete e0;
                    }
                    else
                    {
                        vpEdgesLine.push_back(e0);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
                // ---- edge endpoint 1 ----
                {
                    auto* e1 = new EdgePointToPluckerLinePoseAndDepthNew(
                        Eigen::Vector2d((double)el[0], (double)el[1]),
                        Kinv, Lw);
                    e1->setVertex(0, vPose);
                    e1->setVertex(1, vD1);
                    e1->setMeasurement(Eigen::Vector3d::Zero());
                    e1->setInformation(Eigen::Matrix3d::Identity());
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e1->setRobustKernel(rk);
                    rk->setDelta(sqrt(CHI2_LINE_HARD));
                    e1->setLevel(0);
                    if(!optimizer.addEdge(e1))
                    {
                        delete e1;
                    }
                    else
                    {
                        vpEdgesLine.push_back(e1);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
            } // end for each observation
        } // end for each mapline

        num_lines = lLocalMapLines.size();
        num_edges = nEdges + vpEdgesLine.size();
        
        CheckDuplicateVertexID(optimizer);
        
        // 构建 depth prior ID map（vertex id -> prior）
        std::unordered_map<int, double> priorMap;
        priorMap.reserve(depthPrior.size());
        for (const auto& it : depthPrior)
        {
            if(it.first) {
                priorMap[it.first->id()] = it.second;
            }
        }

        // ----------------------------
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        // std::cerr << "------------0000000000000000000---------------------" << std::endl;
        // std::cerr << "Line LBA: after first opt: numEdges=" << vpEdgesLine.size() << std::endl;
        // std::cout << "Total Depth Vertices: " << vpDepthVerts.size() << std::endl;
        // for (VertexDepth* v : vpDepthVerts)
        //     std::cout << "VertexID: " << (v ? v->id() : -1) << std::endl;
        // for (auto* e : vpEdgesLine)
        // {
        //     if (!e || e->vertices().size() != 2 || !e->vertex(0) || !e->vertex(1)) {
        //         std::cerr << "[ERR] Invalid edge or missing vertices!" << std::endl;
        //             continue;
        //     }
        //     // Optionally check dynamic type too
        //     auto* casted = dynamic_cast<EdgePointToPluckerLinePoseAndDepthNew*>(e);
        //     if (!casted) {
        //         std::cerr << "[ERR] Edge type cast failed!" << std::endl;
        //         continue;
        //     }
        //     // Then safely use it
        //     casted->computeError();
        //     std::cerr << "Edge error after opt: " << casted->error().transpose() << std::endl;
        //     casted->linearizeOplus();
        //     std::cerr << "Edge chi2 after opt: " << casted->chi2() << std::endl;
        // }
        // ✅ external depth optimization (uses safe update)
        //UpdateDepthVertices_ExternalLM_Edge_Safe(vpEdgesLine, vpDepthVerts, /*prior=*/1e-2, /*prior_w=*/1e-2, /*iters=*/3, /*lambda_init=*/1e-3, MIN_DEPTH, MAX_DEPTH);

        // 替代原来的 ExternalLM 函数
        {
            std::unordered_map<int, VertexDepth*> id2newDepth;
            OptimizeDepthSeparately(vpEdgesLine, vpDepthVerts, id2newDepth, 1e-3, 5, MIN_DEPTH, MAX_DEPTH);
            WriteOptimizedDepthBack(id2newDepth, depthVertexMap);
        }

        double chi2_line_sum = 0;
        for(auto* e : vpEdgesLine)
            chi2_line_sum += e->chi2();
        std::cout << "[Line LBA] numEdges=" << vpEdgesLine.size()
          << " chi2_sum=" << chi2_line_sum << std::endl;

        std::cerr << "------------11111111111111111111---------------------" << std::endl;

        std::vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

        const double chi2_mono_thr   = CHI2_MONO_HARD   * value_scale;
        const double chi2_stereo_thr = CHI2_STEREO_HARD * value_scale;
        const double chi2_body_thr   = CHI2_MONO_HARD   * value_scale; // body 用 mono 阈值

        for(size_t i=0; i<vpEdgesMono.size(); ++i)
        {
            auto* e = vpEdgesMono[i]; auto* pMP = vpMapPointEdgeMono[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive());
            if(bad) vToErase.emplace_back(vpEdgeKFMono[i], pMP);
        }
        for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            //MapPoint* pMP = vpMapPointEdgeBody[i];
            // if(pMP->isBad())
            //     continue;
            // if(e->chi2()>chi2_body_thr || !e->isDepthPositive())
            // {
            //     KeyFrame* pKFi = vpEdgeKFBody[i];
            //     vToErase.push_back(make_pair(pKFi,pMP));
            // }
        }
        for(size_t i=0; i<vpEdgesStereo.size(); ++i)
        {
            auto* e = vpEdgesStereo[i]; auto* pMP = vpMapPointEdgeStereo[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) ;
            if(bad) vToErase.emplace_back(vpEdgeKFStereo[i], pMP);
        }
        std::cerr << "------------2222222222222222---------------------" << std::endl;
        // Check inlier observations for lines
        // --- Evaluate line endpoint observation outliers (conservative) ---
        vector<pair<KeyFrame*, MapLine*>> vLineObsToErase;
        for(size_t i=0; i<vpEdgesLine.size(); ++i)
        {
            auto* e = vpEdgesLine[i];
            if(e->chi2() > CHI2_LINE_HARD)
            {
                KeyFrame* pKFi = vpEdgeKFLine[i];
                MapLine*  pML  = vpMapLineEdge[i];
                if(pKFi && pML)
                {
                    pKFi->EraseMapLineMatch(pML);
                    pML->EraseLineObservation(pKFi);
                }       
            }
        }
        // for(size_t i = 0; i < vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     bool bad = false;
        //     // ---- Check depth (if depth vertex exists) ----
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d > MIN_DEPTH && d < MAX_DEPTH))
        //             bad = true;
        //     }
        //     // ---- chi2 check (2D error) ----
        //     if(e->chi2() > CHI2_LINE_HARD)
        //         bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }

        // for(size_t i=0; i<vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLine* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) continue;
        //     bool bad = false;
        //     // try to access depth associated to this VertexDepth
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d>MIN_DEPTH && d<MAX_DEPTH)) bad = true;
        //     }
        //     // chi2 check
        //     if(e->chi2() > CHI2_LINE_HARD) bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }
        

        //std::cerr << "------------333333333333333333333333---------------------" << std::endl;
        // Get Map Mutex
        //unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        // Lines: erase the offending observation only
        if(!vLineObsToErase.empty())
        {
            for(auto &pr : vLineObsToErase)
            {
                KeyFrame* pKFi = pr.first;
                MapLine* pMLi = pr.second;
                if(!pKFi || !pMLi) continue;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
        for(MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;

            for(const auto& obs : pML->GetLineObservations())
            {
                KeyFrame* pKFi = obs.first;
                int idx = std::get<0>(obs.second);
                if(!pKFi || pKFi->isBad()) continue;
                VertexDepth* vD0 = FindDepthVertex_SchemeB(pML, pKFi, 0, depthVertexMap);
                VertexDepth* vD1 = FindDepthVertex_SchemeB(pML, pKFi, 1, depthVertexMap);

                if(!vD0 || !vD1) continue;
                double d0 = vD0->estimate();
                double d1 = vD1->estimate();
                if(d0 > MIN_DEPTH && d0 < MAX_DEPTH)
                    pML->SetObservationLineLsDepth(pKFi, idx, float(d0));
                if(d1 > MIN_DEPTH && d1 < MAX_DEPTH)
                    pML->SetObservationLineLeDepth(pKFi, idx, float(d1));
            }
        }

        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
            pKFi->SetPose(Tiw);
        }
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            if(vPoint)
            {
                pMP->SetWorldPos(vPoint->estimate().cast<float>());
                pMP->UpdateNormalAndDepth();
            }
        }
        // //update the optimization depth
        for(MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;
            std::vector<Eigen::Vector3d> all_pts;
            for(const auto& obs : pML->GetLineObservations())
            {
                KeyFrame* pKFi = obs.first;
                int idx = std::get<0>(obs.second);
                if(!pKFi || pKFi->isBad()) continue;
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
                float d0 = pML->GetObservationDepth0(pKFi, idx);
                float d1 = pML->GetObservationDepth1(pKFi, idx);
                if(!(d0>MIN_DEPTH && d1>MIN_DEPTH)) continue;
                Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(
                                Eigen::Vector2d(sl[0], sl[1]));
                Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(
                                Eigen::Vector2d(el[0], el[1]));
                Sophus::SE3f Tcw = pKFi->GetPose();
                // 1. 先算相机坐标系下的点
                Eigen::Vector3f pc0 = (float)d0 * r0.cast<float>();
                // 2. 再用 SE3 变换点
                Eigen::Vector3f pw0_f = Tcw.inverse() * pc0;
                Eigen::Vector3d pw0 = pw0_f.cast<double>();
                Eigen::Vector3f pc1 = (float)d1 * r1.cast<float>();
                Eigen::Vector3f pw1_f = Tcw.inverse() * pc1;
                Eigen::Vector3d pw1 = pw1_f.cast<double>();
                all_pts.push_back(pw0);
                all_pts.push_back(pw1);
            }

            if(all_pts.size() >= 2)
            {
                Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
                if(Lw.allFinite())
                {
                    pML->SetPluckerLine(Lw);
                    pML->UpdateWorldEndpointsFromObservationPntsAndPluckerLine(Lw, all_pts);
                }
            }
        }

        // // ====== 更新 MapLine 的深度 + 结构一致性检查 ======
        // for(MapLine* pML : lLocalMapLines)
        // {
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     for(const auto &obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad()) 
        //             continue;
        //         // --- 查 depth vertex ID ---
        //         size_t key = UtilSlam::MakeDepthKey(pML, pKFi, idx);
        //         auto it = depthIdMap.find(key);
        //         if(it == depthIdMap.end())
        //             continue;
        //         int idD0 = it->second.first;
        //         int idD1 = it->second.second;
        //         VertexDepth* vD0 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD0));
        //         VertexDepth* vD1 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD1));
        //         if(!vD0 || !vD1) 
        //             continue;
        //         double newd0 = vD0->estimate();
        //         double newd1 = vD1->estimate();
        //         // ---- 基础范围检查 ----
        //         bool accept0 = (newd0 > MIN_DEPTH && newd0 < MAX_DEPTH);
        //         bool accept1 = (newd1 > MIN_DEPTH && newd1 < MAX_DEPTH);
        //         // ---- chi2 检查（你的 edge 是 2 维） ----
        //         for(size_t ei=0; ei < vpEdgesLine.size(); ++ei)
        //         {
        //             if(vpEdgeKFLine[ei] != pKFi) continue;
        //             if(vpMapLineEdge[ei] != pML) continue;
        //             EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[ei];
        //             // edge 的 depth vertex 是 vertex(1) 或 vertex(2)
        //             VertexDepth* vD = nullptr;
        //             // 你只有一个 depth？
        //             // ——如果是双 depth，这里要判断 vertex(1) / vertex(2)
        //             vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //             if(vD)
        //             {
        //                 if(vD == vD0 && e->chi2() > CHI2_LINE_HARD)
        //                     accept0 = false;
        //                 if(vD == vD1 && e->chi2() > CHI2_LINE_HARD)
        //                     accept1 = false;
        //             }
        //         }
        //         // ---- 更新到 MapLine observation ----
        //         if(accept0)
        //             pML->SetObservationLineLsDepth(pKFi, idx, float(newd0));
        //         if(accept1)
        //             pML->SetObservationLineLeDepth(pKFi, idx, float(newd1));
        //     }
        //     // ------------------------------------------------------------------
        //     // (Ⅱ) 使用全部回投点检查线的结构一致性（PCA）
        //     // ------------------------------------------------------------------
        //     std::vector<Eigen::Vector3d> all_pts;
        //     all_pts.reserve(pML->GetLineObservations().size() * 2);
        //     for(const auto& obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad())
        //             continue;
        //         Eigen::Vector2f sl, el;
        //         if(!pKFi->GetLineEndPointEigen(idx, sl, el))
        //             continue;
        //         float d0 = pML->GetObservationDepth0(pKFi, idx);
        //         float d1 = pML->GetObservationDepth1(pKFi, idx);
        //         if(!(d0>MIN_DEPTH && d0<MAX_DEPTH && d1>MIN_DEPTH && d1<MAX_DEPTH))
        //             continue;
        //         Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1]));
        //         Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
        //         g2o::SE3Quat Tcw(pKFi->GetPose().unit_quaternion().cast<double>(),
        //                  pKFi->GetPose().translation().cast<double>());
        //         Eigen::Vector3d pw0 = Tcw.inverse() * (double(d0) * r0);
        //         Eigen::Vector3d pw1 = Tcw.inverse() * (double(d1) * r1);
        //         all_pts.push_back(pw0);
        //         all_pts.push_back(pw1);
        //     }
        //     double ratio = Converter::FirstPCVarianceRatio(all_pts);
        //     // // ------------------------------------------------------------------
        //     // // (Ⅲ) If inconsistent → remove the whole line
        //     // // ------------------------------------------------------------------
        //     // if(all_pts.size() >= 4 && ratio < LINE_COLINEARITY_LOW)
        //     // {
        //     //     std::vector<KeyFrame*> toErase;
        //     //     toErase.reserve(pML->GetLineObservations().size());
        //     //     for(const auto& obs : pML->GetLineObservations())
        //     //         toErase.push_back(obs.first);
        //     //     for(KeyFrame* kf : toErase)
        //     //     {
        //     //         if(kf)
        //     //         {
        //     //             kf->EraseMapLineMatch(pML);
        //     //             pML->EraseLineObservation(kf);
        //     //         }                       
        //     //     }
        //     //     pML->SetBadFlag();
        //     //     continue;
        //     // }
        //     // ------------------------------------------------------------------
        //     // (Ⅳ) 若线一致性高 → 重估 Plücker + 更新端点
        //     // ------------------------------------------------------------------
        //     if(all_pts.size() >= 2 && ratio > LINE_COLINEARITY_HIGH)
        //     {
        //         Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
        //         if(Lw.allFinite())
        //         {
        //             pML->SetPluckerLine(Lw);
        //             pML->UpdateWorldEndpointsFromObservationPntsAndPluckerLine(Lw, all_pts);
        //         }
        //     }
        // }

    }
    //update into Opr
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        opr.addKeyFrame((*lit));
    }
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        if(!(*lit) || (*lit)->isBad()) continue;
        MapPoint* pMP = *lit;
        if(!pMP->isRetrived())
        {
            pMP->setRetrived(true);
            opr.addMapPoint(*lit);
        }
        //else
        //replaceMapPoint(To do Next)
    }
    for(MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;
        if(!pML->isRetrived())
        {
            pML->setRetrived(true);
            opr.addMapLine(pML);
        }
        //else replaceMapLine(To do Next)
    }

}


void Optimizer::OptimizeDepthSeparately(
    std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
    std::vector<VertexDepth*>& vpDepthVerts,
    std::unordered_map<int, VertexDepth*>& id2newDepth_out,
    double lambda,
    int maxIter,
    double min_depth,
    double max_depth)
{
    using namespace g2o;

    std::cout << "=== [DepthOnlyOptimizer] Begin ===" << std::endl;

    // 1. 构建优化器（6 DOF pose + 1 DOF depth）
    typedef BlockSolver<BlockSolverTraits<6,1>> Block;
    Block::LinearSolverType* linearSolver = new LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(linearSolver);
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(solver_ptr);
    
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 2. 构建新的 VertexDepth
    std::unordered_map<int, VertexDepth*> id2newDepth;

    for (VertexDepth* vOld : vpDepthVerts)
    {
        if (!vOld) continue;

        VertexDepth* vNew = new VertexDepth();
        vNew->setId(vOld->id());
        vNew->setEstimate(vOld->estimate());
        vNew->setFixed(false);
        optimizer.addVertex(vNew);
        id2newDepth[vOld->id()] = vNew;
    }

    // 3. 构建新的 pose vertex（只复制一次）
    std::unordered_map<int, g2o::VertexSE3Expmap*> id2pose;
    int edge_id = 1000000;

    for (const auto* eOld : vpEdgesLine)
    {
        if (!eOld || !eOld->vertex(0) || !eOld->vertex(1)) continue;

        const auto* vPoseOld  = static_cast<const g2o::VertexSE3Expmap*>(eOld->vertex(0));
        const auto* vDepthOld = static_cast<const VertexDepth*>(eOld->vertex(1));
        if (!vPoseOld || !vDepthOld) continue;

        int pose_id = vPoseOld->id();
        int depth_id = vDepthOld->id();

        g2o::VertexSE3Expmap* vPoseNew = nullptr;
        if (id2pose.find(pose_id) == id2pose.end())
        {
            vPoseNew = new g2o::VertexSE3Expmap();
            vPoseNew->setId(pose_id);
            vPoseNew->setEstimate(vPoseOld->estimate());
            vPoseNew->setFixed(true);  // pose 不优化
            optimizer.addVertex(vPoseNew);
            id2pose[pose_id] = vPoseNew;
        }
        else
        {
            vPoseNew = id2pose[pose_id];
        }

        VertexDepth* vDepthNew = id2newDepth[depth_id];
        if (!vDepthNew) continue;

        // 新建 edge
        auto* eNew = new EdgePointToPluckerLinePoseAndDepthNew(
            eOld->GetPixel(), eOld->GetKinv(), eOld->GetPlucker());

        eNew->setId(edge_id++);
        eNew->setVertex(0, vPoseNew);
        eNew->setVertex(1, vDepthNew);
        eNew->setMeasurement(Eigen::Vector3d::Zero());
        eNew->setInformation(Eigen::Matrix3d::Identity());

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(5.991);  // 2D chi2 threshold
        eNew->setRobustKernel(rk);

        optimizer.addEdge(eNew);
    }

    // 4. 运行优化器
    optimizer.initializeOptimization();
    optimizer.optimize(maxIter);

    // 5. 打印优化结果
    for (auto& it : id2newDepth)
    {
        VertexDepth* v = it.second;
        if (!v) continue;
        double d = v->estimate();
        d = std::max(min_depth, std::min(max_depth, d));
        std::cout << "[DepthOnly] VertexID=" << v->id() << " d=" << d << std::endl;
    }

    // ========== 输出结果 ==========
    id2newDepth_out = std::move(id2newDepth);
    std::cout << "=== [DepthOnlyOptimizer] End ===" << std::endl;
}


void Optimizer::WriteOptimizedDepthBack(
    const std::unordered_map<int, VertexDepth*>& id2newDepth,
    const std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*>& depthVertexMap)
{
    for (const auto& kv : depthVertexMap)
    {
        const auto& key = kv.first;
        MapLine* pML = std::get<0>(key);
        KeyFrame* pKF = std::get<1>(key);
        int endpoint = std::get<2>(key); // 0 or 1

        if (!pML || !pKF) continue;

        VertexDepth* oldV = kv.second;
        if (!oldV) continue;

        int vid = oldV->id();
        auto itNew = id2newDepth.find(vid);
        if (itNew == id2newDepth.end()) continue;

        double new_d = itNew->second->estimate();
        if (!(new_d > MIN_DEPTH && new_d < MAX_DEPTH)) continue;

        std::tuple<int,int> idx_pair = pML->GetIndexInKeyFrame(pKF);
        int idx = std::get<0>(idx_pair); // 你自己封装一下这个接口
        if (idx < 0) continue;

        if (endpoint == 0)
            pML->SetObservationLineLsDepth(pKF, idx, float(new_d));
        else
            pML->SetObservationLineLeDepth(pKF, idx, float(new_d));
    }
}

void Optimizer::UpdateDepthVertices_ExternalLM_Edge_Safe(
    const std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
    const std::vector<VertexDepth*> &vpDepthVerts,
    double prior,
    double prior_w,
    int iters,
    double lambda_init,
    double min_depth,
    double max_depth)
{
    std::cout << "=== Begin Safe External Depth Optimization ===" << std::endl;

    // Create a map: depth vertex ID -> list of connected edges
    std::unordered_map<int, std::vector<EdgePointToPluckerLinePoseAndDepthNew*>> depthEdgeMap;

    for (auto* edge : vpEdgesLine)
    {
        if (!edge) continue;

        const auto* vDepth = static_cast<const VertexDepth*>(edge->vertex(1));
        if (!vDepth) continue;

        int id = vDepth->id();
        depthEdgeMap[id].push_back(edge);
    }

    for (int iter = 0; iter < iters; ++iter)
    {
        std::cout << "--- Iteration " << iter << " ---" << std::endl;

        for (auto* vDepth : vpDepthVerts)
        {
            if (!vDepth) continue;
            int id = vDepth->id();

            auto it = depthEdgeMap.find(id);
            if (it == depthEdgeMap.end()) continue;

            const auto& edges = it->second;

            double H = 0.0;
            double b = 0.0;

            for (auto* edge : edges)
            {
                if (!edge || !edge->vertex(0) || !edge->vertex(1)) continue;

                try {
                    edge->computeError();
                    edge->linearizeOplus();
                } catch (...) {
                    std::cerr << "[EXCEPTION] edge compute/linearize failed for depthID=" << id << std::endl;
                    continue;
                }

                const Eigen::Vector3d r = edge->ErrorVec();
                const Eigen::Matrix<double,3,1> J = edge->JacobianDepth();
                const Eigen::Matrix3d info = edge->InfoMat();

                double Jtr = J.transpose() * info * r;
                double JtJ = J.transpose() * info * J;

                b += -Jtr;
                H += JtJ;
            }

            if (H <= 1e-12) {
                std::cerr << "[WARN] Skipping vertex " << id << " due to degenerate Hessian." << std::endl;
                continue;
            }

            double step = b / (H + lambda_init);
            double d0 = vDepth->estimate();
            double d_new = UtilSlam::ClampD_ZDG(d0 + step, min_depth, max_depth);
            vDepth->setEstimate(d_new);

            std::cout << "[DEPTH] Vertex " << id
                      << " d0=" << d0
                      << " step=" << step
                      << " d_new=" << d_new << std::endl;
        }
    }

    std::cout << "=== End External Depth Optimization ===" << std::endl;
}


void Optimizer::UpdateDepthVertices_ExternalLM_Safe_Verts(
    const std::vector<VertexDepth*>& depthVerts,
    const std::unordered_map<int, double>& priorMap,
    double prior_w,
    int iters,
    double lambda_init,
    double min_depth,
    double max_depth)
{
    double lambda = lambda_init;

    for (int it = 0; it < iters; ++it)
    {
        for (VertexDepth* vD : depthVerts)
        {
            if (!vD) continue;

            const int vid = vD->id();
            const double d0 = vD->estimate();

            double H = 0.0;
            double b = 0.0;

            // 遍历连接此 vertex 的所有边
            const auto& edges = vD->edges();
            for (auto* edge_base : edges)
            {
                
                // 安全 static_cast，因为你构图时确定只有这种 edge 类型
                auto* edge = static_cast<EdgePointToPluckerLinePoseAndDepthNew*>(edge_base);
                if (!edge || edge->vertex(1) != vD) continue;

                // 保守：确保 vD->estimate() 是当前值
                vD->setEstimate(d0);

                edge->computeError();
                edge->linearizeOplus();

                const Eigen::Vector3d r = edge->ErrorVec();
                const Eigen::Matrix<double, 3, 1> Jd = edge->JacobianDepth();
                const Eigen::Matrix3d W = edge->InfoMat();

                const double JtWJ = (Jd.transpose() * W * Jd)(0, 0);
                const double JtWr = (Jd.transpose() * W * r)(0, 0);

                H += JtWJ;
                b += JtWr;
            }

            // 加入 depth prior
            auto it_prior = priorMap.find(vid);
            if (it_prior != priorMap.end())
            {
                const double d_prior = it_prior->second;
                H += prior_w;
                b += prior_w * (d0 - d_prior);
            }

            // 加入阻尼项（防止病态）
            H += lambda;

            // 避免除以 0 或数值不稳定
            if (!std::isfinite(H) || std::abs(H) < 1e-12)
                continue;

            const double step = -b / H;
            double d1 = d0 + step;

            // 保守检查
            if (!std::isfinite(d1))
                continue;

            // 限制范围
            d1 = UtilSlam::ClampD_ZDG(d1, min_depth, max_depth);

            // 限制跳变
            if (std::abs(step) > 0.5 * std::max(1.0, std::abs(d0)))
                continue;

            vD->setEstimate(d1);
        }
    }
}


void Optimizer::UpdateDepthVertices_ExternalLM_Safe(
    g2o::SparseOptimizer& optimizer,
    const std::unordered_map<int, double>& priorMap,  // use vertex ID -> prior
    double prior_w,
    int iters,
    double lambda_init,
    double min_depth,
    double max_depth)
{
    double lambda = lambda_init;

    for(int it = 0; it < iters; ++it)
    {
        for(const auto& v_pair : optimizer.vertices())
        {
            g2o::HyperGraph::Vertex* hv = v_pair.second;
            auto* v = dynamic_cast<g2o::OptimizableGraph::Vertex*>(hv);
            if (!v) continue;

            VertexDepth* vD = dynamic_cast<VertexDepth*>(v);
            if (!vD) continue;

            const int vid = vD->id();

            const double d0 = vD->estimate();

            double H = 0.0;
            double b = 0.0;

            // Collect connected edges
            const auto& edges = vD->edges();
            for(auto* e : edges)
            {
                if(!e) continue;

                // We only want depth-related edges
                EdgePointToPluckerLinePoseAndDepthNew* edge = dynamic_cast<EdgePointToPluckerLinePoseAndDepthNew*>(e);
                if(!edge) continue;

                // vD is vertex(1)
                if(edge->vertex(1) != vD) continue;

                // Compute residuals and jacobians
                vD->setEstimate(d0);  // reset
                edge->computeError();
                edge->linearizeOplus();

                const Eigen::Vector3d r = edge->ErrorVec();
                const Eigen::Matrix<double, 3, 1> Jd = edge->JacobianDepth();
                const Eigen::Matrix3d W = edge->InfoMat();

                const double JtWJ = (Jd.transpose() * W * Jd)(0, 0);
                const double JtWr = (Jd.transpose() * W * r)(0, 0);

                H += JtWJ;
                b += JtWr;
            }

            // Add prior term if available
            auto it_prior = priorMap.find(vid);
            if(it_prior != priorMap.end())
            {
                const double d_prior = it_prior->second;
                H += prior_w;
                b += prior_w * (d0 - d_prior);
            }

            H += lambda;

            if(!std::isfinite(H) || std::abs(H) < 1e-12)
                continue;

            const double step = - b / H;
            double d1 = d0 + step;

            if(!std::isfinite(d1)) continue;

            d1 = UtilSlam::ClampD_ZDG(d1, min_depth, max_depth);

            // Reject crazy jump (optional)
            if(std::abs(step) > 0.5 * std::max(1.0, std::abs(d0)))
                continue;

            vD->setEstimate(d1);
        }
    }
}


void Optimizer::UpdateDepthVertices_ExternalLM(
    const std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
    const std::vector<VertexDepth*>& vpDepthVerts,
    const std::unordered_map<VertexDepth*, double>& priorMap, // d_prior（可为空）
    double prior_w,
    int iters,
    double lambda_init,
    double min_depth,
    double max_depth)
{
    if(vpEdgesLine.empty() || vpDepthVerts.empty()) return;

    // depth -> edges adjacency
    std::unordered_map<VertexDepth*, std::vector<EdgePointToPluckerLinePoseAndDepthNew*>> adj;
    adj.reserve(vpDepthVerts.size()*2);

    for(auto* e : vpEdgesLine)
    {
        if(!e) continue;
        auto* vD = static_cast<VertexDepth*>(e->vertex(1));
        if(!vD) continue;
        adj[vD].push_back(e);
    }

    double lambda = lambda_init;

    for(int it=0; it<iters; ++it)
    {
        for(VertexDepth* vD : vpDepthVerts)
        {
            if(!vD) continue;
            auto itAdj = adj.find(vD);
            if(itAdj == adj.end()) continue;

            const double d0 = vD->estimate();
            double H = 0.0;
            double b = 0.0;

            // accumulate from connected edges
            for(auto* e : itAdj->second)
            {
                // compute r and J
                vD->setEstimate(d0);
                e->computeError();
                e->linearizeOplus();

                const Eigen::Vector3d r = e->ErrorVec();             // 3x1
                const Eigen::Matrix<double,3,1> Jd = e->JacobianDepth(); // 3x1
                const Eigen::Matrix3d W = e->InfoMat();              // 3x3

                const double JtWJ = (Jd.transpose() * W * Jd)(0,0);
                const double JtWr = (Jd.transpose() * W * r)(0,0);

                H += JtWJ;
                b += JtWr;
            }

            // prior: prior_w * (d - d_prior)^2
            auto itP = priorMap.find(vD);
            if(itP != priorMap.end())
            {
                const double d_prior = itP->second;
                H += prior_w;
                b += prior_w * (d0 - d_prior);
            }

            // LM damping
            H += lambda;

            if(!std::isfinite(H) || std::abs(H) < 1e-12) continue;

            const double step = - b / H;
            double d1 = d0 + step;

            if(!std::isfinite(d1)) continue;
            d1 = UtilSlam::ClampD_ZDG(d1, min_depth, max_depth);

            // optional: reject crazy jump
            // if(std::abs(step) > 0.5 * std::max(1.0, d0)) continue;

            vD->setEstimate(d1);
        }

        // simplest: keep lambda fixed (most stable)
        // lambda *= 0.5;
    }
}

/// ← 线在该 KeyFrame 中的 index // endpoint ∈ {0,1}
VertexDepth* Optimizer::FindDepthVertex_SchemeB(
    MapLine* pML,
    KeyFrame* pKFi,
    int endpoint,
    const std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*>& depthVertexMap)
{
    const auto key = std::make_tuple(pML, pKFi, endpoint);
    auto it = depthVertexMap.find(key);
    return (it == depthVertexMap.end()) ? nullptr : it->second;
}


VertexDepth* Optimizer::CreateDepthVertex(
    g2o::SparseOptimizer& optimizer,
    int& nextVid,
    double initDepth)
{
    VertexDepth* vD = new VertexDepth();
    vD->setId(nextVid++);
    vD->setEstimate(initDepth);
    vD->setFixed(false);

    if(!optimizer.addVertex(vD))
    {
        delete vD;
        return nullptr;
    }
    return vD;
}


#if 0

// VertexDepth* Optimizer::FindDepthVertex_SchemeB(
//     MapLine*  pML,
//     KeyFrame* pKFi,
//     int       lineIdx,   // ← 线在该 KeyFrame 中的 index
//     int       endpoint)  // endpoint ∈ {0,1}
// {
//     // ----------------------------
//     // 1. 构造唯一 key
//     // ----------------------------
//     const std::tuple<MapLine*, KeyFrame*, int, int> key(
//         pML, pKFi, lineIdx, endpoint);

//     // ----------------------------
//     // 2. 查表
//     // ----------------------------
//     auto it = mDepthVertexMap.find(key);
//     if(it == mDepthVertexMap.end())
//     {
//         // Scheme B 设计约定：
//         //   - depth vertex 只在建图阶段创建
//         //   - 查不到说明该观测已被剔除或逻辑错误
//         return nullptr;
//     }

//     return it->second;
// }



void Optimizer::LocalBundleAdjustmentWithLinesPlucker_Alternating(
    KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // --- This function is adapted from the original LocalBundleAdjustment ---
    // Goal: keep all existing KeyFrame and MapPoint vertices exactly as before
    // and *only add MapLine (Plucker) vertices + corresponding line projection edges*.
    // The original pose and point vertices are left unchanged.

    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }
    // Local MapPoints seen in Local Keyframes
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        if(pKFi->mnId==pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
        }
    }
    // Local MapLines seen in Local Keyframes
    list<MapLine*> lLocalMapLines;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        vector<MapLine*> vpMLs = pKFi->GetMapLineMatches(); // <-- assumes KeyFrame::GetMapLineMatches() exists
        for(size_t i=0;i<vpMLs.size();i++){
            MapLine* pML = vpMLs[i];
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF!=pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints/MapLines but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId )
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // // Also consider MapLine observations when building fixed cameras
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // ---------------------------
    // Alternating loop parameters
    // ---------------------------
    double value_scale = 1.0;
    
    const int ALT_ITER = 1; // 2~4 typical
    // Outer alternating loop
    bool initial_iter_plucker_flag = true;
    for(int alt = 0; alt < ALT_ITER; alt++)
    {
        if(alt == 0)      value_scale = 3.0;
        else if(alt == 1) value_scale = 2.0;
        else              value_scale = 1.0;  // alt>=2
        // ----------------------------
        // (A) Fit Plücker lines using current poses + endpoints
        // ----------------------------
        if(initial_iter_plucker_flag)
        {
            for(MapLine* pML : lLocalMapLines)
            {
                std::vector<Eigen::Vector3d> pts_w;
                // Preferred: if MapLine stores world endpoints or cached 3D endpoints, use them:
                // Assume MapLine::GetAllWorldEndPoints() returns vector<Eigen::Vector3d> of world pts (all obs endpoints)
                if(pML->HasCachedWorldObservationLineEndPoints()) {
                    //pts_w = pML->GetAllWorldEndPoints(); // <-- you should implement next...
                } else {
                    // Fallback: for each observation, backproject endpoints using KeyFrame pose and stored per-observation depths
                    // Assumes MapLine stores per-observation endpoint depths: pML->GetObservationData(pKFi) -> {d0,d1}
                    for(auto& obs : pML->GetLineObservations())
                    {
                        KeyFrame* pKFi = obs.first;
                        int line_idx = get<0>(obs.second);
                        // Get pixel endpoints in that KF
                        Eigen::Vector2f sl, el;
                        if(!pKFi->GetLineEndPointEigen(line_idx, sl, el))
                            continue;
                        // Obtain per-observation depths or initial depths (you need to provide or compute these)
                        float d0 = pML->GetObservationDepth0(pKFi, line_idx); // <- implement or store initial depth
                        float d1 = pML->GetObservationDepth1(pKFi, line_idx);
                        // Backproject using intrinsics
                        Eigen::Vector3d ray0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1])); // implement or use K^-1 * [u,v,1]
                        Eigen::Vector3d ray1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
                        // World point = Twc * (d * ray)
                        g2o::SE3Quat Tcw = g2o::SE3Quat(pKFi->GetPose().unit_quaternion().cast<double>(), pKFi->GetPose().translation().cast<double>());
                        g2o::SE3Quat Twc = Tcw.inverse();
                        Eigen::Vector3d pw0 = Twc * (d0 * ray0);
                        Eigen::Vector3d pw1 = Twc * (d1 * ray1);
                        pts_w.push_back(pw0);
                        pts_w.push_back(pw1);
                    }
                }
                // If we have at least 2 points, fit
                if(pts_w.size() >= 2)
                {
                    Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(pts_w);
                    // Write back into MapLine temporary plucker estimate (do not yet commit endpoints)
                    pML->SetPluckerLine(Lw);
                }
            } // end for each mapline
            initial_iter_plucker_flag = false;
        }
            
        // ----------------------------
        // (B) Build g2o optimizer with MapPoints + Poses + Line vertices (line vertices setFixed(true))
        // and edges: original point edges + line projection edges (using observed image line abc).
        // Then optimize (poses and points will change; line vertices fixed).
        // ----------------------------
        g2o::SparseOptimizer optimizer;
        //optimizer.setVerbose(true);
        optimizer.clear();
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        if (pMap->IsInertial())
            solver->setUserLambdaInit(100.0);
        optimizer.setAlgorithm(solver);
        //optimizer.setVerbose(false);
        optimizer.setVerbose(true);
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Add Local KeyFrame vertices (same as original)
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }
        num_OptKF = lLocalKeyFrames.size();

        // Fixed Keyframes (same as original)
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Add MapPoint vertices (unchanged)
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();
        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;    //To do next
        vpEdgesBody.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nPoints = 0;
        int nEdges = 0;

        // Add point vertices and edges (copy from original LBA code)
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            if(!pMP)
            {
                continue;
            }
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;
            const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
            for(auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                KeyFrame* pKFi = mit->first;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);
                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;
                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);
                        e->pCamera = pKFi->mpCamera;
                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        nEdges++;
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0) // Stereo
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);
                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);
                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;
                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }

         //Step 8: MapLine 顶点 + EdgePointToPluckerLine 边
        //-----------------------------
        //std::vector<EdgePointToPluckerLinePoseAndDepth*> vpEdgesLine;
        std::vector<EdgePointToPluckerLinePoseAndDepthNew*> vpEdgesLine;
        std::vector<VertexDepth*> vDepths;
        std::vector<KeyFrame*> vpEdgeKFLine;
        std::vector<MapLine*> vpMapLineEdge;
        // Compute a safe offset for line vertex ids so they do not collide with point ids used above
        int maxMapPointId = 0;
        for(list<MapPoint*>::iterator mit=lLocalMapPoints.begin(), mend=lLocalMapPoints.end(); mit!=mend; mit++){
            if((*mit)->mnId > maxMapPointId) maxMapPointId = (*mit)->mnId;
        }
        int lineIdOffset = static_cast<int>(maxKFid) + maxMapPointId + 100000; // +2 safety
        int nLines = 0;
        //int maxMapLineId = 0;
        // for(list<MapLine*>::iterator mit=lLocalMapLines.begin(), mend=lLocalMapLines.end(); mit!=mend; mit++)
        // {
        //     if((*mit)->mnId > maxMapLineId)
        //     {
        //         maxMapLineId = (*mit)->mnId;
        //     }
        // }
        // maxMapLineId++;
        // int max_all_offset = lineIdOffset + maxMapLineId;
        // now start allocating new ids(depth d0, d1) from max_all_offset + 1
        int nextObId = lineIdOffset;
        // map to store the depth vertex ids for each (MapLine, KeyFrame, idx)
        std::unordered_map<size_t, std::pair<int,int>> depthIdMap; // key as hash of tuple, value pair<idD0,idD1>
#if 1   //有bug，这块，可能是不能求导
        
        for (MapLine* pML : lLocalMapLines)
        {
            if(!pML || pML->isBad()) continue;
            Eigen::Matrix<double,6,1> Lw = pML->GetPluckerLine().cast<double>();
            if(!Lw.allFinite()) continue;
            if(Lw.tail<3>().norm() < 1e-9) continue;
            const auto& obsList = pML->GetLineObservations();
            for(const auto& obsPair : obsList)
            {
                KeyFrame* pKFi = obsPair.first;
                if(!pKFi || pKFi->isBad()) continue;
                int idx = std::get<0>(obsPair.second);
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;
                if(!sl.allFinite() || !el.allFinite()) continue;
                float d0 = pML->GetObservationDepth0(pKFi, idx);
                float d1 = pML->GetObservationDepth1(pKFi, idx);
                // ---------------- Depth vertices ----------------
                int idD0 = nextObId++;
                int idD1 = nextObId++;
                std::cerr << "idD0, idD1: " << idD0 << ", " << idD1 << std::endl;
                VertexDepth* vD0 = nullptr;
                VertexDepth* vD1 = nullptr;
                // create & add vD0 if not exists
                if(optimizer.vertex(idD0) == nullptr) {
                    vD0 = new VertexDepth();
                    vD0->setEstimate(d0);
                    vD0->setId(idD0);
                    optimizer.addVertex(vD0);
                    vDepths.push_back(vD0);
                    //std::cerr << "[LBA] add depth vertex idD0=" << idD0 << " est=" << d0 << std::endl;
                } else {
                    // shouldn't happen due to allocateNextFreeId, 但做保险
                    std::cerr << "[LBA] vertex idD0 already exists: " << idD0 << std::endl;
                }
                // create & add vD1 if not exists
                if(optimizer.vertex(idD1) == nullptr) {
                    vD1 = new VertexDepth();
                    vD1->setEstimate(d1);
                    vD1->setId(idD1);
                    optimizer.addVertex(vD1);
                    vDepths.push_back(vD1);
                    //std::cerr << "[LBA] add depth vertex idD1=" << idD1 << " est=" << d1 << std::endl;
                } else {
                    std::cerr << "[LBA] vertex idD1 already exists: " << idD1 << std::endl;
                }
                // depth key
                size_t key = UtilSlam::MakeDepthKey(pML, pKFi, idx);
                if(depthIdMap.count(key)) {
                    std::cerr << "[LBA] Duplicate depth key detected!\n";
                }
                depthIdMap[key] = std::make_pair(idD0, idD1);
                Eigen::Matrix3d Kinv = pKFi->GetCamKinv();
                // Pose 顶点，从 optimizer 里获取并确保存在且类型正确
                auto* vPoseBase = optimizer.vertex(pKFi->mnId);
                auto* vPose = dynamic_cast<g2o::VertexSE3Expmap*>(vPoseBase);
                if(!vPose) {
                    std::cerr << "[LBA] WARNING: pose vertex not found or wrong type for KF " << pKFi->mnId << std::endl;
                    // 跳过添加边，但不要删除已添加的 depth 顶点（它们将由调用者统一管理/清理）
                    continue;
                }
                // =====================================================
                // Edge for endpoint 0
                // =====================================================
                // --------- 添加边（endpoint 0） ----------
                if(optimizer.vertex(idD0))
                {
                    auto* vertexDepthPtr = optimizer.vertex(idD0);
                    // 类型校验：确保这是 VertexDepth（防止类型错误的 id 被误用）
                    if(dynamic_cast<VertexDepth*>(vertexDepthPtr) == nullptr) {
                        std::cerr << "[LBA] ERROR: optimizer.vertex("<<idD0<<") is not VertexDepth type\n";
                    } else {
                        EdgePointToPluckerLinePoseAndDepthNew* e0 =
                            new EdgePointToPluckerLinePoseAndDepthNew(Eigen::Vector2d(sl[0], sl[1]), Kinv, Lw);
                        e0->setVertex(0, vPose);
                        e0->setVertex(1, vertexDepthPtr);
                        e0->setMeasurement(Eigen::Vector3d::Zero());
                        // measurement dim == 3 -> 信息矩阵 应 为 3x3
                        e0->setInformation(Eigen::Matrix3d::Identity());
                        // Robust kernel 可选（有助于数值稳定）
                        // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        // e0->setRobustKernel(rk);
                        // rk->setDelta(some_value);
                        //std::cerr << "[LBA] adding edge e0 idD0=" << idD0 << " KF=" << pKFi->mnId << std::endl;
                        optimizer.addEdge(e0);
                        vpEdgesLine.push_back(e0);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
                // =====================================================
                // Edge for endpoint 1
                // =====================================================
                if(optimizer.vertex(idD1))
                {
                    auto* vertexDepthPtr = optimizer.vertex(idD1);
                    if(dynamic_cast<VertexDepth*>(vertexDepthPtr) == nullptr) {
                        std::cerr << "[LBA] ERROR: optimizer.vertex("<<idD1<<") is not VertexDepth type\n";
                    } else {
                        EdgePointToPluckerLinePoseAndDepthNew* e1 =
                            new EdgePointToPluckerLinePoseAndDepthNew(Eigen::Vector2d(el[0], el[1]), Kinv, Lw);
                        e1->setVertex(0, vPose);
                        e1->setVertex(1, vertexDepthPtr);
                        e1->setMeasurement(Eigen::Vector3d::Zero());
                        e1->setInformation(Eigen::Matrix3d::Identity());
                        //std::cerr << "[LBA] adding edge e1 idD1=" << idD1 << " KF=" << pKFi->mnId << std::endl;
                        optimizer.addEdge(e1);
                        vpEdgesLine.push_back(e1);
                        vpEdgeKFLine.push_back(pKFi);
                        vpMapLineEdge.push_back(pML);
                    }
                }
            }
        }

        std::cerr << "------------00000000000000000000000---------------------" << std::endl;
        // ----------------------------
        // Optimize (poses + points). Lines are fixed.
#endif
        
        CheckDuplicateVertexID(optimizer);
        // ----------------------------
        optimizer.initializeOptimization();
        optimizer.optimize(1);

        std::cerr << "------------11111111111111111111---------------------" << std::endl;

        std::vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesBody.size()+vpEdgesStereo.size());

        const double chi2_mono_thr   = CHI2_MONO_HARD   * value_scale;
        const double chi2_stereo_thr = CHI2_STEREO_HARD * value_scale;
        const double chi2_body_thr   = CHI2_MONO_HARD   * value_scale; // body 用 mono 阈值

        for(size_t i=0; i<vpEdgesMono.size(); ++i)
        {
            auto* e = vpEdgesMono[i]; auto* pMP = vpMapPointEdgeMono[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_mono_thr) || (!e->isDepthPositive());
            if(bad) vToErase.emplace_back(vpEdgeKFMono[i], pMP);
        }
        for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
            //MapPoint* pMP = vpMapPointEdgeBody[i];
            // if(pMP->isBad())
            //     continue;
            // if(e->chi2()>chi2_body_thr || !e->isDepthPositive())
            // {
            //     KeyFrame* pKFi = vpEdgeKFBody[i];
            //     vToErase.push_back(make_pair(pKFi,pMP));
            // }
        }
        for(size_t i=0; i<vpEdgesStereo.size(); ++i)
        {
            auto* e = vpEdgesStereo[i]; auto* pMP = vpMapPointEdgeStereo[i];
            if(!pMP || pMP->isBad()) continue;
            //bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) || (e->predictedDepth() < MIN_DEPTH);
            bool bad = (e->chi2() > chi2_stereo_thr) || (!e->isDepthPositive()) ;
            if(bad) vToErase.emplace_back(vpEdgeKFStereo[i], pMP);
        }
        std::cerr << "------------2222222222222222---------------------" << std::endl;
        // Check inlier observations for lines
        // --- Evaluate line endpoint observation outliers (conservative) ---
        vector<pair<KeyFrame*, MapLine*>> vLineObsToErase;
        // for(size_t i = 0; i < vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     bool bad = false;
        //     // ---- Check depth (if depth vertex exists) ----
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d > MIN_DEPTH && d < MAX_DEPTH))
        //             bad = true;
        //     }
        //     // ---- chi2 check (2D error) ----
        //     if(e->chi2() > CHI2_LINE_HARD)
        //         bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }

        // for(size_t i=0; i<vpEdgesLine.size(); ++i)
        // {
        //     EdgePointToPluckerLine* e = vpEdgesLine[i];
        //     MapLine* pML = vpMapLineEdge[i];
        //     KeyFrame* pKFi = vpEdgeKFLine[i];
        //     if(!pML || pML->isBad()) continue;
        //     bool bad = false;
        //     // try to access depth associated to this VertexDepth
        //     VertexDepth* vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //     if(vD)
        //     {
        //         double d = vD->estimate();
        //         if(!(d>MIN_DEPTH && d<MAX_DEPTH)) bad = true;
        //     }
        //     // chi2 check
        //     if(e->chi2() > CHI2_LINE_HARD) bad = true;
        //     if(bad)
        //         vLineObsToErase.emplace_back(pKFi, pML);
        // }
        //std::cerr << "------------333333333333333333333333---------------------" << std::endl;
        // Get Map Mutex
        //unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        // Lines: erase the offending observation only
        if(!vLineObsToErase.empty())
        {
            for(auto &pr : vLineObsToErase)
            {
                KeyFrame* pKFi = pr.first;
                MapLine* pMLi = pr.second;
                if(!pKFi || !pMLi) continue;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseLineObservation(pKFi);
            }
        }
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
            pKFi->SetPose(Tiw);
        }
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            if(vPoint)
            {
                pMP->SetWorldPos(vPoint->estimate().cast<float>());
                pMP->UpdateNormalAndDepth();
            }
        }
        // //update the optimization depth
        // // ====== 更新 MapLine 的深度 + 结构一致性检查 ======
        // for(MapLine* pML : lLocalMapLines)
        // {
        //     if(!pML || pML->isBad()) 
        //         continue;
        //     for(const auto &obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad()) 
        //             continue;
        //         // --- 查 depth vertex ID ---
        //         size_t key = UtilSlam::MakeDepthKey(pML, pKFi, idx);
        //         auto it = depthIdMap.find(key);
        //         if(it == depthIdMap.end())
        //             continue;
        //         int idD0 = it->second.first;
        //         int idD1 = it->second.second;
        //         VertexDepth* vD0 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD0));
        //         VertexDepth* vD1 = dynamic_cast<VertexDepth*>(optimizer.vertex(idD1));
        //         if(!vD0 || !vD1) 
        //             continue;
        //         double newd0 = vD0->estimate();
        //         double newd1 = vD1->estimate();
        //         // ---- 基础范围检查 ----
        //         bool accept0 = (newd0 > MIN_DEPTH && newd0 < MAX_DEPTH);
        //         bool accept1 = (newd1 > MIN_DEPTH && newd1 < MAX_DEPTH);
        //         // ---- chi2 检查（你的 edge 是 2 维） ----
        //         for(size_t ei=0; ei < vpEdgesLine.size(); ++ei)
        //         {
        //             if(vpEdgeKFLine[ei] != pKFi) continue;
        //             if(vpMapLineEdge[ei] != pML) continue;
        //             EdgePointToPluckerLinePoseAndDepth* e = vpEdgesLine[ei];
        //             // edge 的 depth vertex 是 vertex(1) 或 vertex(2)
        //             VertexDepth* vD = nullptr;
        //             // 你只有一个 depth？
        //             // ——如果是双 depth，这里要判断 vertex(1) / vertex(2)
        //             vD = dynamic_cast<VertexDepth*>(e->vertex(1));
        //             if(vD)
        //             {
        //                 if(vD == vD0 && e->chi2() > CHI2_LINE_HARD)
        //                     accept0 = false;
        //                 if(vD == vD1 && e->chi2() > CHI2_LINE_HARD)
        //                     accept1 = false;
        //             }
        //         }
        //         // ---- 更新到 MapLine observation ----
        //         if(accept0)
        //             pML->SetObservationLineLsDepth(pKFi, idx, float(newd0));
        //         if(accept1)
        //             pML->SetObservationLineLeDepth(pKFi, idx, float(newd1));
        //     }
        //     // ------------------------------------------------------------------
        //     // (Ⅱ) 使用全部回投点检查线的结构一致性（PCA）
        //     // ------------------------------------------------------------------
        //     std::vector<Eigen::Vector3d> all_pts;
        //     all_pts.reserve(pML->GetLineObservations().size() * 2);
        //     for(const auto& obs : pML->GetLineObservations())
        //     {
        //         KeyFrame* pKFi = obs.first;
        //         int idx = get<0>(obs.second);
        //         if(!pKFi || pKFi->isBad())
        //             continue;
        //         Eigen::Vector2f sl, el;
        //         if(!pKFi->GetLineEndPointEigen(idx, sl, el))
        //             continue;
        //         float d0 = pML->GetObservationDepth0(pKFi, idx);
        //         float d1 = pML->GetObservationDepth1(pKFi, idx);
        //         if(!(d0>MIN_DEPTH && d0<MAX_DEPTH && d1>MIN_DEPTH && d1<MAX_DEPTH))
        //             continue;
        //         Eigen::Vector3d r0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1]));
        //         Eigen::Vector3d r1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
        //         g2o::SE3Quat Tcw(pKFi->GetPose().unit_quaternion().cast<double>(),
        //                  pKFi->GetPose().translation().cast<double>());
        //         Eigen::Vector3d pw0 = Tcw.inverse() * (double(d0) * r0);
        //         Eigen::Vector3d pw1 = Tcw.inverse() * (double(d1) * r1);
        //         all_pts.push_back(pw0);
        //         all_pts.push_back(pw1);
        //     }
        //     double ratio = Converter::FirstPCVarianceRatio(all_pts);
        //     // // ------------------------------------------------------------------
        //     // // (Ⅲ) If inconsistent → remove the whole line
        //     // // ------------------------------------------------------------------
        //     // if(all_pts.size() >= 4 && ratio < LINE_COLINEARITY_LOW)
        //     // {
        //     //     std::vector<KeyFrame*> toErase;
        //     //     toErase.reserve(pML->GetLineObservations().size());
        //     //     for(const auto& obs : pML->GetLineObservations())
        //     //         toErase.push_back(obs.first);
        //     //     for(KeyFrame* kf : toErase)
        //     //     {
        //     //         if(kf)
        //     //         {
        //     //             kf->EraseMapLineMatch(pML);
        //     //             pML->EraseLineObservation(kf);
        //     //         }                       
        //     //     }
        //     //     pML->SetBadFlag();
        //     //     continue;
        //     // }
        //     // ------------------------------------------------------------------
        //     // (Ⅳ) 若线一致性高 → 重估 Plücker + 更新端点
        //     // ------------------------------------------------------------------
        //     if(all_pts.size() >= 2 && ratio > LINE_COLINEARITY_HIGH)
        //     {
        //         Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(all_pts);
        //         if(Lw.allFinite())
        //         {
        //             pML->SetPluckerLine(Lw);
        //             pML->UpdateWorldEndpointsFromObservationPntsAndPluckerLine(Lw, all_pts);
        //         }
        //     }
        // }

    }
    //update into Opr
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        opr.addKeyFrame((*lit));
    }
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        if(!(*lit) || (*lit)->isBad()) continue;
        MapPoint* pMP = *lit;
        if(!pMP->isRetrived())
        {
            pMP->setRetrived(true);
            opr.addMapPoint(*lit);
        }
        //else
        //replaceMapPoint(To do Next)
    }
    for(MapLine* pML : lLocalMapLines)
    {
        if(!pML || pML->isBad()) continue;
        if(!pML->isRetrived())
        {
            pML->setRetrived(true);
            opr.addMapLine(pML);
        }
        //else replaceMapLine(To do Next)
    }

}

#endif

void Optimizer::OptimizeOneIterationLocalBundleAdjustmentLinesPlucker(KeyFrame *pKF,
    bool* pbStopFlag,
    Map* pMap,
    int num_iter,
    std::list<KeyFrame*> pLocalKeyFrames,
    std::list<MapPoint*> pLocalMapPoints,
    std::list<MapLine*> pLocalMapLines,
    std::list<KeyFrame*> pFixedCameras,
    int& num_fixedKF,
    int& num_OptKF,
    int& num_MPs,
    int& num_lines,
    int& num_edges)
{
    Map* pCurrentMap = pKF->GetMap();
    // Outer alternating loop
    for(int alt = 0; alt < num_iter; alt++)
    {
        // ----------------------------
        // (A) Fit Plücker lines using current poses + endpoints
        // ----------------------------
        for(MapLine* pML : pLocalMapLines)
        {
            std::vector<Eigen::Vector3d> pts_w;
            // Preferred: if MapLine stores world endpoints or cached 3D endpoints, use them:
            // Assume MapLine::GetAllWorldEndPoints() returns vector<Eigen::Vector3d> of world pts (all obs endpoints)
            if(pML->HasCachedWorldObservationLineEndPoints()) {
                //pts_w = pML->GetAllWorldEndPoints(); // <-- you should implement next...
            } else {
                // Fallback: for each observation, backproject endpoints using KeyFrame pose and stored per-observation depths
                // Assumes MapLine stores per-observation endpoint depths: pML->GetObservationData(pKFi) -> {d0,d1}
                for(auto& obs : pML->GetLineObservations())
                {
                    KeyFrame* pKFi = obs.first;
                    int line_idx = get<0>(obs.second);
                    // Get pixel endpoints in that KF
                    Eigen::Vector2f sl, el;
                    if(!pKFi->GetLineEndPointEigen(line_idx, sl, el))
                        continue;
                    // Obtain per-observation depths or initial depths (you need to provide or compute these)
                    float d0 = pML->GetObservationDepth0(pKFi, line_idx); // <- implement or store initial depth
                    float d1 = pML->GetObservationDepth1(pKFi, line_idx);
                    // Backproject using intrinsics
                    Eigen::Vector3d ray0 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(sl[0], sl[1])); // implement or use K^-1 * [u,v,1]
                    Eigen::Vector3d ray1 = pKFi->UnprojectToNormalizedPlane(Eigen::Vector2d(el[0], el[1]));
                    // World point = Twc * (d * ray)
                    g2o::SE3Quat Tcw = g2o::SE3Quat(pKFi->GetPose().unit_quaternion().cast<double>(), pKFi->GetPose().translation().cast<double>());
                    g2o::SE3Quat Twc = Tcw.inverse();
                    Eigen::Vector3d pw0 = Twc * (d0 * ray0);
                    Eigen::Vector3d pw1 = Twc * (d1 * ray1);
                    pts_w.push_back(pw0);
                    pts_w.push_back(pw1);
                }
            }
            // If we have at least 2 points, fit
            if(pts_w.size() >= 2)
            {
                Eigen::Matrix<double,6,1> Lw = Converter::FitPluckerLineFromPoints(pts_w);
                // Write back into MapLine temporary plucker estimate (do not yet commit endpoints)
                pML->SetPluckerLine(Lw);
            }
        } // end for each mapline


        // ----------------------------
        // (B) Build g2o optimizer with MapPoints + Poses + Line vertices (line vertices setFixed(true))
        // and edges: original point edges + line projection edges (using observed image line abc).
        // Then optimize (poses and points will change; line vertices fixed).
        // ----------------------------
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        if (pMap->IsInertial())
            solver->setUserLambdaInit(100.0);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);
        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Add Local KeyFrame vertices (same as original)
        for(list<KeyFrame*>::iterator lit=pLocalKeyFrames.begin(), lend=pLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }
        num_OptKF = pLocalKeyFrames.size();

        // Fixed Keyframes (same as original)
        for(list<KeyFrame*>::iterator lit=pFixedCameras.begin(), lend=pFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            Sophus::SE3<float> Tcw = pKFi->GetPose();
            vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Add MapPoint vertices (unchanged)
        const int nExpectedSize = (pLocalKeyFrames.size()+pFixedCameras.size())*pLocalMapPoints.size();
        vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);
        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);
        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        int nPoints = 0;
        int nEdges = 0;

        // Add point vertices and edges (copy from original LBA code)
        for(list<MapPoint*>::iterator lit=pLocalMapPoints.begin(), lend=pLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            nPoints++;

            const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
            for(auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                KeyFrame* pKFi = mit->first;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                {
                    const int leftIndex = get<0>(mit->second);
                    // Monocular observation
                    if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0)
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->pCamera = pKFi->mpCamera;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);

                        nEdges++;
                    }
                    else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]>=0) // Stereo
                    {
                        const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }

         //Step 8: MapLine 顶点 + EdgePointToPluckerLine 边
        //-----------------------------
        std::vector<EdgePointToPluckerLine*> vpEdgesLine;
        std::vector<VertexDepth*> vDepths;
        std::vector<KeyFrame*> vpEdgeKFLine;
        std::vector<MapLine*> vpMapLineEdge;
        // Compute a safe offset for line vertex ids so they do not collide with point ids used above
        int maxMapPointId = 0;
        for(list<MapPoint*>::iterator mit=pLocalMapPoints.begin(), mend=pLocalMapPoints.end(); mit!=mend; mit++){
            if((*mit)->mnId > maxMapPointId) maxMapPointId = (*mit)->mnId;
        }
        int lineIdOffset = static_cast<int>(maxKFid) + maxMapPointId + 2; // +2 safety
        int nLines = 0;
        
        for(MapLine* pML : pLocalMapLines)
        {
            ORB_SLAM3::VertexLinePlucker* vLine = new ORB_SLAM3::VertexLinePlucker();
            Eigen::Matrix<double,6,1> Lw =  pML->GetPluckerLine();
            std::cerr << "vLine->Lw: " << Lw.transpose() << std::endl;
            vLine->setEstimate(pML->GetPluckerLine());  //获取plucker的公式 //这个很重要
            //打印出来，用于处理数据
            int idLine = pML->mnId + lineIdOffset;
            vLine->setId(idLine);
            vLine->setFixed(true);
            optimizer.addVertex(vLine);
            nLines++;
            // 外层 Alternating Loop 的策略：本轮固定 Line
            // 2) 为此 MapLine 的每个观测添加 Depth-Vertex + EdgePointToPluckerLine
            const auto& obsList = pML->GetLineObservations();  // map< KeyFrame*, tuple<int,int> >
            for(const auto& obsPair : obsList)
            {
                KeyFrame* pKFi = obsPair.first;
                if(!pKFi || pKFi->isBad()) continue;
                int idx = std::get<0>(obsPair.second);
                Eigen::Vector2f sl, el;
                if(!pKFi->GetLineEndPointEigen(idx, sl, el)) continue;

                // 3D 点深度（可以从前一轮估计或初始化）
                float d0 = pML->GetObservationDepth0(pKFi, idx);
                float d1 = pML->GetObservationDepth1(pKFi, idx);    //to copy from initial depth image
                //------------------------------------------------------------------
                // A) 为两个 endpoint 分别创建深度顶点 VertexDepth
                //------------------------------------------------------------------
                VertexDepth* vD0 = new VertexDepth();
                vD0->setEstimate(d0);
                int idD0 = idLine + 20000 + idx*2;
                vD0->setId(idD0);
                optimizer.addVertex(vD0);
                VertexDepth* vD1 = new VertexDepth();
                vD1->setEstimate(d1);
                int idD1 = idLine + 20000 + idx*2 + 1;
                vD1->setId(idD1);
                optimizer.addVertex(vD1);
                vDepths.push_back(vD0);
                vDepths.push_back(vD1);

                //------------------------------------------------------------------
                // B) 第一个端点 p0 的 EdgePointToPluckerLine
                //------------------------------------------------------------------
                Eigen::Matrix3d kinv = pKFi->GetCamKinv();
                EdgePointToPluckerLine* e0 = new EdgePointToPluckerLine(Eigen::Vector2d(sl[0], sl[1]), kinv);
                e0->setVertex(0, optimizer.vertex(pKFi->mnId));      // pose
                e0->setVertex(1, vD0);                               // depth
                e0->setVertex(2, vLine);                             // plucker line (fixed in this iteration)
                e0->setMeasurement(Eigen::Vector3d::Zero());
                e0->setInformation(Eigen::Matrix3d::Identity());
                optimizer.addEdge(e0);
                vpEdgesLine.push_back(e0);
                vpEdgeKFLine.push_back(pKFi);
                vpMapLineEdge.push_back(pML);
                //------------------------------------------------------------------
                // C) 第二个端点 p1 的 EdgePointToPluckerLine
                //------------------------------------------------------------------
                EdgePointToPluckerLine* e1 = new EdgePointToPluckerLine(Eigen::Vector2d(el[0], el[1]), kinv);

                e1->setVertex(0, optimizer.vertex(pKFi->mnId));      // pose
                e1->setVertex(1, vD1);                               // depth
                e1->setVertex(2, vLine);                             // plucker line
                e1->setMeasurement(Eigen::Vector3d::Zero());
                e1->setInformation(Eigen::Matrix3d::Identity());

                optimizer.addEdge(e1);
                vpEdgesLine.push_back(e1);
                vpEdgeKFLine.push_back(pKFi);
                vpMapLineEdge.push_back(pML);
            }
        }

        // ----------------------------
        // Optimize (poses + points). Lines are fixed.
        // ----------------------------
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        for(list<KeyFrame*>::iterator lit=pLocalKeyFrames.begin(), lend=pLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
            pKFi->SetPose(Tiw);
        }

        for(list<MapPoint*>::iterator lit=pLocalMapPoints.begin(), lend=pLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            if(vPoint)
            {
                pMP->SetWorldPos(vPoint->estimate().cast<float>());
                pMP->UpdateNormalAndDepth();
                if(!pMP->isRetrived())
                {
                    pMP->setRetrived(true);
                }
            }
        }

        //To do next...
    
    }


}



void Optimizer::LocalBundleAdjustmentWithLinesPluckerBack(
    KeyFrame *pKF, 
    bool* pbStopFlag, 
    Map* pMap, 
    int& num_fixedKF, 
    int& num_OptKF, 
    int& num_MPs, 
    int& num_lines,
    int& num_edges,
    MappingOperation& opr)
{
    // -----------------------------
    // Step 1: 局部关键帧 BFS
    // -----------------------------
    std::list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map* pCurrentMap = pKF->GetMap();

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(KeyFrame* pKFi : vNeighKFs)
    {
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            pKFi->mnBALocalForKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    // -----------------------------
    // Step 2: 局部 MapPoint
    // -----------------------------
    num_fixedKF = 0;
    list<MapPoint*> lLocalMapPoints;
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        for(MapPoint* pMP : pKFi->GetMapPointMatches())
        {
            if(pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap)
            {
                if(pMP->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapPoints.push_back(pMP);
                    pMP->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }
    // -----------------------------
    // Step 3: 局部 MapLine
    // -----------------------------
    list<MapLine*> lLocalMapLines;
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        for(MapLine* pML : pKFi->GetMapLineMatches())
        {
            if(pML && !pML->isBad() && pML->GetMap() == pCurrentMap)
            {
                if(pML->mnBALocalForKF != pKF->mnId)
                {
                    lLocalMapLines.push_back(pML);
                    pML->mnBALocalForKF = pKF->mnId;
                }
            }
        }
    }
    // -----------------------------
    // Step 4: Fixed KeyFrames
    // -----------------------------
    list<KeyFrame*> lFixedCameras;
    for(MapPoint* pMP : lLocalMapPoints)
    {
        for(auto& obs : pMP->GetObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    for(MapLine* pML : lLocalMapLines)
    {
        for(auto& obs : pML->GetLineObservations())
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    num_fixedKF = lFixedCameras.size();
    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LBA: No fixed keyframes, abort", Verbose::VERBOSITY_NORMAL);
        return;
    }
    // -----------------------------
    // Step 5: 构建优化器
    // -----------------------------
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    if(pMap->IsInertial())
        solver->setUserLambdaInit(100.0);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // -----------------------------
    // Step 6: 添加关键帧顶点
    // -----------------------------
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3f Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid) maxKFid = pKFi->mnId;
    }
    for(KeyFrame* pKFi : lFixedCameras)
    {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3f Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid) maxKFid = pKFi->mnId;
    }

    // prepare id allocator and maps
    unsigned long currentId = maxKFid;
    std::unordered_map<int,int> mapPointId2G2oId;
    std::unordered_map<int,int> mapLineId2G2oId;

    // -----------------------------
    // Step 7: MapPoint 顶点 + 边
    // -----------------------------
    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vector<KeyFrame*> vpEdgeKFMono;
    vector<MapPoint*> vpMapPointEdgeMono;
    int nEdges = 0;

    for(MapPoint* pMP : lLocalMapPoints)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        currentId++;
        int g2oPointId = (int)currentId;
        vPoint->setId(g2oPointId);
        mapPointId2G2oId[pMP->mnId] = g2oPointId;
        //vPoint->setId(pMP->mnId + maxKFid + 1);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        for(auto& obs : pMP->GetObservations())
        {
            //mono
            KeyFrame* pKFi = obs.first;
            if(pKFi->isBad()) continue;
            const int idx = get<0>(obs.second);
            if(idx < 0) continue;
            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[idx];
            Eigen::Vector2d meas(kpUn.pt.x, kpUn.pt.y);
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
            e->setVertex(0, vPoint);
            e->setVertex(1, optimizer.vertex(pKFi->mnId));
            e->setMeasurement(meas);
            // 必须加上这个，否则 computeError 会崩溃 
            e->pCamera = pKFi->mpCamera;
            e->setInformation(Eigen::Matrix2d::Identity() * pKFi->mvInvLevelSigma2[kpUn.octave]);
            //e->setInformation(Eigen::Matrix2d::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            e->setRobustKernel(rk);
            rk->setDelta(sqrt(5.991));
            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vpEdgeKFMono.push_back(pKFi);
            vpMapPointEdgeMono.push_back(pMP);
            nEdges++;
        }
    }
    // -----------------------------
    // Step 8: MapLine 顶点 + EdgeSE3ProjectPluckerLine_PoseAndLine 边
    // -----------------------------
    // std::vector<EdgeSE3ProjectPluckerLine_PoseAndLine*> vpEdgesLine;
    // std::vector<KeyFrame*> vpEdgeKFLine;
    // std::vector<MapLine*> vpMapLineEdge;
    // for(MapLine* pML : lLocalMapLines)
    // {
    //     VertexLinePlucker* vLine = new VertexLinePlucker();
    //     vLine->setEstimate(pML->GetPluckerLine());
    //     currentId++;
    //     int g2oLineId = (int)currentId;
    //     vLine->setId(g2oLineId);
    //     mapLineId2G2oId[pML->mnId] = g2oLineId;
    //     //vLine->setId(pML->mnId + maxKFid + 1 + lLocalMapPoints.size());
    //     vLine->setMarginalized(true);
    //     optimizer.addVertex(vLine);
    //     for(auto& obs : pML->GetLineObservations())
    //     {
    //         KeyFrame* pKFi = obs.first;
    //         if(pKFi->isBad()) continue;
    //         EdgeSE3ProjectPluckerLine_PoseAndLine* e = new EdgeSE3ProjectPluckerLine_PoseAndLine();
    //         e->setVertex(0, optimizer.vertex(pKFi->mnId));
    //         e->setVertex(1, vLine);
    //         e->SetCameraIntrinsics(pKFi->fx, pKFi->fy, pKFi->cx, pKFi->cy);
    //         Eigen::Vector3f ob_line_projected_pnt = pML->GetProjectedLineABC(pKFi);
    //         e->SetObservedLineABC(ob_line_projected_pnt[0],ob_line_projected_pnt[1], ob_line_projected_pnt[2]);
    //         g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
    //         e->setRobustKernel(rk);
    //         rk->setDelta(3.0);
    //         optimizer.addEdge(e);
    //         vpEdgesLine.push_back(e);
    //         vpEdgeKFLine.push_back(pKFi);
    //         vpMapLineEdge.push_back(pML);
    //         nEdges++;
    //     }
    // }
    num_MPs = lLocalMapPoints.size();
    num_lines = lLocalMapLines.size();
    num_edges = nEdges;

    // -----------------------------
    // Step 9: 优化
    // -----------------------------
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    {
    std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
    // -----------------------------
    // Step 10: 剔除 MapPoint + MapLine Outlier
    // -----------------------------
    for(size_t i=0; i<vpEdgesMono.size(); i++)
    {
        auto e = vpEdgesMono[i];
        auto pMP = vpMapPointEdgeMono[i];
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            pKFi->EraseMapPointMatch(pMP);
            pMP->EraseObservation(pKFi);
        }
    }

    // for(size_t i=0; i<vpEdgesLine.size(); i++)
    // {
    //     auto e = vpEdgesLine[i];
    //     auto pML = vpMapLineEdge[i];
    //     if(e->chi2()>9.0)
    //     {
    //         KeyFrame* pKFi = vpEdgeKFLine[i];
    //         pKFi->EraseMapLineMatch(pML);
    //         pML->EraseLineObservation(pKFi);
    //     }
    // }
    // -----------------------------
    // Step 11: 更新关键帧 + MapPoint + MapLine
    // -----------------------------
    for(KeyFrame* pKFi : lLocalKeyFrames)
    {
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(vSE3->estimate().rotation().cast<float>(), vSE3->estimate().translation().cast<float>());
        pKFi->SetPose(Tcw);
        opr.addKeyFrame(pKFi);
    }
    for(MapPoint* pMP : lLocalMapPoints)
    {
        int g2oId = mapPointId2G2oId[pMP->mnId];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(g2oId));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
        if(!pMP->isRetrived())
        {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }
    // for(MapLine* pML : lLocalMapLines)
    // {
    //     int g2oId = mapLineId2G2oId[pML->mnId];
    //     VertexLinePlucker* vLine = static_cast<VertexLinePlucker*>(optimizer.vertex(g2oId));
    //     // 1) 写回 Plücker
    //     pML->SetPluckerLine(vLine->estimate());
    //     // 2) **用优化后的 Plücker + 各 KeyFrame 观测反投影来重建/更新世界端点**
    //     //    这个函数在 MapLine 类内部实现（你之前实现的 UpdateFromPluckerLine / UpdateFromPlucker）
    //     //    请确保 MapLine 中实现了这个函数并且线程安全（会读取 Plücker、遍历观测、写入端点）
    //     pML->UpdateFromPluckerLine();  
    //     // 3) 更新描述子（基于新的端点/Plücker）
    //     pML->ComputeDistinctiveDescriptors();
    //     pML->UpdateNormalAndDepth();
    //     if(!pML->isRetrived())
    //     {
    //         pML->setRetrived(true);
    //         opr.addMapLine(pML);
    //     }
    // }
    pMap->IncreaseChangeIndex();
    }
}




void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale,
                                       MappingOperation &opr,
                                       const std::unordered_set<unsigned long> &LoopKeyFrameIds)
{   
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid+1); // For debugging
    Eigen::Vector3d z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF->mnId==pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi]=vScw[nIDi].rotation()*z_vec; // For debugging

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // Inertial edges if inertial
        if(pKF->bImu && pKF->mPrevKF)
        {
            g2o::Sim3 Spw;
            LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if(itp!=NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            g2o::Sim3 Spi = Spw * Swi;
            g2o::EdgeSim3* ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    opr.reserveKeyFrames(vpKFs.size());
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);

        opr.addKeyFrame(pKFi, LoopKeyFrameIds.find(pKFi->mnId) != LoopKeyFrameIds.end());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    opr.reserveMapPoints(vpMPs.size());
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();

        if (!pMP->isRetrived()) {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<bool> vpGoodPose(nMaxKFid+1);
    vector<bool> vpBadPose(nMaxKFid+1);

    const int minFeat = 100;

    for(KeyFrame* pKFi : vpFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for(KeyFrame* pKFi : vpFixedCorrectedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
        vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(),Tcw_bef.translation(),1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<KeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(),vpFixedKFs.begin(),vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(),vpFixedCorrectedKFs.begin(),vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(),vpNonFixedKFs.begin(),vpNonFixedKFs.end());
    set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    for(KeyFrame* pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if(vpGoodPose[nIDi])
            correctedSwi = vCorrectedSwc[nIDi];
        if(vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw;
            bool bHasRelation = false;

            if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if(vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if(bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }

        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId<pKFi->mnId)
            {
                g2o::Sim3 Slw;
                bool bHasRelation = false;

                if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                }
                else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }


                if(bHasRelation)
                {
                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3* el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if(!pKFn->isBad() && pKFn->mnId<pKFi->mnId)
                {

                    g2o::Sim3 Snw =  vScw[pKFn->mnId];
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                    {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                    {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3* en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if(num_connections == 0 )
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation() / s);

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(MapPoint* pMPi : vpNonCorrectedMPs)
    {
        if(pMPi->isBad())
            continue;

        KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
        while(pRefKF->isBad())
        {
            if(!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if(vpBadPose[pRefKF->mnId])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);

            pMPi->UpdateNormalAndDepth();
        }
        else
        {
            cout << "ERROR: MapPoint has a reference KF from another map" << endl;
        }

    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    ORB_SLAM3::VertexSim3Expmap * vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pKF1->mpCamera;
    vSim3->pCamera2 = pKF2->mpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    vbIsInKF2.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;
    int nInKF2 = 0;
    int nOutKF2 = 0;
    int nMatchWithoutMP = 0;

    vector<int> vIdsOnlyInKF2;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            if(!pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if(i2<0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(P3D2c(2) < 0)
        {
            Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if(i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++;
        }
        else
        {
            float invz = 1/P3D2c(2);
            float x = P3D2c(0)*invz;
            float y = P3D2c(1)*invz;

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);

            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    int nBadOutKF2 = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            if(!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if(e12->chi2()>th2 || e21->chi2()>th2){
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else{
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, MappingOperation& opr, bool bLarge, bool bRecInit)
{
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt=10;
    int opt_it=10;
    if(bLarge)
    {
        maxOpt=25;
        opt_it=4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2,maxOpt);
    const unsigned long maxKFid = pKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<KeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for(int i=0, iend=vpNeighsKFs.size(); i<iend; i++)
    {
        if(lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame* pKFi = vpNeighsKFs[i];
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 200;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if(lFixedKeyFrames.size()>=maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    if(bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }


    // Set Local temporal KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KeyFrame vertices
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            if(i==N-1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if(i==N-1)
                    vei[i]->setInformation(vei[i]->information()*1e-2);
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);           

            optimizer.addEdge(vear[i]);
        }
        else
            cout << "ERROR building inertial edge" << endl;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);



    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    map<int,int> mVisEdges;
    for(int i=0;i<N;i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                continue;

            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                if(leftIndex != -1 && pKFi->mvuRight[leftIndex]<0)
                {
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                else if(leftIndex != -1)// Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if(pKFi->mpCamera2){
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 ){
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        EdgeMono* e = new EdgeMono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    {
        assert(mit->second>=3);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it); // Originally to 2
    float err_end = optimizer.activeRobustChi2();
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        bool bClose = pMP->mTrackDepth<10.f;

        if(pMP->isBad())
            continue;

        if((e->chi2()>chi2Mono2 && !bClose) || (e->chi2()>1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }


    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    // TODO: Some convergence problems have been detected here
    if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
        return;
    }



    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Recover optimized data
    // Local temporal Keyframes
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));

        }

        opr.addKeyFrame(pKFi);
    }

    // Local visual KeyFrame
    opr.reserveKeyFrames(lpOptVisKFs.size());
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;

        opr.addKeyFrame(pKFi);
    }

    //Points
    opr.reserveMapPoints(lLocalMapPoints.size());
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();

        if (!pMP->isRetrived()) {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }

    pMap->IncreaseChangeIndex();
}

Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end-start+1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end+1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
{
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    if (priorG!=0.f)
        solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(maxKFid*2+5);
    VS->setFixed(!bMono); // Fixed for stereo case
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());
    //std::cout << "build optimization graph" << std::endl;

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
            if(!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    scale = VS->estimate();

    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();


    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
    Rwg = VGDir->estimate().Rwg;

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate(); // Velocity is scaled after
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);


    }
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(1.0);
    VS->setId(maxKFid*2+5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (all variables are fixed)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2*(maxKFid+1)+(pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3*(maxKFid+1)+(pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    int count_edges = 0;
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
                
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->mnId);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) +  ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                continue;
            }
            count_edges++;
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    float err_end = optimizer.activeRobustChi2();
    // Recover optimized data
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag)
{
    bool bShowImages = false;

    vector<MapPoint*> vpMPs;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame*> spKeyFrameBA;

    Map* pCurrentMap = pMainKF->GetMap();

    // Set fixed KeyFrame vertices
    int numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpFixedKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
        {
            Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                    if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge=pMainKF->mnId;
                        numInsertedPoints++;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set non fixed Keyframe vertices
    set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
            {
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                {
                    if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpAdjustKF.size()+vpFixedKF.size())*vpMPs.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    map<KeyFrame*, int> mpObsKFs;
    map<KeyFrame*, int> mpObsFinalKFs;
    map<MapPoint*, int> mpObsMPs;
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMPi->GetWorldPos().cast<double>());
        const int id = pMPi->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);


        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi]+=2;
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    map<unsigned long int, int> mWrongObsKF;
    if(bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badMonoMP++;
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        Verbose::PrintMess("[BA]: First optimization(Huber), there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " stereo bad edges", Verbose::VERBOSITY_DEBUG);

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    set<MapPoint*> spErasedMPs;
    set<KeyFrame*> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    Verbose::PrintMess("[BA]: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsFinalKFs[pKF]++;
            }
        }
    }

    // Recover optimized data
    // Keyframes
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
        vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);

            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        pKFi->SetPose(Tiw);
    }

    //Points
    for(MapPoint* pMPi : vpMPs)
    {
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPi->mnId+maxKFid+1));
        pMPi->SetWorldPos(vPoint->estimate().cast<float>());
        pMPi->UpdateNormalAndDepth();

    }
}

void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
{
    const int Nd = 6;
    const unsigned long maxKFid = pCurrKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2*Nd);

    // For cov KFS, inertial parameters are not optimized
    const int maxCovKF = 30;
    vector<KeyFrame*> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Add temporal neighbours to merge KF (previous and next KFs)
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    // Previous KFs
    for(int i=1; i<(Nd/2); i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    // We fix just once the old map
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pCurrKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    if(pMergeKF->mNextKF)
    {
        vpOptimizableKFs.push_back(pMergeKF->mNextKF);
        vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    }

    while(vpOptimizableKFs.size()<(2*Nd))
    {
        if(vpOptimizableKFs.back()->mNextKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    map<MapPoint*,int> mLocalObs;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pCurrKF->mnId)
                    {
                        mLocalObs[pMP]=1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pCurrKF->mnId;
                    }
                    else {
                        mLocalObs[pMP]++;
                    }
        }
    }

    std::vector<std::pair<MapPoint*, int>> pairs;
    pairs.reserve(mLocalObs.size());
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
        pairs.push_back(*itr);
    sort(pairs.begin(), pairs.end(),sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    int i=0;
    for(vector<pair<MapPoint*,int>>::iterator lit=pairs.begin(), lend=pairs.end(); lit!=lend; lit++, i++)
    {
        map<KeyFrame*,tuple<int,int>> observations = lit->first->GetObservations();
        if(i>=maxCovKF)
            break;
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pCurrKF->mnId && pKFi->mnBAFixedForKF!=pCurrKF->mnId) // If optimizable or already included...
            {
                pKFi->mnBALocalForKF=pCurrKF->mnId;
                if(!pKFi->isBad())
                {
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set Local KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov=vpOptimizableCovKFs.size();
    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);
    for(int i=0;i<N;i++)
    {
        //cout << "inserting inertial edge " << i << endl;
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    }

    Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);


    // Set MapPoint vertices
    const int nExpectedSize = (N+Ncov+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        if (!pMP)
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (!pKFi)
                continue;

            if ((pKFi->mnBALocalForKF!=pCurrKF->mnId) && (pKFi->mnBAFixedForKF!=pCurrKF->mnId))
                continue;

            if (pKFi->mnId>maxKFid){
                continue;
            }


            if(optimizer.vertex(id)==NULL || optimizer.vertex(pKFi->mnId)==NULL)
                continue;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

                if(pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(8);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Mono2)
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }


    // Recover optimized data
    //Keyframes
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    KeyFrame* pKF = pFrame->mpLastKeyFrame;
    VertexPose* VPk = new VertexPose(pKF);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    float chi2Mono[4]={12,7.5,5.991,5.991};
    float chi2Stereo[4]={15.6,9.8,7.815,7.815};

    int its[4]={10,10,10,10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        float chi2close = 1.5*chi2Mono[it];

        // For monocular observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        // For stereo observations
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }

    }

    // If not too much tracks, recover not too bad points
    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double,15,15> H;
    H.setZero();

    H.block<9,9>(0,0)+= ei->GetHessian2();
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Current Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;
                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Previous Frame Vertex
    Frame* pFp = pFrame->mpPrevFrame;

    VertexPose* VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);

    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);
    egr->setVertex(1,VG);
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);
    ear->setVertex(1,VA);
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

    ep->setVertex(0,VPk);
    ep->setVertex(1,VVk);
    ep->setVertex(2,VGk);
    ep->setVertex(3,VAk);
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
    const int its[4]={10,10,10,10};

    int nBad=0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers=0;
    for(size_t it=0; it<4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers=0;
        nInliersMono=0;
        nInliersStereo=0;
        float chi2close = 1.5*chi2Mono[it];

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            break;
        }
    }


    if ((nInliers<30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;

        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    nInliers = nInliersMono + nInliersStereo;


    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    Eigen::Matrix<double,30,30> H;
    H.setZero();

    H.block<24,24>(0,0)+= ei->GetHessian();

    Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
    H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
    H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
    H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
    H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

    Eigen::Matrix<double,6,6> Har = ear->GetHessian();
    H.block<3,3>(12,12) += Har.block<3,3>(0,0);
    H.block<3,3>(12,27) += Har.block<3,3>(0,3);
    H.block<3,3>(27,12) += Har.block<3,3>(3,0);
    H.block<3,3>(27,27) += Har.block<3,3>(3,3);

    H.block<15,15>(0,0) += ep->GetHessian();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    H = Marginalize(H,0,14);

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    return nInitialCorrespondences-nBad;
}

void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       MappingOperation& opr,
                                       const std::unordered_set<unsigned long> &LoopKeyFrameIds)
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    vector<VertexPose4DoF*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;
    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexPose4DoF* V4DoF;

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF);
        }

        if(pKF==pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi]=V4DoF;
    }
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
    matLambda(0,0) = 1e3;
    matLambda(1,1) = 1e3;
    matLambda(0,0) = 1e3;

    // Set Loop edges
    Edge4DoF* e_loop;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3) = 1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // 1. Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // 1.1.0 Spanning tree edge
        KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        KeyFrame* prevKF = pKF->mPrevKF;
        if(prevKF)
        {
            int nIDj = prevKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Swl;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3,3>(0,0) = Sil.rotation().toRotationMatrix();
                Til.block<3,1>(0,3) = Sil.translation();
                Til(3,3) = 1.;

                Edge4DoF* e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && pKFn!=prevKF && pKFn!=pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Swn;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3,3>(0,0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3,1>(0,3) = Sin.translation();
                    Tin(3,3) = 1.;
                    Edge4DoF* e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    opr.reserveKeyFrames(vpKFs.size());
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
        Eigen::Vector3d ti = Vi->estimate().tcw[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri,ti,1.);
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation());
        pKFi->SetPose(Tiw.cast<float>());

        opr.addKeyFrame(pKFi, LoopKeyFrameIds.find(pKFi->mnId) != LoopKeyFrameIds.end());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    opr.reserveMapPoints(vpMPs.size());
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;

        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        nIDr = pRefKF->mnId;

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();

        if (!pMP->isRetrived()) {
            pMP->setRetrived(true);
            opr.addMapPoint(pMP);
        }
    }
    pMap->IncreaseChangeIndex();
}

} //namespace ORB_SLAM
