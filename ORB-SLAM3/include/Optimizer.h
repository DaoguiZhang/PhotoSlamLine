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


#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "Atlas.h"

#include <math.h>
#include <unordered_set>

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

#include <Thirdparty/g2o/g2o/core/base_edge.h> // BaseEdge 定义了 Edge 接口
#include <Thirdparty/g2o/g2o/core/base_unary_edge.h> 
#include <Thirdparty/g2o/g2o/core/base_binary_edge.h> 
#include <Thirdparty/g2o/g2o/core/base_multi_edge.h>

#include "VertexToLinePlucker.h"
#include "OptimizableTypes.h"
#include "util_slam.h"


namespace ORB_SLAM3
{

class LoopClosing;

class Optimizer
{
public:

    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);

    void static BundleAdjustmentWithLine(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const std::vector<MapLine *> &vpML,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust);

    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    
    void static GlobalBundleAdjustemntWithLine(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);

    void static FullInertialBA(Map *pMap, int its, const bool bFixLocal=false, const unsigned long nLoopKF=0, bool *pbStopFlag=NULL, bool bInit=false, float priorG = 1e2, float priorA=1e6, Eigen::VectorXd *vSingVal = NULL, bool *bHess=NULL);

    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, MappingOperation& opr);

    void static LocalBundleAdjustmentWithLine(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, int& num_Lines, MappingOperation& opr);
    void static LocalBundleAdjustmentWithLine_Optimization(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, int& num_Lines, MappingOperation& opr);
    void static LocalBundleAdjustmentWithLine_Optimization_Reg(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, int& num_Lines, MappingOperation& opr);
    
    void static LocalBundleAdjustmentWithLinesPluckerOld(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_lines,int& num_edges,MappingOperation& opr);
    void static LocalBundleAdjustmentWithLinesPlucker(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_lines,int& num_edges,MappingOperation& opr);
    void static TestPluckerLineEdgeJacobian();
    //debug function
    void static TestEdgeSE3ProjectPointToLine2D_Jacobian();
    void static TestEdgeLineLengthPrior_Jacobian_SAFE();
    void static TestEdgeLineDirectionPrior_Jacobian_SAFE();
    static g2o::SE3Quat se3Plus_ZDG(const g2o::SE3Quat& T, const Eigen::Matrix<double,6,1>& dx);
    void static TestEdgeSE3ProjectXYZOnlyPose();
    void static TestEdgeSE3ProjectLine_PoseAndPoints();
    void static TestEdgeSE3ProjectLineXYZOnlyPose_PointToLineOld();
    void static TestEdgeSE3ProjectLineXYZOnlyPose_PointToLine();
    static Eigen::Matrix<double,6,1> NormalizePluckerLine(const Eigen::Matrix<double,6,1>& L);
    static Eigen::Matrix<double, 2, 6> ComputeNumericalJacobianPose(
        ORB_SLAM3::EdgeSE3ProjectPluckerLine_PoseAndLine* edge,
        g2o::VertexSE3Expmap* vPose);
    static Eigen::Matrix<double,2,6> ComputeNumericalJacobianLine(
        ORB_SLAM3::EdgeSE3ProjectPluckerLine_PoseAndLine* edge,
        ORB_SLAM3::VertexLinePlucker* vLine);


    void static TestPluckerLinesBundleEdge();
    void static CheckJacobianNumerical();
    void static TestNumericalJacobian_PointToPlucker();
    //end debug function
    void static LocalBundleAdjustmentWithLinesPluckerBack(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_lines,int& num_edges,MappingOperation& opr);
    void static CheckDuplicateVertexID(g2o::SparseOptimizer& optimizer);
    void static LocalBundleAdjustmentWithLinesPlucker_Alternating(KeyFrame *pKF,bool* pbStopFlag,Map* pMap,int& num_fixedKF,int& num_OptKF,int& num_MPs,int& num_lines,int& num_edges,MappingOperation& opr);
    void static LocalBundleAdjustmentWithLinesPlucker_Depth_Alternating(KeyFrame *pKF,bool* pbStopFlag,Map* pMap,int& num_fixedKF,int& num_OptKF,int& num_MPs,int& num_lines,int& num_edges,MappingOperation& opr);

    void static WriteOptimizedDepthBack(
        const std::unordered_map<int, VertexDepth*>& id2newDepth,
        const std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*>& depthVertexMap);

    void static OptimizeDepthSeparately(
        std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
        std::vector<VertexDepth*>& vpDepthVerts,
        std::unordered_map<int, VertexDepth*>& id2newDepth_out,
        double lambda = 1e-3,
        int maxIter = 5,
        double min_depth = 0.01,
        double max_depth = 100.0);

    void static UpdateDepthVertices_ExternalLM_Edge_Safe(
        const std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
        const std::vector<VertexDepth*> &vpDepthVerts,
        double prior,
        double prior_w,
        int iters,
        double lambda_init,
        double min_depth,
        double max_depth);

    void static UpdateDepthVertices_ExternalLM_Safe_Verts(
        const std::vector<VertexDepth*>& depthVerts,
        const std::unordered_map<int, double>& priorMap,
        double prior_w,
        int iters,
        double lambda_init,
        double min_depth,
        double max_depth);

    void static UpdateDepthVertices_ExternalLM_Safe(
        g2o::SparseOptimizer& optimizer,
        const std::unordered_map<int, double>& priorMap,  // use vertex ID -> prior
        double prior_w,
        int iters,
        double lambda_init,
        double min_depth,
        double max_depth);

    void static UpdateDepthVertices_ExternalLM(
        const std::vector<EdgePointToPluckerLinePoseAndDepthNew*>& vpEdgesLine,
        const std::vector<VertexDepth*>& vpDepthVerts,
        const std::unordered_map<VertexDepth*, double>& priorMap, // d_prior（可为空）
        double prior_w,
        int iters,
        double lambda_init,
        double min_depth,
        double max_depth);

    static VertexDepth* FindDepthVertex_SchemeB(
        MapLine* pML,
        KeyFrame* pKFi,
        int endpoint,
        const std::map<std::tuple<MapLine*, KeyFrame*, int>, VertexDepth*>& depthVertexMap);
    static VertexDepth* CreateDepthVertex(g2o::SparseOptimizer& optimizer,int& nextVid,double initDepth);

    void static OptimizeOneIterationLocalBundleAdjustmentLinesPlucker(KeyFrame *pKF,bool* pbStopFlag,Map* pMap,int num_iter,std::list<KeyFrame*> pLocalKeyFrames,std::list<MapPoint*> pLocalMapPoints,
        std::list<MapLine*> pLocalMapLines, std::list<KeyFrame*> pFixedCameras, int& num_fixedKF,int& num_OptKF,int& num_MPs,int& num_lines,int& num_edges);
    int static PoseOptimization(Frame* pFrame);
    //int static PoseOptimizationWithLineEndLine(Frame* pFrame);
    int static PoseOptimizationWithLine(Frame* pFrame);
    int static PoseInertialOptimizationLastKeyFrame(Frame* pFrame, bool bRecInit = false);
    int static PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit = false);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale,
                                       MappingOperation &opr,
                                       const std::unordered_set<unsigned long> &LoopKeyFrameIds);
    void static OptimizeEssentialGraph(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs);

     // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraphWithLine(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale,
                                       MappingOperation &opr,
                                       const std::unordered_set<unsigned long> &LoopKeyFrameIds);
                                       
    void static OptimizeEssentialGraphWithLine(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs, vector<MapLine*> &vpNonCorrectedMLs);

    // For inertial loopclosing
    void static OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       MappingOperation &opr,
                                       const std::unordered_set<unsigned long> &LoopKeyFrameIds);


    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono) (NEW)
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale,
                            Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints=false);

    // For inertial systems

    void static LocalInertialBA(KeyFrame* pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, MappingOperation& opr, bool bLarge = false, bool bRecInit = false);
    void static MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses);

    // Local BA in welding area when two maps are merged
    void static LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag);

    // Local BA in welding area when two maps are merged with lines
    void static LocalBundleAdjustmentWithLine(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag);

    // Marginalize block element (start:end,start:end). Perform Schur complement.
    // Marginalized elements are filled with zeros.
    static Eigen::MatrixXd Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end);

    // Inertial pose-graph
    void static InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel=false, bool bGauss=false, float priorG = 1e2, float priorA = 1e6);
    void static InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG = 1e2, float priorA = 1e6);
    void static InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
//protected:
//    std::map<std::tuple<MapLine*, KeyFrame*, int /*lineIdx*/, int /*endpoint*/>, VertexDepth*> mDepthVertexMap;
};

} //namespace ORB_SLAM3

#endif // OPTIMIZER_H
