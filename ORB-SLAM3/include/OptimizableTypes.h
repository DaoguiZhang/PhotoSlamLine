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

#ifndef ORB_SLAM3_OPTIMIZABLETYPES_H
#define ORB_SLAM3_OPTIMIZABLETYPES_H

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/types/sim3.h>
#include <Thirdparty/g2o/g2o/core/base_multi_edge.h>
#include <Thirdparty/g2o/g2o/types/types_sba.h>        // VertexSBAPointXYZ

#include <Eigen/Geometry>
#include <include/CameraModels/GeometricCamera.h>
#include "VertexToLinePlucker.h"


namespace ORB_SLAM3 {
class  EdgeSE3ProjectXYZOnlyPose: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)


    EdgeSE3ProjectXYZOnlyPose(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project(v1->estimate().map(Xw));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return (v1->estimate().map(Xw))(2)>0.0;
    }


    virtual void linearizeOplus();

    Eigen::Vector3d Xw;
    GeometricCamera* pCamera;
    //Eigen::Matrix<double,2,6> mJacobianOplusXi;     //test for jacobian
};

class  EdgeSE3ProjectXYZOnlyPoseToBody: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)


    EdgeSE3ProjectXYZOnlyPoseToBody(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project((mTrl * v1->estimate()).map(Xw));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(Xw))(2)>0.0;
    }


    virtual void linearizeOplus();

    Eigen::Vector3d Xw;
    GeometricCamera* pCamera;

    g2o::SE3Quat mTrl;
};

class  EdgeSE3ProjectXYZ: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectXYZ();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project(v1->estimate().map(v2->estimate()));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        return ((v1->estimate().map(v2->estimate()))(2)>0.0);
    }

    virtual void linearizeOplus();

    GeometricCamera* pCamera;
};

class  EdgeSE3ProjectXYZToBody: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectXYZToBody();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project((mTrl * v1->estimate()).map(v2->estimate()));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(v2->estimate()))(2)>0.0;
    }

    virtual void linearizeOplus();

    GeometricCamera* pCamera;
    g2o::SE3Quat mTrl;
};

class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    VertexSim3Expmap();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        _estimate = g2o::Sim3();
    }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<g2o::Vector7d> update(const_cast<double*>(update_));

        if (_fix_scale)
            update[6] = 0;

        g2o::Sim3 s(update);
        setEstimate(s*estimate());
    }

    GeometricCamera* pCamera1, *pCamera2;

    bool _fix_scale;
};


class EdgeSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, ORB_SLAM3::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSim3ProjectXYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap* v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs-v1->pCamera1->project(v1->estimate().map(v2->estimate()));
    }

    // virtual void linearizeOplus();

};

class EdgeInverseSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d,  g2o::VertexSBAPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeInverseSim3ProjectXYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap* v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs-v1->pCamera2->project((v1->estimate().inverse().map(v2->estimate())));
    }

    // virtual void linearizeOplus();

};

// class EdgeSE3ProjectLineXYZOnlyPose
//     : public g2o::BaseUnaryEdge<4, Eigen::Matrix<double,4,1>, g2o::VertexSE3Expmap> 
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
//     EdgeSE3ProjectLineXYZOnlyPose() : pCamera(nullptr) {}
//     // Measurement = [u1, v1, u2, v2]
//     void computeError() override {
//         const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         g2o::SE3Quat Tcw = vSE3->estimate();
//         // Transform endpoints from world to camera
//         Eigen::Vector3d Xc1 = Tcw.map(Xw1);
//         Eigen::Vector3d Xc2 = Tcw.map(Xw2);
//         // Project endpoints
//         Eigen::Vector2d uv1, uv2;
//         uv1 = cam2pixel(Xc1);
//         uv2 = cam2pixel(Xc2);
//         // Residuals
//         _error(0) = _measurement(0) - uv1(0);
//         _error(1) = _measurement(1) - uv1(1);
//         _error(2) = _measurement(2) - uv2(0);
//         _error(3) = _measurement(3) - uv2(1);
//     }
//     bool isDepthPositive() const {
//         return (Xw1(2)>0 && Xw2(2)>0);
//     }
//     virtual void linearizeOplus() override {
//         if (!pCamera) return; // optional
//         BaseUnaryEdge::linearizeOplus();
//     }
//     Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
//         return Eigen::Vector2d(pCamera->fx * Xc(0)/Xc(2) + pCamera->cx,
//                                pCamera->fy * Xc(1)/Xc(2) + pCamera->cy);
//     }
//     // Inputs
//     struct CameraParams {
//         double fx, fy, cx, cy;
//     } *pCamera;
//     Eigen::Vector3d Xw1, Xw2;  // Line endpoints in world coordinates
// };

// class EdgeStereoSE3ProjectLineXYZOnlyPose
//     : public g2o::BaseUnaryEdge<6, Eigen::Matrix<double,6,1>, g2o::VertexSE3Expmap> 
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
//     EdgeStereoSE3ProjectLineXYZOnlyPose() : fx(0), fy(0), cx(0), cy(0), bf(0) {}
//     void computeError() override {
//         const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         g2o::SE3Quat Tcw = vSE3->estimate();
//         Eigen::Vector3d Xc1 = Tcw.map(Xw1);
//         Eigen::Vector3d Xc2 = Tcw.map(Xw2);
//         Eigen::Vector2d uv1l(fx*Xc1(0)/Xc1(2)+cx, fy*Xc1(1)/Xc1(2)+cy);
//         Eigen::Vector2d uv2l(fx*Xc2(0)/Xc2(2)+cx, fy*Xc2(1)/Xc2(2)+cy);
//         Eigen::Vector2d uv1r = uv1l; uv1r(0) -= bf / Xc1(2);
//         Eigen::Vector2d uv2r = uv2l; uv2r(0) -= bf / Xc2(2);
//         _error(0) = _measurement(0) - uv1l(0);
//         _error(1) = _measurement(1) - uv1l(1);
//         _error(2) = _measurement(2) - uv2l(0);
//         _error(3) = _measurement(3) - uv2l(1);
//         _error(4) = _measurement(4) - uv1r(0);
//         _error(5) = _measurement(5) - uv2r(0);
//     }
//     bool isDepthPositive() const {
//         return (Xw1(2)>0 && Xw2(2)>0);
//     }
//     Eigen::Vector3d Xw1, Xw2;
//     double fx, fy, cx, cy, bf;
// };

// class EdgeSE3ProjectLineXYZOnlyPoseToBody
//     : public g2o::BaseUnaryEdge<4, Eigen::Matrix<double,4,1>, g2o::VertexSE3Expmap> 
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
//     EdgeSE3ProjectLineXYZOnlyPoseToBody() : fx(0), fy(0), cx(0), cy(0) {}
//     void computeError() override {
//         const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         g2o::SE3Quat Tcw = vSE3->estimate();
//         g2o::SE3Quat Tcb = Tcw * mTrl; // body transform to left camera
//         Eigen::Vector3d Xc1 = Tcb.map(Xw1);
//         Eigen::Vector3d Xc2 = Tcb.map(Xw2);
//         Eigen::Vector2d uv1(fx*Xc1(0)/Xc1(2)+cx, fy*Xc1(1)/Xc1(2)+cy);
//         Eigen::Vector2d uv2(fx*Xc2(0)/Xc2(2)+cx, fy*Xc2(1)/Xc2(2)+cy);
//         _error(0) = _measurement(0) - uv1(0);
//         _error(1) = _measurement(1) - uv1(1);
//         _error(2) = _measurement(2) - uv2(0);
//         _error(3) = _measurement(3) - uv2(1);
//     }
//     g2o::SE3Quat mTrl; // body-to-left transform
//     Eigen::Vector3d Xw1, Xw2;
//     double fx, fy, cx, cy;
// };

class EdgeSE3ProjectLineXYZOnlyPose
    : public g2o::BaseUnaryEdge<4, Eigen::Matrix<double,4,1>, g2o::VertexSE3Expmap>
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLineXYZOnlyPose() : fx(0), fy(0), cx(0), cy(0) {}

    void computeError() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        _error(0) = _measurement(0) - uv1(0);
        _error(1) = _measurement(1) - uv1(1);
        _error(2) = _measurement(2) - uv2(0);
        _error(3) = _measurement(3) - uv2(1);
    }

    bool isDepthPositive() const {
        return (Xw1(2) > 0 && Xw2(2) > 0);
    }

    // ----------- 关键部分：解析雅可比 ----------
    void linearizeOplus() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        Eigen::Matrix<double,2,6> J1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> J2 = projectJacobian(Xc2);

        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<2,6>(0,0) = -J1;
        _jacobianOplusXi.block<2,6>(2,0) = -J2;
    }

    Eigen::Matrix<double,2,6> projectJacobian(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);
        const double z2 = z*z;

        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z, 0, -fx*x/z2,
               0, fy/z, -fy*y/z2;

        Eigen::Matrix<double,3,6> dXc_dse3;
        dXc_dse3.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        dXc_dse3(0,3) = 0;    dXc_dse3(0,4) =  z;   dXc_dse3(0,5) = -y;
        dXc_dse3(1,3) = -z;   dXc_dse3(1,4) =  0;   dXc_dse3(1,5) =  x;
        dXc_dse3(2,3) =  y;   dXc_dse3(2,4) = -x;   dXc_dse3(2,5) =  0;

        return Jpi * dXc_dse3;
    }

    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
        return Eigen::Vector2d(fx*Xc(0)/Xc(2) + cx, fy*Xc(1)/Xc(2) + cy);
    }

    Eigen::Vector3d Xw1, Xw2;
    double fx, fy, cx, cy;
};

class EdgeStereoSE3ProjectLineXYZOnlyPose
    : public g2o::BaseUnaryEdge<6, Eigen::Matrix<double,6,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeStereoSE3ProjectLineXYZOnlyPose()
        : fx(0), fy(0), cx(0), cy(0), bf(0) {}

    // Measurement: [u1l, v1l, u2l, v2l, u1r, u2r]^T
    void computeError() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        // left projections
        const Eigen::Vector2d uv1l = cam2pixel(Xc1);
        const Eigen::Vector2d uv2l = cam2pixel(Xc2);
        // right x coordinates (u = left_u - bf / z)
        const double u1r = uv1l(0) - bf / Xc1(2);
        const double u2r = uv2l(0) - bf / Xc2(2);

        _error(0) = _measurement(0) - uv1l(0);
        _error(1) = _measurement(1) - uv1l(1);
        _error(2) = _measurement(2) - uv2l(0);
        _error(3) = _measurement(3) - uv2l(1);
        _error(4) = _measurement(4) - u1r;
        _error(5) = _measurement(5) - u2r;
    }

    void linearizeOplus() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        // standard projection jacobians for left image (2x6)
        Eigen::Matrix<double,2,6> J1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> J2 = projectJacobian(Xc2);

        // For right u rows: dur/dpose = du_left/dpose + bf * dZ/dpose / z^2
        // where dZ/dpose is third row of dXc_dse3 (see projectJacobian construction)
        Eigen::Matrix<double,3,6> dXc1_dse3 = dXc_dse3(Xc1);
        Eigen::Matrix<double,3,6> dXc2_dse3 = dXc_dse3(Xc2);

        Eigen::Matrix<double,1,6> J1_ur; // derivative of u1r
        Eigen::Matrix<double,1,6> J2_ur; // derivative of u2r

        // du_left/dpose is first row of J1
        J1_ur = J1.row(0);
        J2_ur = J2.row(0);

        double z1 = Xc1(2);
        double z2 = Xc2(2);

        // add bf * dZ/dpose / z^2 term
        Eigen::Matrix<double,1,6> dZ1 = dXc1_dse3.row(2);
        Eigen::Matrix<double,1,6> dZ2 = dXc2_dse3.row(2);

        J1_ur += (bf / (z1*z1)) * dZ1;
        J2_ur += (bf / (z2*z2)) * dZ2;

        // Fill jacobian OplusXi (6 x 6)
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<2,6>(0,0) = -J1;      // -d(uv1l)/dpose
        _jacobianOplusXi.block<2,6>(2,0) = -J2;      // -d(uv2l)/dpose
        _jacobianOplusXi.block<1,6>(4,0) = -J1_ur;   // -d(u1r)/dpose
        _jacobianOplusXi.block<1,6>(5,0) = -J2_ur;   // -d(u2r)/dpose
    }

    // helper: 2x6 jacobian of [u v] wrt se3 at Xc
    Eigen::Matrix<double,2,6> projectJacobian(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);
        const double z2 = z*z;

        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z, 0, -fx*x/z2,
               0, fy/z, -fy*y/z2;

        Eigen::Matrix<double,3,6> dXc = dXc_dse3(Xc);
        return Jpi * dXc;
    }

    // dXc_dse3: derivative of Xc = [x,y,z]^T wrt se3 perturbation (6 DOF)
    Eigen::Matrix<double,3,6> dXc_dse3(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        Eigen::Matrix<double,3,6> J;
        J.setZero();
        // translation part
        J(0,0) = 1; J(1,1) = 1; J(2,2) = 1;
        // rotation part (small-angle approx)
        J(0,3) = 0;    J(0,4) =  z;  J(0,5) = -y;
        J(1,3) = -z;   J(1,4) =  0;  J(1,5) =  x;
        J(2,3) =  y;   J(2,4) = -x;  J(2,5) =  0;
        return J;
    }

    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
        return Eigen::Vector2d(fx*Xc(0)/Xc(2) + cx, fy*Xc(1)/Xc(2) + cy);
    }

    // members
    Eigen::Vector3d Xw1, Xw2;
    double fx, fy, cx, cy, bf;
};

class EdgeSE3ProjectLineXYZOnlyPoseToBody
    : public g2o::BaseUnaryEdge<4, Eigen::Matrix<double,4,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    EdgeSE3ProjectLineXYZOnlyPoseToBody() : fx(0), fy(0), cx(0), cy(0) {}

    // Measurement: [u1, v1, u2, v2]^T (left camera pixel coords after body transform)
    void computeError() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        // body->camera transform Tcb = Tcw * mTrl
        g2o::SE3Quat Tcb = Tcw * mTrl;

        Eigen::Vector3d Xc1 = Tcb.map(Xw1);
        Eigen::Vector3d Xc2 = Tcb.map(Xw2);

        const Eigen::Vector2d uv1 = cam2pixel(Xc1);
        const Eigen::Vector2d uv2 = cam2pixel(Xc2);

        _error(0) = _measurement(0) - uv1(0);
        _error(1) = _measurement(1) - uv1(1);
        _error(2) = _measurement(2) - uv2(0);
        _error(3) = _measurement(3) - uv2(1);
    }

    void linearizeOplus() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();
        g2o::SE3Quat Tcb = Tcw * mTrl;

        Eigen::Vector3d Xc1 = Tcb.map(Xw1);
        Eigen::Vector3d Xc2 = Tcb.map(Xw2);

        Eigen::Matrix<double,2,6> J1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> J2 = projectJacobian(Xc2);

        // BUT we must account that Xc = Tcb.map(Xw) and Tcb depends on Tcw:
        // d Xc / d se3 = d( (Tcw * mTrl) * Xw ) / d se3 = R(mTrl) * d (Tcw * X')? 
        // Simpler approach: treat Xcb = (Tcw * (mTrl.map(Xw))) i.e. pre-transform world point by mTrl
        // So compute Xb1 = mTrl.map(Xw1) and then Xc = Tcw.map(Xb1). The derivative as in single-camera case with Xc using Xb.
        Eigen::Vector3d Xb1 = mTrl.map(Xw1);
        Eigen::Vector3d Xb2 = mTrl.map(Xw2);

        Eigen::Matrix<double,2,6> J1_final = projectJacobian(Tcw.map(Xb1));
        Eigen::Matrix<double,2,6> J2_final = projectJacobian(Tcw.map(Xb2));

        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<2,6>(0,0) = -J1_final;
        _jacobianOplusXi.block<2,6>(2,0) = -J2_final;
    }

    // helper
    Eigen::Matrix<double,2,6> projectJacobian(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);
        const double z2 = z*z;

        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z, 0, -fx*x/z2,
               0, fy/z, -fy*y/z2;

        Eigen::Matrix<double,3,6> dXc = dXc_dse3(Xc);
        return Jpi * dXc;
    }

    Eigen::Matrix<double,3,6> dXc_dse3(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        Eigen::Matrix<double,3,6> J;
        J.setZero();
        J(0,0) = 1; J(1,1) = 1; J(2,2) = 1;
        J(0,3) = 0;    J(0,4) =  z;  J(0,5) = -y;
        J(1,3) = -z;   J(1,4) =  0;  J(1,5) =  x;
        J(2,3) =  y;   J(2,4) = -x;  J(2,5) =  0;
        return J;
    }

    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
        return Eigen::Vector2d(fx*Xc(0)/Xc(2) + cx, fy*Xc(1)/Xc(2) + cy);
    }

    // members
    g2o::SE3Quat mTrl; // body-to-left transform (set by caller)
    Eigen::Vector3d Xw1, Xw2;
    double fx, fy, cx, cy;
};

// Point-to-line error formulation, 通过数值测试，可以直接用于相机优化
///前 3 维：旋转（李代数 so(3)，角轴，小角度，单位：rad）后 3 维：平移（单位：米）VertexSE3Expmap 右扰动求导的雅可比 这些非常重要！！！
class EdgeSE3ProjectLineXYZOnlyPose_PointToLine
    : public g2o::BaseUnaryEdge<2, Eigen::Matrix<double,2,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLineXYZOnlyPose_PointToLine()
        : fx(0), fy(0), cx(0), cy(0), a(0), b(0), c(0)
    {
        //resize(1); // one vertex (the pose)
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // ---------------- intrinsics ----------------
    void SetCameraIntrinsics(double _fx, double _fy, double _cx, double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    // ---------------- observed line ----------------
    void SetObservedLineByEndpoints(double u1,double v1,double u2,double v2)
    {
        double dx = u2 - u1;
        double dy = v2 - v1;
        double na = dy;
        double nb = -dx;
        double norm = std::sqrt(na*na + nb*nb);
        if(norm < 1e-12){ a=b=c=0.0; return; }
        a = na / norm;
        b = nb / norm;
        c = -(a*u1 + b*v1);
    }

    void SetObservedLineABC(double _a,double _b,double _c)
    {
        double norm = std::sqrt(_a*_a + _b*_b);
        if(norm < 1e-12){ a=_a; b=_b; c=_c; return; }
        a = _a / norm;
        b = _b / norm;
        c = _c / norm;
    }

    // ---------------- world endpoints ----------------
    void SetXw(const Eigen::Vector3d &X1,const Eigen::Vector3d &X2)
    { Xw1=X1; Xw2=X2; }

    // ---------------- compute error ----------------
    void computeError() override
    {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        // 打印 Tcw 和 Xw1, Xw2
        ///std::cout << "Tcw: " << Tcw.rotation().toRotationMatrix() << ", Translation: " << Tcw.translation().transpose() << std::endl;
        ///std::cout << "Xw1: " << Xw1.transpose() << ", Xw2: " << Xw2.transpose() << std::endl;

        // Transform world endpoints to camera frame
        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        //// 打印 Xc1, Xc2 的值
        ///std::cout << "Mapped Xc1: " << Xc1.transpose() << std::endl;
        ///std::cout << "Mapped Xc2: " << Xc2.transpose() << std::endl;

        // Ensure positive depth (to avoid division by zero)
        if (Xc1(2) <= 0 || Xc2(2) <= 0) {
            std::cout << "Invalid depth detected (Z <= 0)!" << std::endl;
            _error(0) = 1e3;  // Set large residual in case of invalid depth
            _error(1) = 1e3;
            return;
        }

        // Project to pixels
        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        // Residuals: signed distance to line (a*b normalized)
        _error(0) = a * uv1(0) + b * uv1(1) + c;
        _error(1) = a * uv2(0) + b * uv2(1) + c;
    }

    // ---------------- Jacobian ----------------
    void linearizeOplus() override
    {
        const auto* vSE3 =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        Eigen::Vector3d Xc1 = vSE3->estimate().map(Xw1);
        Eigen::Vector3d Xc2 = vSE3->estimate().map(Xw2);

        if(Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9){
            _jacobianOplusXi.setZero();
            return;
        }

        Eigen::Matrix<double,2,6> Jp1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> Jp2 = projectJacobian(Xc2);

        Eigen::RowVector2d ab(a,b);

        //std::cerr<< "Jacobian computation: " << std::endl;
        //std::cerr<< "Xc1: " << Xc1.transpose() << ", Xc2: " << Xc2.transpose() << std::endl;
        //std::cerr<< "Jp1: \n" << Jp1 << std::endl;
        //std::cerr<< "Jp2: \n" << Jp2 << std::endl;

         //_jacobianOplusXi.setZero();  // 清空 Jacobian
        //std::cerr << "11" <<std::endl;
        _jacobianOplusXi.row(0) = ab * Jp1;
        _jacobianOplusXi.row(1) = ab * Jp2;
        
        //std::cerr<< "Jacobian OplusXi: \n" << _jacobianOplusXi << std::endl;
        //mJacobianPose.row(0) = ab * Jp1;
        //mJacobianPose.row(1) = ab * Jp2;
    }

public:
    // ---------------- helpers ----------------
    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const
    {
        return Eigen::Vector2d(
            fx * Xc(0) / Xc(2) + cx,
            fy * Xc(1) / Xc(2) + cy
        );
    }

    // right perturbation, se(3) = [w, t]
    Eigen::Matrix<double,2,6>
    projectJacobian(const Eigen::Vector3d &Xc) const
    {
        const double x=Xc(0), y=Xc(1), z=Xc(2), z2=z*z;

        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z,     0, -fx*x/z2,
                    0, fy/z, -fy*y/z2;

        Eigen::Matrix<double,3,6> dXc_dxi;
        dXc_dxi.setZero();

        // rotation part: -skew(Xc)
        dXc_dxi(0,1)= z;   dXc_dxi(0,2)= -y;
        dXc_dxi(1,0)= -z;  dXc_dxi(1,2)=  x;
        dXc_dxi(2,0)=  y;  dXc_dxi(2,1)= -x;

        // translation part
        dXc_dxi.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        return Jpi * dXc_dxi;
    }

    const Eigen::Matrix<double,2,6>& JPose() const {
        return mJacobianPose;
    }

private:
    Eigen::Matrix<double,2,6> mJacobianPose;
    Eigen::Vector3d Xw1, Xw2;
    double fx, fy, cx, cy;
    double a, b, c;
};

#if 0
/**
 * EdgeSE3ProjectLineXYZOnlyPose (point-to-line residual)
 *
 * Residual dimension: 2  (two endpoints -> two scalar point-to-line distances)
 *
 * Measurement representation:
 *   We store observed line L in the edge as unit-normal form (a,b,c) such that
 *     a*u + b*v + c = 0
 *   where (u,v) are pixel coordinates and (a,b) is unit (sqrt(a^2+b^2)=1).
 *
 * Error vector:
 *   e = [ a*u1 + b*v1 + c,
 *         a*u2 + b*v2 + c ]^T
 *
 * The jacobian rows are:
 *   J_row = [a b] * J_point   (1x6)
 * where J_point (2x6) is the usual projection jacobian for the endpoint.
 *
 * NOTE: This edge optimizes only the camera pose (VertexSE3Expmap). The 3D endpoints Xw1/Xw2
 * are provided by the corresponding MapLine and treated as constants here.
 */
class EdgeSE3ProjectLineXYZOnlyPose_PointToLine
    : public g2o::BaseUnaryEdge<2, Eigen::Matrix<double,2,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    EdgeSE3ProjectLineXYZOnlyPose_PointToLine()
        : fx(0), fy(0), cx(0), cy(0)
    {
        // initialize line params to zero (must be set by caller)
        a = b = c = 0.0;
    }

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

    // Set camera intrinsics
    void SetCameraIntrinsics(double _fx, double _fy, double _cx, double _cy) {
        fx = _fx; fy = _fy; cx = _cx; cy = _cy;
    }

    // Set observed line by two pixel endpoints (will compute normalized (a,b,c))
    void SetObservedLineByEndpoints(double u1, double v1, double u2, double v2)
    {
        // direction
        double dx = u2 - u1;
        double dy = v2 - v1;

        // normal (not yet normalized): n = (dy, -dx)
        double na = dy;
        double nb = -dx;
        double norm = std::sqrt(na*na + nb*nb);
        if (norm < 1e-12) {
            // degenerate: choose default
            a = 0; b = 0; c = 0;
            return;
        }
        a = na / norm;
        b = nb / norm;
        c = - (a * u1 + b * v1); // ensure a*u1 + b*v1 + c = 0
    }

    // Alternatively caller may set (a,b,c) directly (should ensure a^2 + b^2 = 1)
    void SetObservedLineABC(double _a, double _b, double _c) {
        double norm = std::sqrt(_a*_a + _b*_b);
        if (norm < 1e-12) {
            a = _a; b = _b; c = _c;
        } else {
            a = _a / norm; b = _b / norm; c = _c / norm;
        }
    }

    // Set world endpoints (3D) BEFORE adding to optimizer
    void SetXw(const Eigen::Vector3d &X1, const Eigen::Vector3d &X2) {
        Xw1 = X1;
        Xw2 = X2;
    }

    // computeError: project endpoints and compute point-to-line signed distances
    void computeError() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        // Transform world endpoints to camera frame
        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        // Check Z positive? If negative, still compute (but caller should check); optionally set large error
        if (Xc1(2) <= 0 || Xc2(2) <= 0) {
            // If you prefer to invalidate the measurement, you could set a large residual:
            // _error(0) = 1e3; _error(1) = 1e3; return;
            // Here we'll still compute projection (may be NaN if z<=0).
        }

        // Project to pixels
        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        // Residuals: signed distance to line (a*b normalized)
        _error(0) = a * uv1(0) + b * uv1(1) + c;
        _error(1) = a * uv2(0) + b * uv2(1) + c;
    }

    // linearizeOplus: analytic jacobian
    void linearizeOplus() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        Eigen::Vector3d Xc1 = Tcw.map(Xw1);
        Eigen::Vector3d Xc2 = Tcw.map(Xw2);

        // ensure positive depth to avoid division by zero
        const double eps = 1e-9;
        if (Xc1(2) < eps || Xc2(2) < eps) {
            _jacobianOplusXi.setZero();
            return;
        }

        // Compute 2x6 jacobians for each endpoint: J_point = d[ u v ] / dxi
        Eigen::Matrix<double,2,6> Jp1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> Jp2 = projectJacobian(Xc2);

        // The scalar jacobian row for point-to-line is [a b] * Jp  (1x6)
        Eigen::Matrix<double,1,6> row1 = Eigen::Matrix<double,1,6>::Zero();
        Eigen::Matrix<double,1,6> row2 = Eigen::Matrix<double,1,6>::Zero();

        // row = [a b] * Jp
        row1 = (Eigen::Matrix<double,1,2>() << a, b).finished() * Jp1;
        row2 = (Eigen::Matrix<double,1,2>() << a, b).finished() * Jp2;

        // Because error defined as e = a*u + b*v + c (i.e. pred appears positively),
        // and g2o expects the jacobian of error w.r.t. xi (no extra negative), we put:
        // _error = meas - pred  OR define consistent sign. In our computeError we used e = a*u + b*v + c,
        // so derivative is: de/dxi = [a b] * d[uv]/dxi. If you later set measurement differently,
        // check sign. For consistency with typical e = meas - pred, you might use negative.
        // Here we follow the computeError form, so set jacobian as below:
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.row(0) = row1;
        _jacobianOplusXi.row(1) = row2;

        // If you want residual = meas - pred, use _jacobianOplusXi.row(i) = -row;
    }

    // Helper: project 3D camera point to pixel coordinates (no distortion)
    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
        return Eigen::Vector2d(fx * Xc(0) / Xc(2) + cx,
                               fy * Xc(1) / Xc(2) + cy);
    }

    // Helper: 2x6 jacobian of [u v] wrt se3 at Xc (chain rule)
    Eigen::Matrix<double,2,6> projectJacobian(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);
        const double z2 = z*z;
        // d pi / d Xc  (2x3)
        Eigen::Matrix<double,2,3> Jpi;
        Jpi(0,0) = fx / z;      Jpi(0,1) = 0.0;        Jpi(0,2) = -fx * x / z2;
        Jpi(1,0) = 0.0;         Jpi(1,1) = fy / z;     Jpi(1,2) = -fy * y / z2;
        // d Xc / d xi  (3x6)  for left-multiplicative perturbation
        Eigen::Matrix<double,3,6> dXc_dxi;
        dXc_dxi.setZero();
        // translation part
        dXc_dxi(0,0) = 1.0; dXc_dxi(1,1) = 1.0; dXc_dxi(2,2) = 1.0;
        // rotation part (-[Xc]_x) rows:
        // [-[Xc]_x] = [ [ 0  z -y]; [-z 0 x]; [ y -x 0 ] ]
        dXc_dxi(0,3) = 0.0;  dXc_dxi(0,4) =  z;  dXc_dxi(0,5) = -y;
        dXc_dxi(1,3) = -z;   dXc_dxi(1,4) = 0.0; dXc_dxi(1,5) =  x;
        dXc_dxi(2,3) =  y;   dXc_dxi(2,4) = -x;  dXc_dxi(2,5) = 0.0;
        return Jpi * dXc_dxi; // 2x6
    }
    // Members
    Eigen::Vector3d Xw1, Xw2;   // world endpoints (set by caller)
    double fx, fy, cx, cy;      // camera intrinsics (set by caller)
    // observed line parameters (unit normal) a*u + b*v + c = 0
    double a, b, c;
};

// class EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine
//     : public g2o::BaseUnaryEdge<4, Eigen::Vector4d, g2o::VertexSE3Expmap>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
//     EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine(const Eigen::Vector3d& p1w,
//                                                     const Eigen::Vector3d& p2w,
//                                                     const cv::Mat& K,
//                                                     const double bf)
//         : Pw1(p1w), Pw2(p2w), mK(K), m_bf(bf)
//     {
//         fx = mK.at<double>(0, 0);
//         fy = mK.at<double>(1, 1);
//         cx = mK.at<double>(0, 2);
//         cy = mK.at<double>(1, 2);
//     }
//     void SetObservedLines(const Eigen::Vector3d& lineLeft, const Eigen::Vector3d& lineRight)
//     {
//         // 确保 a^2 + b^2 = 1
//         obsLineLeft  = lineLeft  / std::sqrt(lineLeft.head<2>().squaredNorm());
//         obsLineRight = lineRight / std::sqrt(lineRight.head<2>().squaredNorm());
//     }
//     void computeError() override
//     {
//         const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         Eigen::Matrix<double, 3, 4> P_left;
//         P_left << fx, 0, cx, 0,
//                   0, fy, cy, 0,
//                   0, 0, 1, 0;
//         Eigen::Matrix<double, 3, 4> P_right = P_left;
//         P_right(0, 3) = -fx * m_bf;  // baseline
//         Eigen::Vector3d Xc1 = vSE3->estimate().map(Pw1);
//         Eigen::Vector3d Xc2 = vSE3->estimate().map(Pw2);
//         Eigen::Vector3d uvL1 = P_left  * Xc1; uvL1 /= uvL1[2];
//         Eigen::Vector3d uvL2 = P_left  * Xc2; uvL2 /= uvL2[2];
//         Eigen::Vector3d uvR1 = P_right * Xc1; uvR1 /= uvR1[2];
//         Eigen::Vector3d uvR2 = P_right * Xc2; uvR2 /= uvR2[2];
//         const double aL = obsLineLeft(0),  bL = obsLineLeft(1),  cL = obsLineLeft(2);
//         const double aR = obsLineRight(0), bR = obsLineRight(1), cR = obsLineRight(2);
//         _error(0) = aL * uvL1(0) + bL * uvL1(1) + cL;
//         _error(1) = aL * uvL2(0) + bL * uvL2(1) + cL;
//         _error(2) = aR * uvR1(0) + bR * uvR1(1) + cR;
//         _error(3) = aR * uvR2(0) + bR * uvR2(1) + cR;
//     }
//     void linearizeOplus() override
//     {
//         const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         Eigen::Vector3d Xc[2] = {
//             vSE3->estimate().map(Pw1),
//             vSE3->estimate().map(Pw2)
//         };
//         const double aL = obsLineLeft(0),  bL = obsLineLeft(1);
//         const double aR = obsLineRight(0), bR = obsLineRight(1);
//         for (int i = 0; i < 2; i++) {
//             const Eigen::Vector3d& X = Xc[i];
//             double Xx = X[0], Yy = X[1], Zz = X[2];
//             Eigen::Matrix<double, 2, 3> Jpi;
//             Jpi << fx / Zz, 0, -fx * Xx / (Zz * Zz),
//                     0, fy / Zz, -fy * Yy / (Zz * Zz);
//             Eigen::RowVector3d d_rL_dXc = (Eigen::RowVector2d(aL, bL) * Jpi);
//             Eigen::RowVector3d d_rR_dXc = (Eigen::RowVector2d(aR, bR) * Jpi);
//             Eigen::Matrix<double, 3, 6> dXc_dxi;
//             dXc_dxi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
//             dXc_dxi.block<3, 3>(0, 3) = Sophus::SO3d::hat(X);
//             if (i == 0) {
//                 _jacobianOplusXi.block<1, 6>(0, 0) = d_rL_dXc * dXc_dxi;
//                 _jacobianOplusXi.block<1, 6>(2, 0) = d_rR_dXc * dXc_dxi;
//             } else {
//                 _jacobianOplusXi.block<1, 6>(1, 0) = d_rL_dXc * dXc_dxi;
//                 _jacobianOplusXi.block<1, 6>(3, 0) = d_rR_dXc * dXc_dxi;
//             }
//         }
//     }
// private:
//     Eigen::Vector3d Pw1, Pw2;
//     Eigen::Vector3d obsLineLeft, obsLineRight;
//     cv::Mat mK;
//     double fx, fy, cx, cy;
//     double m_bf;
// };

#endif

// Point-to-line error formulation for stereo camera (old version)，它是基于之前的实现，现已被更新版本取代
///它是左扰动，且前三维是平移，后三维是旋转（角轴），与当前主流实现不一致，建议使用更新版本 EdgeSE3ProjectLineXYZOnlyPose_PointToLine
class EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine_old
    : public g2o::BaseUnaryEdge<4, Eigen::Vector4d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine_old(const Eigen::Vector3d& p1w,
                                                    const Eigen::Vector3d& p2w,
                                                    const cv::Mat& K,
                                                    const double bf)
        : Pw1(p1w), Pw2(p2w), mK(K), m_bf(bf)
    {
        fx = mK.at<double>(0, 0);
        fy = mK.at<double>(1, 1);
        cx = mK.at<double>(0, 2);
        cy = mK.at<double>(1, 2);
    }

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

    void SetObservedLines(const Eigen::Vector3d& lineLeft, const Eigen::Vector3d& lineRight)
    {
        double nL = std::sqrt(lineLeft(0)*lineLeft(0) + lineLeft(1)*lineLeft(1));
        double nR = std::sqrt(lineRight(0)*lineRight(0) + lineRight(1)*lineRight(1));
        if (nL > 1e-12) obsLineLeft  = lineLeft  / nL;
        else obsLineLeft = lineLeft;
        if (nR > 1e-12) obsLineRight = lineRight / nR;
        else obsLineRight = lineRight;
    }

    void computeError() override
    {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        // Camera frame coordinates
        Eigen::Vector3d Xc1 = vSE3->estimate().map(Pw1);
        Eigen::Vector3d Xc2 = vSE3->estimate().map(Pw2);

        // check depth
        if (Xc1(2) <= 0 || Xc2(2) <= 0) {
            // set large residuals or compute anyway; here we compute anyway (but caller may discard)
        }

        // left projection
        Eigen::Vector2d uvL1(fx * Xc1(0) / Xc1(2) + cx, fy * Xc1(1) / Xc1(2) + cy);
        Eigen::Vector2d uvL2(fx * Xc2(0) / Xc2(2) + cx, fy * Xc2(1) / Xc2(2) + cy);

        // right projection: u_r = u_l - bf / Z, v_r = v_l
        Eigen::Vector2d uvR1 = uvL1;
        Eigen::Vector2d uvR2 = uvL2;
        uvR1(0) -= m_bf / Xc1(2);
        uvR2(0) -= m_bf / Xc2(2);

        const double aL = obsLineLeft(0),  bL = obsLineLeft(1),  cL = obsLineLeft(2);
        const double aR = obsLineRight(0), bR = obsLineRight(1), cR = obsLineRight(2);

        _error(0) = aL * uvL1(0) + bL * uvL1(1) + cL;
        _error(1) = aL * uvL2(0) + bL * uvL2(1) + cL;
        _error(2) = aR * uvR1(0) + bR * uvR1(1) + cR;
        _error(3) = aR * uvR2(0) + bR * uvR2(1) + cR;
    }

    void linearizeOplus() override
    {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d Xc1 = vSE3->estimate().map(Pw1);
        Eigen::Vector3d Xc2 = vSE3->estimate().map(Pw2);

        const double eps = 1e-9;
        if (Xc1(2) < eps || Xc2(2) < eps) {
            _jacobianOplusXi.setZero();
            return;
        }

        const double aL = obsLineLeft(0), bL = obsLineLeft(1);
        const double aR = obsLineRight(0), bR = obsLineRight(1);

        // compute Jpi for each endpoint (2x3)
        auto compute_Jpi = [&](const Eigen::Vector3d &Xc)->Eigen::Matrix<double,2,3>{
            double x = Xc(0), y = Xc(1), z = Xc(2), z2 = z*z;
            Eigen::Matrix<double,2,3> Jpi;
            Jpi(0,0) = fx / z;      Jpi(0,1) = 0.0;        Jpi(0,2) = -fx * x / z2;
            Jpi(1,0) = 0.0;         Jpi(1,1) = fy / z;     Jpi(1,2) = -fy * y / z2;
            return Jpi;
        };

        Eigen::Matrix<double,2,3> Jpi1 = compute_Jpi(Xc1);
        Eigen::Matrix<double,2,3> Jpi2 = compute_Jpi(Xc2);

        // d r / d Xc = [a b] * Jpi  (1x3)
        Eigen::RowVector3d drL_dXc1 = (Eigen::RowVector2d(aL, bL) * Jpi1);
        Eigen::RowVector3d drL_dXc2 = (Eigen::RowVector2d(aL, bL) * Jpi2);
        Eigen::RowVector3d drR_dXc1 = (Eigen::RowVector2d(aR, bR) * Jpi1);
        Eigen::RowVector3d drR_dXc2 = (Eigen::RowVector2d(aR, bR) * Jpi2);

        // dXc/dxi = [ I , -[Xc]_x ]  (3x6)
        auto dXc_dxi = [&](const Eigen::Vector3d &Xc)->Eigen::Matrix<double,3,6>{
            double x = Xc(0), y = Xc(1), z = Xc(2);
            Eigen::Matrix<double,3,6> J; J.setZero();
            J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();           // dXc / d( delta t )
            // rotation part is - [Xc]_x
            Eigen::Matrix3d Xhat;
            Xhat <<    0.0, -z,    y,
                      z,    0.0, -x,
                     -y,    x,    0.0;
            J.block<3,3>(0,3) = -Xhat;
            return J;
        };

        Eigen::Matrix<double,3,6> Jx1 = dXc_dxi(Xc1);
        Eigen::Matrix<double,3,6> Jx2 = dXc_dxi(Xc2);

        // Now assemble jacobian rows (order: L1,L2,R1,R2)
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<1,6>(0,0) = drL_dXc1 * Jx1; // r1,L
        _jacobianOplusXi.block<1,6>(1,0) = drL_dXc2 * Jx2; // r2,L

        // For right u: u_r = u_l - bf / z  -> dr/dXc includes contribution from u_r = u_l - bf/z
        // However we already used Jpi for left projection; for u_r we must consider du_r/dXc:
        // du_r/dXc = du_l/dXc + d(-bf/z)/dXc = du_l/dXc + bf/z^2 * [0,0,1]
        // But drR_dXc = [aR bR] * Jpi_right: since v component unchanged, we can use drR_dXc computed from Jpi (approx),
        // and correct the u-derivative extra term below:

        // Correct drR for u_r extra term:
        // For endpoint1:
        double z1 = Xc1(2);
        Eigen::RowVector3d d_extra1; d_extra1.setZero();
        // d(u_r)/dXc = d(u_l)/dXc + [ -(-bf) * d(1/z)/dXc ]? simpler to add bf * d(1/z)/dXc
        // d(1/z)/dXc = [0,0,-1/z^2]
        // contribution to r (scalar) is aR * ( - bf / z ), so derivative wrt Xc is aR * ( bf / z^2 ) * [0,0,1]
        d_extra1(2) = aR * (m_bf / (z1*z1));
        // Similarly for endpoint2:
        double z2 = Xc2(2);
        Eigen::RowVector3d d_extra2; d_extra2.setZero();
        d_extra2(2) = aR * (m_bf / (z2*z2));

        // now drR rows:
        _jacobianOplusXi.block<1,6>(2,0) = (drR_dXc1 + d_extra1) * Jx1; // r1,R
        _jacobianOplusXi.block<1,6>(3,0) = (drR_dXc2 + d_extra2) * Jx2; // r2,R

        // Note: depending on your residual sign convention (e = meas - pred), you may need to negate jacobian.
    }

private:
    Eigen::Vector3d Pw1, Pw2;
    Eigen::Vector3d obsLineLeft, obsLineRight;
    cv::Mat mK;
    double fx, fy, cx, cy;
    double m_bf;
};

class EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine
    : public g2o::BaseUnaryEdge<4, Eigen::Vector4d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine(const Eigen::Vector3d& p1w,
                                                    const Eigen::Vector3d& p2w,
                                                    const cv::Mat& K,
                                                    const double bf)
        : Pw1(p1w), Pw2(p2w), mK(K.clone()), m_bf(bf)
    {
        fx = mK.at<double>(0, 0);
        fy = mK.at<double>(1, 1);
        cx = mK.at<double>(0, 2);
        cy = mK.at<double>(1, 2);

        obsLineLeft.setZero();
        obsLineRight.setZero();
    }

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

    // line: ax + by + c = 0 (expect in pixel coords). We normalize by sqrt(a^2+b^2).
    void SetObservedLines(const Eigen::Vector3d& lineLeft, const Eigen::Vector3d& lineRight)
    {
        const double nL = std::sqrt(lineLeft(0)*lineLeft(0) + lineLeft(1)*lineLeft(1));
        const double nR = std::sqrt(lineRight(0)*lineRight(0) + lineRight(1)*lineRight(1));
        obsLineLeft  = (nL > 1e-12) ? (lineLeft  / nL) : lineLeft;
        obsLineRight = (nR > 1e-12) ? (lineRight / nR) : lineRight;
    }

    void computeError() override
    {
        const auto* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::SE3Quat Tcw = vSE3->estimate();

        const Eigen::Vector3d Xc1 = Tcw.map(Pw1);
        const Eigen::Vector3d Xc2 = Tcw.map(Pw2);

        // left pixels
        const Eigen::Vector2d uvL1 = cam2pixelLeft(Xc1);
        const Eigen::Vector2d uvL2 = cam2pixelLeft(Xc2);

        // right pixels: uR = uL - bf/z, vR=vL
        const Eigen::Vector2d uvR1 = cam2pixelRightFromLeft(Xc1, uvL1);
        const Eigen::Vector2d uvR2 = cam2pixelRightFromLeft(Xc2, uvL2);

        const double aL = obsLineLeft(0),  bL = obsLineLeft(1),  cL = obsLineLeft(2);
        const double aR = obsLineRight(0), bR = obsLineRight(1), cR = obsLineRight(2);

        _error(0) = aL * uvL1(0) + bL * uvL1(1) + cL;
        _error(1) = aL * uvL2(0) + bL * uvL2(1) + cL;
        _error(2) = aR * uvR1(0) + bR * uvR1(1) + cR;
        _error(3) = aR * uvR2(0) + bR * uvR2(1) + cR;
    }

    void linearizeOplus() override
    {
        const auto* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::SE3Quat Tcw = vSE3->estimate();

        const Eigen::Vector3d Xc1 = Tcw.map(Pw1);
        const Eigen::Vector3d Xc2 = Tcw.map(Pw2);

        const double eps = 1e-12;
        if (Xc1(2) < eps || Xc2(2) < eps) {
            _jacobianOplusXi.setZero();
            return;
        }

        const double aL = obsLineLeft(0),  bL = obsLineLeft(1);
        const double aR = obsLineRight(0), bR = obsLineRight(1);

        // dXc/dxi for VertexSE3Expmap RIGHT perturbation, se3 order [w, t]
        // => dXc/dxi = [ -skew(Xc) | I ]
        const Eigen::Matrix<double,3,6> Jx1 = dXc_dxi_right(Xc1);
        const Eigen::Matrix<double,3,6> Jx2 = dXc_dxi_right(Xc2);

        // Left projection Jacobian wrt Xc: d(ul,vl)/dXc (2x3)
        const Eigen::Matrix<double,2,3> JpiL1 = J_proj_left_wrt_Xc(Xc1);
        const Eigen::Matrix<double,2,3> JpiL2 = J_proj_left_wrt_Xc(Xc2);

        // Right projection Jacobian wrt Xc: d(ur,vr)/dXc (2x3)
        const Eigen::Matrix<double,2,3> JpiR1 = J_proj_right_wrt_Xc(Xc1);
        const Eigen::Matrix<double,2,3> JpiR2 = J_proj_right_wrt_Xc(Xc2);

        // Each residual row: r = a*u + b*v + c
        // => dr/dXc = [a b] * d(uv)/dXc   (1x3)
        const Eigen::RowVector2d abL(aL, bL);
        const Eigen::RowVector2d abR(aR, bR);

        const Eigen::RowVector3d drL1_dXc = abL * JpiL1;
        const Eigen::RowVector3d drL2_dXc = abL * JpiL2;
        const Eigen::RowVector3d drR1_dXc = abR * JpiR1;
        const Eigen::RowVector3d drR2_dXc = abR * JpiR2;

        // Chain to se3: dr/dxi = dr/dXc * dXc/dxi  => (1x6)
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<1,6>(0,0) = drL1_dXc * Jx1;
        _jacobianOplusXi.block<1,6>(1,0) = drL2_dXc * Jx2;
        _jacobianOplusXi.block<1,6>(2,0) = drR1_dXc * Jx1;
        _jacobianOplusXi.block<1,6>(3,0) = drR2_dXc * Jx2;

        // 如果你定义 residual = meas - pred，那么这里整体再乘 -1
        // _jacobianOplusXi = -_jacobianOplusXi;
    }

private:
    // ---------- projection ----------
    inline Eigen::Vector2d cam2pixelLeft(const Eigen::Vector3d& Xc) const
    {
        const double invz = 1.0 / Xc(2);
        return Eigen::Vector2d(fx * Xc(0) * invz + cx,
                               fy * Xc(1) * invz + cy);
    }

    inline Eigen::Vector2d cam2pixelRightFromLeft(const Eigen::Vector3d& Xc,
                                                  const Eigen::Vector2d& uvL) const
    {
        Eigen::Vector2d uvR = uvL;
        uvR(0) -= m_bf / Xc(2);
        return uvR;
    }

    // d(ul,vl)/dXc
    inline Eigen::Matrix<double,2,3> J_proj_left_wrt_Xc(const Eigen::Vector3d& Xc) const
    {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        const double z2 = z*z;

        Eigen::Matrix<double,2,3> J;
        J(0,0) = fx / z;   J(0,1) = 0.0;     J(0,2) = -fx * x / z2;
        J(1,0) = 0.0;      J(1,1) = fy / z;  J(1,2) = -fy * y / z2;
        return J;
    }

    // Right: ur = ul - bf/z, vr = vl
    // So:
    // dur/dXc = dul/dXc + [0,0, +bf/z^2]
    // dvr/dXc = dvl/dXc
    inline Eigen::Matrix<double,2,3> J_proj_right_wrt_Xc(const Eigen::Vector3d& Xc) const
    {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        const double z2 = z*z;

        Eigen::Matrix<double,2,3> J;
        // dul/dXc
        const double dul_dx = fx / z;
        const double dul_dy = 0.0;
        const double dul_dz = -fx * x / z2;

        // d(-bf/z)/dXc = [0,0, +bf/z^2]
        const double extra_dz = m_bf / z2;

        // ur
        J(0,0) = dul_dx;
        J(0,1) = dul_dy;
        J(0,2) = dul_dz + extra_dz;

        // vr == vl
        J(1,0) = 0.0;
        J(1,1) = fy / z;
        J(1,2) = -fy * y / z2;
        return J;
    }

    // ---------- right perturbation for VertexSE3Expmap: se3 order [w, t] ----------
    inline Eigen::Matrix<double,3,6> dXc_dxi_right(const Eigen::Vector3d& Xc) const
    {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        Eigen::Matrix<double,3,6> J; J.setZero();

        // rotation: -skew(Xc)
        // skew(Xc) = [ 0 -z  y; z 0 -x; -y x 0 ]
        // -skew(Xc) = [ 0 z -y; -z 0 x; y -x 0 ]
        J(0,0) = 0.0;  J(0,1) =  z;   J(0,2) = -y;
        J(1,0) = -z;   J(1,1) = 0.0;  J(1,2) =  x;
        J(2,0) =  y;   J(2,1) = -x;   J(2,2) = 0.0;

        // translation: I
        J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
        return J;
    }

private:
    Eigen::Vector3d Pw1, Pw2;
    Eigen::Vector3d obsLineLeft, obsLineRight;

    cv::Mat mK;
    double fx=0, fy=0, cx=0, cy=0;
    double m_bf=0;
};

// Pose-only + fixed body->left extrinsic (Trl) + point-to-line residual (2D)
// g2o::VertexSE3Expmap convention: RIGHT perturbation, xi = [w(3), t(3)]
// dXc/dxi = [ -skew(Xc) | I ]
class EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine()
        : fx(0.0), fy(0.0), cx(0.0), cy(0.0), a(0.0), b(0.0), c(0.0),
          mTrl(g2o::SE3Quat()) // identity by default
    {
        // Important: init debug jacobian storage
        mJacobianPose.setZero();
        Xw1.setZero();
        Xw2.setZero();
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // ---------------- intrinsics ----------------
    void SetCameraIntrinsics(double _fx, double _fy, double _cx, double _cy)
    {
        fx = _fx; fy = _fy; cx = _cx; cy = _cy;
    }

    // ---------------- observed line ----------------
    // line in image: a*u + b*v + c = 0, with sqrt(a^2+b^2)=1
    void SetObservedLineByEndpoints(double u1, double v1, double u2, double v2)
    {
        const double dx = u2 - u1;
        const double dy = v2 - v1;

        // normal = (dy, -dx)
        const double na = dy;
        const double nb = -dx;
        const double nn = std::sqrt(na * na + nb * nb);

        if (nn < 1e-12) { a = b = c = 0.0; return; }

        a = na / nn;
        b = nb / nn;
        c = -(a * u1 + b * v1);
    }

    void SetObservedLineABC(double _a, double _b, double _c)
    {
        const double nn = std::sqrt(_a * _a + _b * _b);
        if (nn < 1e-12) { a = _a; b = _b; c = _c; return; }
        a = _a / nn;
        b = _b / nn;
        c = _c / nn;
    }

    // ---------------- world endpoints ----------------
    void SetXw(const Eigen::Vector3d& X1, const Eigen::Vector3d& X2)
    {
        Xw1 = X1;
        Xw2 = X2;
    }

    // body->left camera extrinsic (constant)
    void SetTrl(const g2o::SE3Quat& Trl_)
    {
        mTrl = Trl_;
    }

    // ---------------- compute error ----------------
    void computeError() override
    {
        const auto* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::SE3Quat Tcw = vSE3->estimate();

        // Xb = Trl * Xw  (Trl: body->left camera)
        const Eigen::Vector3d Xb1 = mTrl.map(Xw1);
        const Eigen::Vector3d Xb2 = mTrl.map(Xw2);

        // Xc = Tcw * Xb
        const Eigen::Vector3d Xc1 = Tcw.map(Xb1);
        const Eigen::Vector3d Xc2 = Tcw.map(Xb2);

        // if invalid depth, give large residual (optional but safer)
        if (Xc1(2) <= 1e-12 || Xc2(2) <= 1e-12) {
            _error.setConstant(1e3);
            return;
        }

        const Eigen::Vector2d uv1 = cam2pixel(Xc1);
        const Eigen::Vector2d uv2 = cam2pixel(Xc2);

        // residuals: signed distance to observed line
        _error(0) = a * uv1(0) + b * uv1(1) + c;
        _error(1) = a * uv2(0) + b * uv2(1) + c;
    }

    // ---------------- analytic jacobian ----------------
    void linearizeOplus() override
    {
        const auto* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const g2o::SE3Quat Tcw = vSE3->estimate();

        const Eigen::Vector3d Xb1 = mTrl.map(Xw1);
        const Eigen::Vector3d Xb2 = mTrl.map(Xw2);

        const Eigen::Vector3d Xc1 = Tcw.map(Xb1);
        const Eigen::Vector3d Xc2 = Tcw.map(Xb2);

        if (Xc1(2) <= 1e-12 || Xc2(2) <= 1e-12) {
            _jacobianOplusXi.setZero();
            mJacobianPose.setZero();
            return;
        }

        // Jp: d[u v] / dxi (2x6), aligned with VertexSE3Expmap (RIGHT perturb, [w t])
        const Eigen::Matrix<double,2,6> Jp1 = projectJacobian_se3(Xc1);
        const Eigen::Matrix<double,2,6> Jp2 = projectJacobian_se3(Xc2);

        const Eigen::RowVector2d ab(a, b);

        _jacobianOplusXi.row(0) = ab * Jp1;   // 1x6
        _jacobianOplusXi.row(1) = ab * Jp2;

        // optional: store for your test function
        mJacobianPose.row(0) = _jacobianOplusXi.row(0);
        mJacobianPose.row(1) = _jacobianOplusXi.row(1);
    }

    // expose for testing
    const Eigen::Matrix<double,2,6>& JPose() const { return mJacobianPose; }

private:
    // ---------------- helpers ----------------
    inline Eigen::Vector2d cam2pixel(const Eigen::Vector3d& Xc) const
    {
        const double invz = 1.0 / Xc(2);
        return Eigen::Vector2d(fx * Xc(0) * invz + cx,
                               fy * Xc(1) * invz + cy);
    }

    // d[u v] / dxi, with xi = [w(3), t(3)] and RIGHT perturbation
    inline Eigen::Matrix<double,2,6> projectJacobian_se3(const Eigen::Vector3d& Xc) const
    {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        const double z2 = z * z;

        // duv/dXc (2x3)
        Eigen::Matrix<double,2,3> Jpi;
        Jpi(0,0) = fx / z;   Jpi(0,1) = 0.0;    Jpi(0,2) = -fx * x / z2;
        Jpi(1,0) = 0.0;      Jpi(1,1) = fy / z; Jpi(1,2) = -fy * y / z2;

        // dXc/dxi (3x6): [ -skew(Xc) | I ]
        Eigen::Matrix<double,3,6> dXc_dxi;
        dXc_dxi.setZero();

        // rotation part: -skew(Xc)
        // skew(Xc) = [  0  -z   y
        //              z   0  -x
        //             -y   x   0 ]
        // -skew(Xc) = [ 0  z  -y
        //              -z 0   x
        //               y -x  0 ]
        dXc_dxi(0,0) =  0.0; dXc_dxi(0,1) =  z;  dXc_dxi(0,2) = -y;
        dXc_dxi(1,0) = -z;   dXc_dxi(1,1) = 0.0; dXc_dxi(1,2) =  x;
        dXc_dxi(2,0) =  y;   dXc_dxi(2,1) = -x;  dXc_dxi(2,2) = 0.0;

        // translation part
        dXc_dxi.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        return Jpi * dXc_dxi; // 2x6
    }

private:
    // debug jacobian storage (for your numeric check)
    Eigen::Matrix<double,2,6> mJacobianPose;

public:
    // endpoints in world
    Eigen::Vector3d Xw1, Xw2;

    // observed line params (unit normal): a*u + b*v + c = 0
    double a, b, c;

    // intrinsics
    double fx, fy, cx, cy;

    // constant body->left camera transform
    g2o::SE3Quat mTrl;
};
// Pose-only + fixed body->left extrinsic (Trl) + point-to-line residual (2D)

//这个是旧版本的实现，保留以备参考，它是基于LEFT perturbation的，需要注意和上面的区别
// g2o::VertexSE3Expmap convention: LEFT perturbation, xi = [t(3), w(3)]
// dXc/dxi = [ I | -skew(Xc) ]
class EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine_old
    : public g2o::BaseUnaryEdge<2, Eigen::Matrix<double,2,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine_old()
        : fx(0.0), fy(0.0), cx(0.0), cy(0.0)
    {
        a = b = c = 0.0;
    }

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

    // Set intrinsics
    void SetCameraIntrinsics(double _fx, double _fy, double _cx, double _cy) {
        fx = _fx; fy = _fy; cx = _cx; cy = _cy;
    }

    // Set observed line by two image endpoints (left image): will normalize (a,b,c)
    void SetObservedLineByEndpoints(double u1, double v1, double u2, double v2)
    {
        double dx = u2 - u1;
        double dy = v2 - v1;
        double na = dy;
        double nb = -dx;
        double nn = std::sqrt(na*na + nb*nb);
        if (nn < 1e-12) {
            a = 0; b = 0; c = 0;
            return;
        }
        a = na / nn;
        b = nb / nn;
        c = - (a * u1 + b * v1);
    }

    // Set observed line directly (a,b,c) will be normalized so sqrt(a^2+b^2)=1
    void SetObservedLineABC(double _a, double _b, double _c) {
        double nn = std::sqrt(_a*_a + _b*_b);
        if (nn < 1e-12) {
            a = _a; b = _b; c = _c;
        } else {
            a = _a / nn; b = _b / nn; c = _c / nn;
        }
    }

    // Set the two world endpoints of the MapLine
    void SetXw(const Eigen::Vector3d &X1, const Eigen::Vector3d &X2) {
        Xw1 = X1;
        Xw2 = X2;
    }

    // Set the constant body->left camera transform (g2o::SE3Quat)
    void SetTrl(const g2o::SE3Quat &Trl_) {
        mTrl = Trl_;
    }

    // computeError: project endpoints and compute signed distances to observed line
    void computeError() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        // Pre-transform world points by mTrl (body->camera offset): Xb = mTrl.map(Xw)
        Eigen::Vector3d Xb1 = mTrl.map(Xw1);
        Eigen::Vector3d Xb2 = mTrl.map(Xw2);

        // Then transform with Tcw: Xc = Tcw.map(Xb)
        Eigen::Vector3d Xc1 = Tcw.map(Xb1);
        Eigen::Vector3d Xc2 = Tcw.map(Xb2);

        // Avoid division by zero; still compute but caller may skip if z<=0
        if (Xc1(2) <= 0 || Xc2(2) <= 0) {
            // Keep computing; you may later detect invalid measurements by checking depth
        }

        // project to pixel coordinates (no distortion)
        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        // signed distances (residuals): a*u + b*v + c
        _error(0) = a * uv1(0) + b * uv1(1) + c;
        _error(1) = a * uv2(0) + b * uv2(1) + c;
    }

    // analytic jacobian (2x6)
    void linearizeOplus() override {
        const g2o::VertexSE3Expmap* vSE3 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat Tcw = vSE3->estimate();

        // Pre-transform by mTrl
        Eigen::Vector3d Xb1 = mTrl.map(Xw1);
        Eigen::Vector3d Xb2 = mTrl.map(Xw2);

        // Xc = Tcw.map(Xb)
        Eigen::Vector3d Xc1 = Tcw.map(Xb1);
        Eigen::Vector3d Xc2 = Tcw.map(Xb2);

        const double eps = 1e-9;
        if (Xc1(2) < eps || Xc2(2) < eps) {
            _jacobianOplusXi.setZero();
            return;
        }

        // Compute 2x6 projection jacobian for each endpoint Jp = d[u v] / d xi
        Eigen::Matrix<double,2,6> Jp1 = projectJacobian(Xc1);
        Eigen::Matrix<double,2,6> Jp2 = projectJacobian(Xc2);

        // row = [a b] * Jp  (1x6)
        Eigen::Matrix<double,1,6> row1 = (Eigen::Matrix<double,1,2>() << a, b).finished() * Jp1;
        Eigen::Matrix<double,1,6> row2 = (Eigen::Matrix<double,1,2>() << a, b).finished() * Jp2;

        // NOTE about sign convention:
        // computeError sets e = a*u + b*v + c (signed). Then de/dxi = [a b] * d[uv]/dxi.
        // If you prefer e = meas - pred, flip signs consistently (both error and jacobian).
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.row(0) = row1;
        _jacobianOplusXi.row(1) = row2;
    }

protected:
    // project to pixel (no distortion)
    Eigen::Vector2d cam2pixel(const Eigen::Vector3d &Xc) const {
        return Eigen::Vector2d(fx * Xc(0) / Xc(2) + cx,
                               fy * Xc(1) / Xc(2) + cy);
    }

    // 2x6 jacobian of [u v] wrt se3 at Xc (for left-multiplicative perturbation)
    Eigen::Matrix<double,2,6> projectJacobian(const Eigen::Vector3d &Xc) const {
        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);
        const double z2 = z * z;

        Eigen::Matrix<double,2,3> Jpi;
        Jpi(0,0) = fx / z;      Jpi(0,1) = 0.0;        Jpi(0,2) = -fx * x / z2;
        Jpi(1,0) = 0.0;         Jpi(1,1) = fy / z;     Jpi(1,2) = -fy * y / z2;

        Eigen::Matrix<double,3,6> dXc_dxi;
        dXc_dxi.setZero();
        // translation part
        dXc_dxi(0,0) = 1.0; dXc_dxi(1,1) = 1.0; dXc_dxi(2,2) = 1.0;
        // rotation part (-[Xc]_x)
        dXc_dxi(0,3) = 0.0;  dXc_dxi(0,4) =  z;  dXc_dxi(0,5) = -y;
        dXc_dxi(1,3) = -z;   dXc_dxi(1,4) = 0.0; dXc_dxi(1,5) =  x;
        dXc_dxi(2,3) =  y;   dXc_dxi(2,4) = -x;  dXc_dxi(2,5) = 0.0;

        return Jpi * dXc_dxi; // 2x6
    }

public:
    // world endpoints
    Eigen::Vector3d Xw1, Xw2;

    // observed line parameters (unit normal): a*u + b*v + c = 0
    double a, b, c;

    // intrinsics
    double fx, fy, cx, cy;

    // body->left transform (constant)
    g2o::SE3Quat mTrl;
};



class EdgeSE3ProjectLine_PoseAndPoints_old : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLine_PoseAndPoints_old()
        : fx(0), fy(0), cx(0), cy(0), a(0), b(0), c(0)
    {
        resize(3); // 0: pose, 1: point1, 2: point2
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    void SetObservedLineByEndpoints(double u1,double v1,double u2,double v2)
    {
        double dx = u2-u1, dy = v2-v1;
        double na = dy, nb = -dx;
        double norm = std::sqrt(na*na + nb*nb);
        if(norm < 1e-12){ a=b=c=0.0; return; }
        a = na / norm;
        b = nb / norm;
        c = -(a*u1 + b*v1);
    }

    void SetObservedLineABC(double _a,double _b,double _c)
    {
        double norm = std::sqrt(_a*_a + _b*_b);
        if(norm < 1e-12){ a=_a; b=_b; c=_c; return; }
        a = _a/norm;
        b = _b/norm;
        c = _c/norm;
    }

    void computeError() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        const VertexSE3Expmap* vPose  = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* vP1  = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const VertexSBAPointXYZ* vP2  = static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        Eigen::Vector3d Xc1 = vPose->estimate().map(vP1->estimate());
        Eigen::Vector3d Xc2 = vPose->estimate().map(vP2->estimate());

        if (Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9) {
            _error.setConstant(1e3);
            return;
        }

        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        _error(0) = a*uv1(0) + b*uv1(1) + c;
        _error(1) = a*uv2(0) + b*uv2(1) + c;
    }

    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        const VertexSE3Expmap* vPose  = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* vP1  = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const VertexSBAPointXYZ* vP2  = static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        Eigen::Vector3d Xc1 = vPose->estimate().map(vP1->estimate());
        Eigen::Vector3d Xc2 = vPose->estimate().map(vP2->estimate());

        Eigen::Matrix3d R = vPose->estimate().rotation().toRotationMatrix();

        if (Xc1(2)<=1e-9 || Xc2(2)<=1e-9) {
            _jacobianOplus[0] = Eigen::Matrix<double,2,6>::Zero();
            _jacobianOplus[1] = Eigen::Matrix<double,2,3>::Zero();
            _jacobianOplus[2] = Eigen::Matrix<double,2,3>::Zero();
            return;
        }

        // Projection Jacobians
        Eigen::Matrix<double,2,3> Jpi1 = projectionJacobian_Xc(Xc1);
        Eigen::Matrix<double,2,3> Jpi2 = projectionJacobian_Xc(Xc2);

        // dXc/dxi (3x6)
        Eigen::Matrix<double,3,6> dXc1_dxi = dXc_dxi_from_Xc(Xc1);
        Eigen::Matrix<double,3,6> dXc2_dxi = dXc_dxi_from_Xc(Xc2);

        // Pose Jacobian (2x6)
        Eigen::Matrix<double,2,6> Jpose1 = Jpi1 * dXc1_dxi;
        Eigen::Matrix<double,2,6> Jpose2 = Jpi2 * dXc2_dxi;

        Eigen::RowVector2d ab(a,b);
        Eigen::Matrix<double,1,6> row1 = ab * Jpose1;
        Eigen::Matrix<double,1,6> row2 = ab * Jpose2;

        Eigen::Matrix<double,2,6> Jpose;
        Jpose.row(0) = row1;
        Jpose.row(1) = row2;
        _jacobianOplus[0] = Jpose;

        // Point Jacobians (2x3)
        _jacobianOplus[1] = Jpi1 * R;
        _jacobianOplus[2] = Jpi2 * R;
    }

private:
    Eigen::Vector2d cam2pixel(const Eigen::Vector3d& Xc) const {
        return Eigen::Vector2d(fx*Xc(0)/Xc(2)+cx,
                               fy*Xc(1)/Xc(2)+cy);
    }

    Eigen::Matrix<double,2,3> projectionJacobian_Xc(const Eigen::Vector3d &Xc) const {
        double x=Xc(0), y=Xc(1), z=Xc(2), z2=z*z;
        Eigen::Matrix<double,2,3> J;
        J << fx/z, 0, -fx*x/z2,
             0, fy/z, -fy*y/z2;
        return J;
    }

    Eigen::Matrix<double,3,6> dXc_dxi_from_Xc(const Eigen::Vector3d& Xc) const {
        Eigen::Matrix<double,3,6> m = Eigen::Matrix<double,3,6>::Zero();
        m.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        double x=Xc(0), y=Xc(1), z=Xc(2);
        m(0,4)= z;   m(0,5)= -y;
        m(1,3)= -z;  m(1,5)= x;
        m(2,3)= y;   m(2,4)= -x;
        return m;
    }

public:
    double fx, fy, cx, cy;
    double a, b, c;
};


//测试通过，VertexSE3Expmap 右扰动测试。后面其它方法，求导也要注意右扰动
//前 3 维：旋转（李代数 so(3)，角轴，小角度，单位：rad）后 3 维：平移（单位：米）
class EdgeSE3ProjectLine_PoseAndPoints
    : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLine_PoseAndPoints()
        : fx(0), fy(0), cx(0), cy(0), a(0), b(0), c(0),
          valid_obs_(false)
    {
        resize(3); // 0: pose, 1: point1, 2: point2
        mJacobianPose.setZero();
        mJacobianP1.setZero();
        mJacobianP2.setZero();
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // ---------------- intrinsics ----------------
    void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    // ---------------- observed line ----------------
    // ---------------- observed line ----------------
    // Return false if degenerate observation (zero-length 2D line).
    bool SetObservedLineByEndpoints(double u1,double v1,double u2,double v2)
    {
        const double dx = u2 - u1;
        const double dy = v2 - v1;
        const double na = dy;
        const double nb = -dx;
        const double norm = std::sqrt(na*na + nb*nb);
        if(!std::isfinite(norm) || norm < 1e-12)
        {
            a=b=c=0.0;
            valid_obs_ = false;
            return false;
        }
        a = na / norm;
        b = nb / norm;
        c = -(a*u1 + b*v1);
        valid_obs_ = std::isfinite(a) && std::isfinite(b) && std::isfinite(c);
        return valid_obs_;
    }

    bool SetObservedLineABC(double _a,double _b,double _c)
    {
        const double norm = std::sqrt(_a*_a + _b*_b);
        if(!std::isfinite(norm) || norm < 1e-12)
        {
            a=_a; b=_b; c=_c;
            valid_obs_ = false;
            return false;
        }
        a = _a / norm;
        b = _b / norm;
        c = _c / norm; // keep your original normalization behavior
        valid_obs_ = std::isfinite(a) && std::isfinite(b) && std::isfinite(c);
        return valid_obs_;
    }

    // ---------------- compute error ----------------
    void computeError() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        // vertex null check (safety)
        if(!_vertices[0] || !_vertices[1] || !_vertices[2] || !valid_obs_)
        {
            _error.setConstant(1e3);
            return;
        }

        const VertexSE3Expmap* vPose =
            static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* vP1 =
            static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const VertexSBAPointXYZ* vP2 =
            static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        Eigen::Vector3d Xc1 = vPose->estimate().map(vP1->estimate());
        Eigen::Vector3d Xc2 = vPose->estimate().map(vP2->estimate());

        if (Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9) {
            _error.setConstant(1e3);
            return;
        }

        // segment degeneracy check in camera frame (BA may collapse endpoints)
        const Eigen::Vector3d d = Xc2 - Xc1;
        if (d.squaredNorm() < 1e-12)
        {
            _error.setConstant(1e3);
            return;
        }

        Eigen::Vector2d uv1 = cam2pixel(Xc1);
        Eigen::Vector2d uv2 = cam2pixel(Xc2);

        _error(0) = a*uv1(0) + b*uv1(1) + c;
        _error(1) = a*uv2(0) + b*uv2(1) + c;

        if(!std::isfinite(_error(0)) || !std::isfinite(_error(1)))
            _error.setConstant(1e3);
    }

    // ---------------- Jacobians ----------------
    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        // default to zeros
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();
        mJacobianPose.setZero();
        mJacobianP1.setZero();
        mJacobianP2.setZero();

        if(!_vertices[0] || !_vertices[1] || !_vertices[2] || !valid_obs_)
            return;

        const VertexSE3Expmap* vPose =
            static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSBAPointXYZ* vP1 =
            static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const VertexSBAPointXYZ* vP2 =
            static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        Eigen::Vector3d Xc1 = vPose->estimate().map(vP1->estimate());
        Eigen::Vector3d Xc2 = vPose->estimate().map(vP2->estimate());
        Eigen::Matrix3d R   = vPose->estimate().rotation().toRotationMatrix();

        if (Xc1(2)<=1e-9 || Xc2(2)<=1e-9) {
            _jacobianOplus[0].setZero();
            _jacobianOplus[1].setZero();
            _jacobianOplus[2].setZero();
            return;
        }

        if (Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9)
            return;

        const Eigen::Vector3d d = Xc2 - Xc1;
        if (d.squaredNorm() < 1e-12)
            return;

        // duv / dXc
        Eigen::Matrix<double,2,3> Jpi1 = projectionJacobian_Xc(Xc1);
        Eigen::Matrix<double,2,3> Jpi2 = projectionJacobian_Xc(Xc2);

        // dXc / dxi (right perturbation, VertexSE3Expmap)
        Eigen::Matrix<double,3,6> dXc1_dxi = dXc_dxi_from_Xc(Xc1);
        Eigen::Matrix<double,3,6> dXc2_dxi = dXc_dxi_from_Xc(Xc2);

        // duv / dxi
        Eigen::Matrix<double,2,6> Jpose1 = Jpi1 * dXc1_dxi;
        Eigen::Matrix<double,2,6> Jpose2 = Jpi2 * dXc2_dxi;

        Eigen::RowVector2d ab(a,b);

        // ---- Pose Jacobian (2x6)
        _jacobianOplus[0].row(0) = ab * Jpose1;
        _jacobianOplus[0].row(1) = ab * Jpose2;
        mJacobianPose.row(0) = ab * Jpose1; // for test
        mJacobianPose.row(1) = ab * Jpose2; // for test

        // ---- Point Jacobians (2x3)  [关键修正点]
        _jacobianOplus[1].setZero();
        _jacobianOplus[2].setZero();

        _jacobianOplus[1].row(0) = ab * Jpi1 * R; // P1 only affects error(0)
        _jacobianOplus[2].row(1) = ab * Jpi2 * R; // P2 only affects error(1)
        mJacobianP1.row(0) = ab * Jpi1 * R;
        mJacobianP1.row(1).setZero();

        mJacobianP2.row(0).setZero();
        mJacobianP2.row(1) = ab * Jpi2 * R;

        // final finite check (avoid NaN poisoning Hessian)
        if(!allFinite(_jacobianOplus[0]) ||
           !allFinite(_jacobianOplus[1]) ||
           !allFinite(_jacobianOplus[2]))
        {
            _jacobianOplus[0].setZero();
            _jacobianOplus[1].setZero();
            _jacobianOplus[2].setZero();
            mJacobianPose.setZero();
            mJacobianP1.setZero();
            mJacobianP2.setZero();
        }
    }

private:
    bool valid_obs_;

    // ---------------- helpers ----------------
    Eigen::Vector2d cam2pixel(const Eigen::Vector3d& Xc) const {
        return Eigen::Vector2d(
            fx * Xc(0) / Xc(2) + cx,
            fy * Xc(1) / Xc(2) + cy
        );
    }

    Eigen::Matrix<double,2,3>
    projectionJacobian_Xc(const Eigen::Vector3d &Xc) const
    {
        double x = Xc(0), y = Xc(1), z = Xc(2), z2 = z*z;
        Eigen::Matrix<double,2,3> J;
        J << fx/z,     0, -fx*x/z2,
                 0, fy/z, -fy*y/z2;
        return J;
    }

    // right perturbation: dXc/dxi = [ -skew(Xc) | I ]
    Eigen::Matrix<double,3,6>
    dXc_dxi_from_Xc(const Eigen::Vector3d& Xc) const
    {
        Eigen::Matrix<double,3,6> J;
        J.setZero();

        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);

        // -skew(Xc)
        J(0,1) =  z;   J(0,2) = -y;
        J(1,0) = -z;  J(1,2) =  x;
        J(2,0) =  y;  J(2,1) = -x;

        // translation
        J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        return J;
    }

    template<typename Derived>
    static bool allFinite(const Eigen::MatrixBase<Derived>& M)
    {
        for (int r=0; r<M.rows(); ++r)
            for (int c=0; c<M.cols(); ++c)
                if(!std::isfinite(M(r,c))) return false;
        return true;
    }

public:
    const Eigen::Matrix<double,2,6>& JPose() const {
        return mJacobianPose;
    }
    const Eigen::Matrix<double,2,3>& JP1() const {
        return mJacobianP1;
    }
    const Eigen::Matrix<double,2,3>& JP2() const {
        return mJacobianP2;
    }
    Eigen::Matrix<double, 2, 6> mJacobianPose; // for test
    Eigen::Matrix<double, 2, 3> mJacobianP1;   // for test
    Eigen::Matrix<double, 2, 3> mJacobianP2;   // for test
    double fx, fy, cx, cy;
    double a, b, c;
};


// Edge: point (world) + pose -> distance to observed 2D line (a u + b v + c = 0)
//前 3 维：旋转（李代数 so(3)，角轴，小角度，单位：rad）后 3 维：平移（单位：米）; VertexSE3Expmap 左扰动测试。后面其它方法，求导也要注意左扰动
// 1 residual: point-to-line distance
class EdgeSE3ProjectPointToLine2D
    : public g2o::BaseBinaryEdge<1, Eigen::Vector3d,
                                 g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectPointToLine2D()
        : fx(0), fy(0), cx(0), cy(0) {}

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    // line = [a,b,c] should already be normalized s.t. sqrt(a^2+b^2)=1
    void computeError() override
    {
        const auto* vP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        const Eigen::Vector3d Xc = vT->estimate().map(vP->estimate());
        if (Xc(2) <= 1e-9 || !Xc.allFinite())
        {
            _error[0] = 1e3;
            return;
        }

        const double invz = 1.0 / Xc(2);
        const double u = fx * Xc(0) * invz + cx;
        const double v = fy * Xc(1) * invz + cy;

        const Eigen::Vector3d& L = _measurement; // (a,b,c)
        _error[0] = L(0)*u + L(1)*v + L(2);
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero(); // wrt point (1x3)
        _jacobianOplusXj.setZero(); // wrt pose  (1x6)

        const auto* vP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* vT = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

        const Eigen::Vector3d Xw = vP->estimate();
        const Eigen::Vector3d Xc = vT->estimate().map(Xw);
        if (Xc(2) <= 1e-9 || !Xc.allFinite()) return;

        const Eigen::Matrix3d R = vT->estimate().rotation().toRotationMatrix();
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        const double z2 = z*z;

        // duv/dXc (2x3)
        Eigen::Matrix<double,2,3> Jpi;
        Jpi << fx/z,   0,   -fx*x/z2,
                 0, fy/z,   -fy*y/z2;

        const Eigen::RowVector2d ab(_measurement(0), _measurement(1)); // (a,b)
        const Eigen::RowVector3d Jimg = ab * Jpi; // 1x3  (dres/dXc)

        // dXc/dXw = R
        _jacobianOplusXi = Jimg * R; // 1x3

        // dXc/dxi (left perturb) = [ -skew(Xc) | I ]
        Eigen::Matrix<double,3,6> Jse3;
        Jse3.setZero();
        Jse3(0,1)= z;   Jse3(0,2)=-y;
        Jse3(1,0)=-z;   Jse3(1,2)= x;
        Jse3(2,0)= y;   Jse3(2,1)=-x;
        Jse3.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        _jacobianOplusXj = Jimg * Jse3; // 1x6

        // safety
        if (!std::isfinite(_jacobianOplusXi.sum()) || !std::isfinite(_jacobianOplusXj.sum()))
        {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }

        //mJPoint = Jimg * R;
        //mJPose = Jimg * Jse3;
    }

    // ---- SAFE getters (不要用 jacobianOplusXi() 这种版本相关接口) ----
    //inline const Eigen::Matrix<double,1,3>& JPoint() const { return mJPoint; }
   // inline const Eigen::Matrix<double,1,6>& JPose()  const { return mJPose; }

    //Eigen::Matrix<double, 1, 3> mJPoint;
    //Eigen::Matrix<double, 1, 6> mJPose;
    double fx, fy, cx, cy;
};

// Edge: line segment length prior
// 1 residual: length prior
class EdgeLineLengthPrior
    : public g2o::BaseBinaryEdge<1, double,
                                 g2o::VertexSBAPointXYZ,
                                 g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeLineLengthPrior(double L0, double lambda)
        : L0_(L0), lambda_(lambda) {}

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void computeError() override
    {
        const auto* v1 =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* v2 =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);

        const Eigen::Vector3d d = v2->estimate() - v1->estimate();
        const double len = d.norm();

        _error[0] = lambda_ * (len - L0_);
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero(); // wrt P1
        _jacobianOplusXj.setZero(); // wrt P2

        const auto* v1 =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* v2 =
            static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);

        const Eigen::Vector3d d = v2->estimate() - v1->estimate();
        const double len = d.norm();
        if (len < 1e-9) return;

        const Eigen::RowVector3d J =
            lambda_ * (d / len).transpose();

        _jacobianOplusXi = -J;
        _jacobianOplusXj =  J;
    }

private:
    double L0_;
    double lambda_;
};

class EdgeLineDirectionPrior
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // measurement: d0_unit (must be normalized)
    EdgeLineDirectionPrior(const Eigen::Vector3d& d0_unit, double lambdaD)
        : d0_(d0_unit), lambdaD_(lambdaD) {}

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void computeError() override
    {
        const auto* v1 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        Eigen::Vector3d d = v2->estimate() - v1->estimate();
        if (!d.allFinite() || !d0_.allFinite()) {
            _error.setZero();
            return;
        }
        // r = lambda * (d x d0)
        _error = lambdaD_ * (d.cross(d0_));
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero(); // 3x3 wrt P1
        _jacobianOplusXj.setZero(); // 3x3 wrt P2

        if (!d0_.allFinite()) return;

        // d x d0 = - [d0]_x d
        // d = P2 - P1
        // J_P1 = lambda * [d0]_x
        // J_P2 = -lambda * [d0]_x
        Eigen::Matrix3d Sd0 = Skew(d0_);
        _jacobianOplusXi = +lambdaD_ * Sd0;
        _jacobianOplusXj = -lambdaD_ * Sd0;
    }

private:
    inline Eigen::Matrix3d Skew(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d S;
        S <<     0, -v.z(),  v.y(),
            v.z(),     0, -v.x(),
            -v.y(),  v.x(),     0;
        return S;
    }
    Eigen::Vector3d d0_;
    double lambdaD_ = 1.0;
};

class EdgeLineMidpointPrior
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // measurement: m0
    EdgeLineMidpointPrior(const Eigen::Vector3d& m0, double lambdaM)
        : m0_(m0), lambdaM_(lambdaM) {}

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void computeError() override
    {
        const auto* v1 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const auto* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        Eigen::Vector3d m = 0.5 * (v1->estimate() + v2->estimate());
        if (!m.allFinite() || !m0_.allFinite()) {
            _error.setZero();
            return;
        }
        _error = lambdaM_ * (m - m0_);
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero(); // 3x3
        _jacobianOplusXj.setZero(); // 3x3

        Eigen::Matrix3d J = 0.5 * lambdaM_ * Eigen::Matrix3d::Identity();
        _jacobianOplusXi = J;
        _jacobianOplusXj = J;
    }

private:
    Eigen::Vector3d m0_;
    double lambdaM_ = 1.0;
};


// 4 residuals: 加入了正则化项 长度保持（Length Regularization） 和 方向保持（Direction Regularization）
// Residuals:
//   Given observed 2D line (a,b,c): a*u + b*v + c = 0 
//   rL​=λL​(∥P1​−P2​∥−L0​); rD​=λD​⋅∥d×d0​∥
//   e0: point1 signed distance to observed 2D line
//   e1: point2 signed distance to observed 2D line
//   e2: length regularization
//   e3: direction regularization
class EdgeSE3ProjectLine_PoseAndPoints_Reg
    : public g2o::BaseMultiEdge<4, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLine_PoseAndPoints_Reg()
        : fx(0), fy(0), cx(0), cy(0), a(0), b(0), c(0),
          wLength(1e-3), wDirection(1e-3),
          length0(0.0)
    {
        resize(3); // 0: pose, 1: endpoint P1, 2: endpoint P2
        dir0.setZero();
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // ---------------- intrinsics ----------------
    inline void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    // ---------------- observed 2D line ----------------
    inline void SetObservedLineByEndpoints(double u1,double v1,double u2,double v2)
    {
        double dx = u2 - u1;
        double dy = v2 - v1;
        double na = dy;
        double nb = -dx;
        double norm = std::sqrt(na*na + nb*nb);
        if(norm < 1e-12){ a=b=c=0.0; return; }
        a = na / norm;
        b = nb / norm;
        c = -(a*u1 + b*v1);
    }

    inline void SetObservedLineABC(double _a,double _b,double _c)
    {
        double norm = std::sqrt(_a*_a + _b*_b);
        if(norm < 1e-12){ a=_a; b=_b; c=_c; return; }
        a = _a / norm;
        b = _b / norm;
        c = _c / norm;
    }

    // ---------------- regularization config ----------------
    inline void SetRegularizationWeights(double w_len, double w_dir)
    {
        wLength = w_len;
        wDirection = w_dir;
    }

    // Call ONCE when you create line endpoint vertices (use current MapLine endpoints).
    inline void SetInitialLine(const Eigen::Vector3d& P1w0,
                               const Eigen::Vector3d& P2w0)
    {
        Eigen::Vector3d d0 = P1w0 - P2w0;
        length0 = d0.norm();
        if(length0 > 1e-9) dir0 = d0 / length0;
        else dir0.setZero();
    }

    // ---------------- compute error ----------------
    void computeError() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        const auto* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const auto* vP1   = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const auto* vP2   = static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        const Eigen::Vector3d Pw1 = vP1->estimate();
        const Eigen::Vector3d Pw2 = vP2->estimate();

        const Eigen::Vector3d Xc1 = vPose->estimate().map(Pw1);
        const Eigen::Vector3d Xc2 = vPose->estimate().map(Pw2);

        // If behind camera: make error large (keep stable)
        if (Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9) {
            _error.setConstant(1e3);
            return;
        }

        const Eigen::Vector2d uv1 = cam2pixel(Xc1);
        const Eigen::Vector2d uv2 = cam2pixel(Xc2);

        // e0,e1: point-to-line distance
        _error(0) = a*uv1(0) + b*uv1(1) + c;
        _error(1) = a*uv2(0) + b*uv2(1) + c;

        // regularization: based in WORLD endpoints (directly on vertex estimates)
        const Eigen::Vector3d d = Pw1 - Pw2;
        const double L = d.norm();

        // e2: length
        _error(2) = wLength * (L - length0);

        // e3: direction (sin(theta) = ||dir x dir0||)
        if (L > 1e-9 && dir0.squaredNorm() > 0.5) {
            const Eigen::Vector3d dir = d / L;
            const Eigen::Vector3d crossv = dir.cross(dir0);
            _error(3) = wDirection * crossv.norm();
        } else {
            _error(3) = 0.0;
        }
    }

    // ---------------- analytic Jacobians ----------------
    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;
        using g2o::VertexSBAPointXYZ;

        const auto* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const auto* vP1   = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);
        const auto* vP2   = static_cast<const VertexSBAPointXYZ*>(_vertices[2]);

        const Eigen::Vector3d Pw1 = vP1->estimate();
        const Eigen::Vector3d Pw2 = vP2->estimate();

        const Eigen::Vector3d Xc1 = vPose->estimate().map(Pw1);
        const Eigen::Vector3d Xc2 = vPose->estimate().map(Pw2);

        // init all jacobians to zero
        _jacobianOplus[0].setZero(); // 4x6
        _jacobianOplus[1].setZero(); // 4x3
        _jacobianOplus[2].setZero(); // 4x3

        if (Xc1(2) <= 1e-9 || Xc2(2) <= 1e-9) {
            return;
        }

        const Eigen::Matrix3d Rcw = vPose->estimate().rotation().toRotationMatrix();

        // ---- image part (e0,e1) ----
        const Eigen::Matrix<double,2,3> Jpi1 = projectionJacobian_Xc(Xc1);
        const Eigen::Matrix<double,2,3> Jpi2 = projectionJacobian_Xc(Xc2);

        const Eigen::Matrix<double,3,6> dXc1_dxi = dXc_dxi_from_Xc(Xc1); // right perturbation
        const Eigen::Matrix<double,3,6> dXc2_dxi = dXc_dxi_from_Xc(Xc2);

        const Eigen::Matrix<double,2,6> Jpose1 = Jpi1 * dXc1_dxi;
        const Eigen::Matrix<double,2,6> Jpose2 = Jpi2 * dXc2_dxi;

        const Eigen::RowVector2d ab(a,b);

        // pose jac: e0,e1
        _jacobianOplus[0].row(0) = ab * Jpose1;
        _jacobianOplus[0].row(1) = ab * Jpose2;

        // point jac (world point -> camera): dXc/dPw = Rcw
        // e0 depends only on P1; e1 depends only on P2
        _jacobianOplus[1].row(0) = ab * Jpi1 * Rcw;
        _jacobianOplus[2].row(1) = ab * Jpi2 * Rcw;

        // ---- regularization part (e2,e3) ----
        const Eigen::Vector3d d = Pw1 - Pw2;
        const double L = d.norm();

        // e2 = wLength*(L - L0)
        if (L > 1e-9) {
            const Eigen::RowVector3d dL_dPw1 = (d / L).transpose();   // 1x3
            const Eigen::RowVector3d dL_dPw2 = -(d / L).transpose();  // 1x3

            _jacobianOplus[1].row(2) = wLength * dL_dPw1;
            _jacobianOplus[2].row(2) = wLength * dL_dPw2;
        }

        // e3 = wDirection * || dir x dir0 ||,  dir = d/L
        // Let c = dir x dir0, s = ||c||
        // dr/d(dir) = wDirection*(1/s)* c^T * (-[dir0]_x)  (1x3)
        // d(dir)/d(d) = (I - dir dir^T)/L
        // d(d)/dPw1 = I, d(d)/dPw2 = -I
        if (L > 1e-9 && dir0.squaredNorm() > 0.5) {
            const Eigen::Vector3d dir = d / L;
            const Eigen::Vector3d cvec = dir.cross(dir0);
            const double s = cvec.norm();

            if (s > 1e-12) {
                // -[dir0]_x
                Eigen::Matrix3d minus_dir0_hat;
                minus_dir0_hat <<  0.0,      dir0(2), -dir0(1),
                                  -dir0(2),  0.0,      dir0(0),
                                   dir0(1), -dir0(0),  0.0;

                // (1/s) * c^T * (-[dir0]_x)  => 1x3
                const Eigen::RowVector3d dr_dDir =
                    (wDirection / s) * (cvec.transpose() * minus_dir0_hat);

                // dDir/dD = (I - dir*dir^T)/L
                const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                const Eigen::Matrix3d dDir_dD = (I - dir*dir.transpose()) / L;

                const Eigen::RowVector3d dr_dD = dr_dDir * dDir_dD; // 1x3

                _jacobianOplus[1].row(3) = dr_dD;     // Pw1
                _jacobianOplus[2].row(3) = -dr_dD;    // Pw2
            }
        }

        // pose does not affect (e2,e3) because we regularize in WORLD endpoints
        // _jacobianOplus[0].row(2/3) are already zero.
    }

private:
    // ---------------- helpers ----------------
    inline Eigen::Vector2d cam2pixel(const Eigen::Vector3d& Xc) const {
        return Eigen::Vector2d(
            fx * Xc(0) / Xc(2) + cx,
            fy * Xc(1) / Xc(2) + cy
        );
    }

    inline Eigen::Matrix<double,2,3>
    projectionJacobian_Xc(const Eigen::Vector3d &Xc) const
    {
        const double x = Xc(0), y = Xc(1), z = Xc(2);
        const double z2 = z*z;
        Eigen::Matrix<double,2,3> J;
        J << fx/z,     0, -fx*x/z2,
                 0, fy/z, -fy*y/z2;
        return J;
    }

    // right perturbation (VertexSE3Expmap typical):
    // dXc/dxi = [ -skew(Xc) | I ]
    inline Eigen::Matrix<double,3,6>
    dXc_dxi_from_Xc(const Eigen::Vector3d& Xc) const
    {
        Eigen::Matrix<double,3,6> J;
        J.setZero();

        const double x = Xc(0);
        const double y = Xc(1);
        const double z = Xc(2);

        // -skew(Xc)
        J(0,1) =  z;   J(0,2) = -y;
        J(1,0) = -z;   J(1,2) =  x;
        J(2,0) =  y;   J(2,1) = -x;

        // translation
        J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
        return J;
    }

public:
    // intrinsics
    double fx, fy, cx, cy;

    // observed 2D line (normalized): a*u + b*v + c = 0
    double a, b, c;

    // regularization
    double wLength;
    double wDirection;

    double length0;        // initial length
    Eigen::Vector3d dir0;  // initial direction (unit)
};




// Edge for SE3 pose and Plucker line optimization (通过测试，可用， 需要再测试数值是否正确)
class EdgeSE3ProjectPluckerLine_PoseAndLine
    : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectPluckerLine_PoseAndLine()
        : fx(0), fy(0), cx(0), cy(0), a(0), b(0), c(0)
    {
        resize(2); // vertex 0: SE3, vertex 1: Plucker line
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    inline void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    inline void SetObservedLineABC(double _a,double _b,double _c)
    {
        double n = std::sqrt(_a*_a + _b*_b);
        if(n < 1e-12) { a=_a; b=_b; c=_c; return; }
        a=_a/n; b=_b/n; c=_c/n;
    }

    //inline std::vector<Eigen::Matrix<double,2,6>> jacobians() const {
    //   return _jacobianOplus;
    //}

    void computeError() override
    {
        using g2o::VertexSE3Expmap;

        const VertexSE3Expmap* vPose =
            static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine =
            static_cast<const VertexLinePlucker*>(_vertices[1]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix<double,6,1> Lw = vLine->estimate();

        Eigen::Vector3d n_w = Lw.head<3>();
        Eigen::Vector3d v_w = Lw.tail<3>();

        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Vector3d n_c = Rcw*n_w + tcw.cross(Rcw*v_w);

        Eigen::Vector3d l;
        l(0) = fx * n_c(0) + cx * n_c(2);
        l(1) = fy * n_c(1) + cy * n_c(2);
        l(2) = n_c(2);

        _error(0) = l(0) - a;
        _error(1) = l(1) - b;
    }

    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;

        const VertexSE3Expmap* vPose =
            static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine =
            static_cast<const VertexLinePlucker*>(_vertices[1]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix<double,6,1> Lw = vLine->estimate();

        Eigen::Vector3d n_w = Lw.head<3>();
        Eigen::Vector3d v_w = Lw.tail<3>();

        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Vector3d n_c = Rcw*n_w + tcw.cross(Rcw*v_w);

        // --- Jacobian wrt pose (2x6) ---
        Eigen::Matrix<double,3,3> dldnc;
        dldnc << fx, 0, cx,
                 0, fy, cy,
                 0,  0, 1;

        Eigen::Matrix<double,3,6> dncdxi;
        dncdxi.setZero();
        //dncdxi.block<3,3>(0,0) = -skew(Rcw*v_w);
        //dncdxi.block<3,3>(0,3) = -Rcw*skew(n_w) - skew(tcw)*Rcw*skew(v_w);
        dncdxi.block<3,3>(0,0) = -Rcw*skew(n_w) - skew(tcw)*Rcw*skew(v_w);  // d n_c / d rotation
        dncdxi.block<3,3>(0,3) = -skew(Rcw * v_w);                         // d n_c / d translation

        Eigen::Matrix<double,3,6> dldxi = dldnc * dncdxi;

        Eigen::Matrix<double,2,6> Jpose;
        Jpose.row(0) = dldxi.row(0);
        Jpose.row(1) = dldxi.row(1);
        _jacobianOplus[0] = Jpose;  // 直接赋值，不用 setZero
        mJacobianPose = Jpose;  //test store

        // --- Jacobian wrt line (2x6) ---
        //Eigen::Matrix<double,2,6> Jline;
        //Jline.setZero();
        //Jline.block<2,3>(0,0) = dldnc.block<2,3>(0,0) * Rcw;
        //Jline.block<2,3>(0,3) = dldnc.block<2,3>(0,0) * skew(tcw) * Rcw;
        // --- Line Jacobian (2x6) ---
        Eigen::Matrix<double,3,3> dnc_dnw = Rcw;
        Eigen::Matrix<double,3,3> dnc_dvw = skew(tcw) * Rcw;

        Eigen::Matrix<double,3,6> dnc_dLw;
        dnc_dLw.block<3,3>(0,0) = dnc_dnw;
        dnc_dLw.block<3,3>(0,3) = dnc_dvw;
        Eigen::Matrix<double,3,6> dldLw = dldnc * dnc_dLw;
        Eigen::Matrix<double,2,6> Jline;
        Jline.row(0) = dldLw.row(0);
        Jline.row(1) = dldLw.row(1);

        _jacobianOplus[1] = Jline;
        mJacobianLine = Jline;  //test store
    }

    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
              v(2),    0, -v(0),
             -v(1), v(0),   0;
        return S;
    }

    inline Eigen::Matrix<double,2,6> getJacobianPose() const {
        
       return mJacobianPose;
    }

    inline Eigen::Matrix<double,2,6> getJacobianLine() const {
       return mJacobianLine;
    }
    

private:
    Eigen::Matrix<double,2,6> mJacobianPose;
    Eigen::Matrix<double,2,6> mJacobianLine;
    double fx, fy, cx, cy;
    double a, b, c;
};


class EdgeSE3ProjectPluckerLine_PoseOnly
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectPluckerLine_PoseOnly()
        : fx(0), fy(0), cx(0), cy(0)
    {}

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; }

    void SetObservedPluckerLine(const Eigen::Vector3d& n, const Eigen::Vector3d& v)
    {
        n_w = n;
        v_w = v;
    }

    void SetObservedLineABC(double _a,double _b,double _c)
    {
        double n = std::sqrt(_a*_a + _b*_b);
        if(n < 1e-12) { a=_a; b=_b; c=_c; return; }
        a=_a/n; b=_b/n; c=_c/n;
    }

    void computeError() override
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Vector3d n_c = Rcw*n_w + tcw.cross(Rcw*v_w);

        Eigen::Vector3d l;
        l(0) = fx * n_c(0) + cx * n_c(2);
        l(1) = fy * n_c(1) + cy * n_c(2);
        l(2) = n_c(2);

        _error(0) = l(0) - a;
        _error(1) = l(1) - b;
    }

    void linearizeOplus() override
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Matrix<double,3,3> dldnc;
        dldnc << fx, 0, cx,
                 0, fy, cy,
                 0,  0, 1;

        Eigen::Matrix<double,3,6> dncdxi;
        dncdxi.setZero();
        dncdxi.block<3,3>(0,0) = -skew(Rcw*v_w);
        dncdxi.block<3,3>(0,3) = -Rcw*skew(n_w) - skew(tcw)*Rcw*skew(v_w);

        Eigen::Matrix<double,3,6> dldxi = dldnc * dncdxi;

        // 直接写入BaseUnaryEdge提供的 _jacobianOplusXi
        _jacobianOplusXi.block<2,6>(0,0) = dldxi.block<2,6>(0,0);
    }

private:
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
              v(2),    0, -v(0),
             -v(1), v(0),   0;
        return S;
    }

private:
    double fx, fy, cx, cy;
    double a=0, b=0, c=0;
    Eigen::Vector3d n_w;
    Eigen::Vector3d v_w;
};

//LineSegmentToPlucker
class EdgeStereoSE3ProjectPluckerLine_PoseOnly
    : public g2o::BaseUnaryEdge<4, Eigen::Vector4d, g2o::VertexSE3Expmap>
{
public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereoSE3ProjectPluckerLine_PoseOnly()
        : fx(0), fy(0), cx(0), cy(0), bf(0) 
    {
        // 只有一个顶点：相机位姿
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    inline void SetCameraIntrinsics(double _fx,double _fy,double _cx,double _cy, double _bf)
    { fx=_fx; fy=_fy; cx=_cx; cy=_cy; bf=_bf; }

    inline void SetObservedPluckerLine(const Eigen::Vector3d& n, const Eigen::Vector3d& v)
    {
        n_w = n;
        v_w = v;
    }

    inline void SetObservedLineStereo(double uL1, double vL1, double uL2, double vL2)
    {
        obs(0) = uL1; obs(1) = vL1;
        obs(2) = uL2; obs(3) = vL2;
    }

    void computeError() override
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Vector3d n_c = Rcw*n_w + tcw.cross(Rcw*v_w);

        Eigen::Vector2d uvL, uvR;
        uvL(0) = fx*n_c(0)/n_c(2) + cx;
        uvL(1) = fy*n_c(1)/n_c(2) + cy;
        uvR(0) = uvL(0) - bf / n_c(2); // 双目视差
        uvR(1) = uvL(1);

        _error(0) = uvL(0) - obs(0);
        _error(1) = uvL(1) - obs(1);
        _error(2) = uvR(0) - obs(2);
        _error(3) = uvR(1) - obs(3);
    }

    void linearizeOplus() override
    {
        const g2o::VertexSE3Expmap* vPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        Eigen::Vector3d n_c = Rcw*n_w + tcw.cross(Rcw*v_w);

        // --- Jacobian wrt pose (4x6) ---
        Eigen::Matrix<double,3,6> dncdxi;
        dncdxi.setZero();
        dncdxi.block<3,3>(0,0) = -skew(Rcw*v_w);
        dncdxi.block<3,3>(0,3) = -Rcw*skew(n_w) - skew(tcw)*Rcw*skew(v_w);

        Eigen::Matrix<double,4,6> Jpose;
        double x = n_c(0), y = n_c(1), z = n_c(2);
        double z2 = z*z;

        // 左相机 uL,vL
        Jpose(0,0) = fx / z * dncdxi(0,0) - fx * x / z2 * dncdxi(2,0); // duL/dxi
        Jpose(0,1) = fx / z * dncdxi(0,1) - fx * x / z2 * dncdxi(2,1);
        Jpose(0,2) = fx / z * dncdxi(0,2) - fx * x / z2 * dncdxi(2,2);
        Jpose(0,3) = fx / z * dncdxi(0,3) - fx * x / z2 * dncdxi(2,3);
        Jpose(0,4) = fx / z * dncdxi(0,4) - fx * x / z2 * dncdxi(2,4);
        Jpose(0,5) = fx / z * dncdxi(0,5) - fx * x / z2 * dncdxi(2,5);

        Jpose(1,0) = fy / z * dncdxi(1,0) - fy * y / z2 * dncdxi(2,0); // dvL/dxi
        Jpose(1,1) = fy / z * dncdxi(1,1) - fy * y / z2 * dncdxi(2,1);
        Jpose(1,2) = fy / z * dncdxi(1,2) - fy * y / z2 * dncdxi(2,2);
        Jpose(1,3) = fy / z * dncdxi(1,3) - fy * y / z2 * dncdxi(2,3);
        Jpose(1,4) = fy / z * dncdxi(1,4) - fy * y / z2 * dncdxi(2,4);
        Jpose(1,5) = fy / z * dncdxi(1,5) - fy * y / z2 * dncdxi(2,5);

        // 右相机 uR,vR
        Jpose(2,0) = Jpose(0,0) - bf / z2 * dncdxi(2,0); // duR/dxi
        Jpose(2,1) = Jpose(0,1) - bf / z2 * dncdxi(2,1);
        Jpose(2,2) = Jpose(0,2) - bf / z2 * dncdxi(2,2);
        Jpose(2,3) = Jpose(0,3) - bf / z2 * dncdxi(2,3);
        Jpose(2,4) = Jpose(0,4) - bf / z2 * dncdxi(2,4);
        Jpose(2,5) = Jpose(0,5) - bf / z2 * dncdxi(2,5);

        Jpose(3,0) = Jpose(1,0);
        Jpose(3,1) = Jpose(1,1);
        Jpose(3,2) = Jpose(1,2);
        Jpose(3,3) = Jpose(1,3);
        Jpose(3,4) = Jpose(1,4);
        Jpose(3,5) = Jpose(1,5);

        _jacobianOplusXi = Jpose;
    }

    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
              v(2),    0, -v(0),
             -v(1), v(0),   0;
        return S;
    }

private:
    double fx, fy, cx, cy, bf;  // bf = baseline*fx
    Eigen::Vector3d n_w, v_w;
    Eigen::Vector4d obs;         // uL,vL,uR,vR
};

class EdgeSE3ProjectPluckerLine_OnlyPoseToBody
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectPluckerLine_OnlyPoseToBody()
        : fx(0), fy(0), cx(0), cy(0)
    {
        // 顶点是机体相对于世界的位姿
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    void SetCameraIntrinsics(double _fx, double _fy, double _cx, double _cy)
    {
        fx = _fx; fy = _fy; cx = _cx; cy = _cy;
    }

    // 设置观测 Plucker 线在相机坐标系下的 n,v
    void SetObservedPluckerLineCamera(const Eigen::Vector3d& n_cam, const Eigen::Vector3d& v_cam)
    {
        n_c = n_cam;
        v_c = v_cam;
    }

    // 设置观测线 (a,b,c)
    void SetObservedLineABC(double _a,double _b,double _c)
    {
        double norm = std::sqrt(_a*_a + _b*_b);
        if(norm < 1e-12) { a=_a; b=_b; c=_c; return; }
        a = _a / norm; b = _b / norm; c = _c / norm;
    }

    void computeError() override
    {
        const g2o::VertexSE3Expmap* vBodyPose =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat Twb = vBodyPose->estimate();  // body -> world
        Eigen::Matrix3d Rwb = Twb.rotation().toRotationMatrix();
        Eigen::Vector3d twb = Twb.translation();

        // 将机体坐标系下的 Plucker 线投影到相机坐标
        Eigen::Vector3d n_cam_proj = n_c + twb.cross(v_c);  // 简化为机体位移作用
        Eigen::Vector2d uv;
        uv(0) = fx * n_cam_proj(0) + cx * n_cam_proj(2);
        uv(1) = fy * n_cam_proj(1) + cy * n_cam_proj(2);

        _error(0) = uv(0) - a;
        _error(1) = uv(1) - b;
    }

    void linearizeOplus() override
    {
        // 顶点是机体 SE3，雅可比类似 Pose-only
        Eigen::Matrix3d dldnc;
        dldnc << fx, 0, cx,
                 0, fy, cy,
                 0,  0, 1;

        Eigen::Matrix<double,3,6> dncdxi;
        dncdxi.setZero();
        dncdxi.block<3,3>(0,0) = -skew(v_c);        // translation
        dncdxi.block<3,3>(0,3) = -skew(n_c);        // rotation

        Eigen::Matrix<double,3,6> dldxi = dldnc * dncdxi;

        Eigen::Matrix<double,2,6> Jpose;
        Jpose.row(0) = dldxi.row(0);
        Jpose.row(1) = dldxi.row(1);

        _jacobianOplusXi = Jpose;
    }

    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
              v(2),    0, -v(0),
             -v(1), v(0),   0;
        return S;
    }

private:
    double fx, fy, cx, cy;
    double a=0, b=0, c=0;
    Eigen::Vector3d n_c;  // Plucker n in camera frame
    Eigen::Vector3d v_c;  // Plucker v in camera frame
};


//TO CHECK NEXT
class EdgeSE3ProjectCameraEndPointToPluckerLine
    : public g2o::BaseMultiEdge<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectCameraEndPointToPluckerLine()
    {
        resize(2); // 0: SE3 pose (Tcw), 1: Plucker line (n_w, v_w)
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // Set the observed 3D endpoint in the camera coordinate (p_c_obs)
    inline void setMeasurement(const Eigen::Vector3d& p) { _measurement = p; }

    void computeError() override
    {
        using g2o::VertexSE3Expmap;
        const VertexSE3Expmap* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine = static_cast<const VertexLinePlucker*>(_vertices[1]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix<double,6,1> Lw = vLine->estimate();
        Eigen::Vector3d n_w = Lw.head<3>(), v_w = Lw.tail<3>();

        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        // transform line to camera frame
        Eigen::Vector3d v_c = Rcw * v_w;
        Eigen::Vector3d n_c = Rcw * n_w + tcw.cross(v_c);

        // choose a point on line: p0 = (v x n) / (v.v)
        double s = v_c.dot(v_c);
        Eigen::Vector3d p0_c = Eigen::Vector3d::Zero();
        if (s > 1e-12) p0_c = v_c.cross(n_c) / s;

        Eigen::Vector3d p = _measurement;
        Eigen::Vector3d r = p.cross(v_c) - p0_c.cross(v_c); // (p - p0) x v_c
        _error = r;
    }

    // numeric-differentiation for the small 3x3 blocks, combined with analytic blocks for pose/line transforms
    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;
        const VertexSE3Expmap* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine = static_cast<const VertexLinePlucker*>(_vertices[1]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix<double,6,1> Lw = vLine->estimate();
        Eigen::Vector3d n_w = Lw.head<3>(), v_w = Lw.tail<3>();

        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        // intermediates
        Eigen::Vector3d v_c = Rcw * v_w;
        Eigen::Vector3d n_c = Rcw * n_w + tcw.cross(v_c);
        double s = v_c.dot(v_c);
        Eigen::Vector3d p0_c = Eigen::Vector3d::Zero();
        if (s > 1e-12) p0_c = v_c.cross(n_c) / s;
        Eigen::Vector3d p = _measurement;

        // compute residual again (for safety)
        Eigen::Vector3d r = p.cross(v_c) - p0_c.cross(v_c);

        // ---------- compute partials of r wrt v_c and n_c (3x3 blocks)
        // Use small finite difference for these 3x3 blocks (robust & short)
        Eigen::Matrix<double,3,3> Jr_vc = Eigen::Matrix<double,3,3>::Zero();
        Eigen::Matrix<double,3,3> Jr_nc = Eigen::Matrix<double,3,3>::Zero();
        const double eps = 1e-8;
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d dv = Eigen::Vector3d::Zero(); dv[i] = eps;
            Eigen::Vector3d vcp = v_c + dv;
            Eigen::Vector3d ncp = n_c;
            double sp = vcp.dot(vcp);
            Eigen::Vector3d p0p = Eigen::Vector3d::Zero();
            if (sp > 1e-12) p0p = vcp.cross(ncp) / sp;
            Eigen::Vector3d rp = p.cross(vcp) - p0p.cross(vcp);
            Jr_vc.col(i) = (rp - r) / eps;

            Eigen::Vector3d dn = Eigen::Vector3d::Zero(); dn[i] = eps;
            vcp = v_c; ncp = n_c + dn;
            sp = vcp.dot(vcp);
            p0p = Eigen::Vector3d::Zero();
            if (sp > 1e-12) p0p = vcp.cross(ncp) / sp;
            rp = p.cross(vcp) - p0p.cross(vcp);
            Jr_nc.col(i) = (rp - r) / eps;
        }

        // ---------- compute analytic Jacobians of v_c,n_c wrt pose (6) and wrt line parameters (n_w,v_w) (6)
        // stack [v_c; n_c] (6x1)
        // For pose (using left perturbation: exp(dxi) * Tcw)
        // compute numeric directional derivatives for [v_c; n_c] wrt 6-dof twist
        Eigen::Matrix<double,6,6> J_vn_pose; J_vn_pose.setZero();
        {
            const double eps2 = 1e-7;
            Eigen::Matrix<double,6,1> base;
            base.block<3,1>(0,0) = v_c;
            base.block<3,1>(3,0) = n_c;
            for (int i = 0; i < 6; ++i) {
                Eigen::Matrix<double,6,1> xi = Eigen::Matrix<double,6,1>::Zero();
                xi(i) = eps2;
                Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
                Eigen::Vector3d dt = tcw;
                if (i < 3) {
                    // small rotation left-multiplied
                    Eigen::Vector3d w = xi.segment<3>(0);
                    Eigen::Matrix3d W; W << 0,-w(2),w(1), w(2),0,-w(0), -w(1),w(0),0;
                    dR = (Eigen::Matrix3d::Identity() + W); // linear approx of exp(W)
                    // apply dR * Rcw
                    dR = dR * Rcw;
                } else {
                    dR = Rcw;
                    dt(i-3) += eps2;
                }
                Eigen::Vector3d vcp = dR * v_w;
                Eigen::Vector3d ncp = dR * n_w + dt.cross(vcp);
                Eigen::Matrix<double,6,1> vp; vp.block<3,1>(0,0)=vcp; vp.block<3,1>(3,0)=ncp;
                J_vn_pose.col(i) = (vp - base) / eps2;
            }
        }

        // For line parameters [n_w; v_w] (columns order: n_w(3), v_w(3)):
        Eigen::Matrix<double,6,6> J_vn_line; J_vn_line.setZero();
        // rows 0..2 -> v_c, rows 3..5 -> n_c
        // dv_c/dn_w = 0 ; dv_c/dv_w = Rcw
        J_vn_line.block<3,3>(0,0).setZero();
        J_vn_line.block<3,3>(0,3) = Rcw;
        // dn_c/dn_w = Rcw ; dn_c/dv_w = skew(tcw) * Rcw
        Eigen::Matrix3d S_t; S_t << 0,-tcw(2),tcw(1), tcw(2),0,-tcw(0), -tcw(1),tcw(0),0;
        J_vn_line.block<3,3>(3,0) = Rcw;
        J_vn_line.block<3,3>(3,3) = S_t * Rcw;

        // ---------- compose: Jr_pose = [Jr_vc Jr_nc] * J_vn_pose
        Eigen::Matrix<double,3,6> Jpose = Eigen::Matrix<double,3,6>::Zero();
        Eigen::Matrix<double,3,6> Jr_vn; Jr_vn.setZero();
        Jr_vn.block<3,3>(0,0) = Jr_vc;
        Jr_vn.block<3,3>(0,3) = Jr_nc;
        Jpose = Jr_vn * J_vn_pose;

        // ---------- compose: Jr_line = Jr_vn * J_vn_line
        Eigen::Matrix<double,3,6> Jline = Jr_vn * J_vn_line;

        _jacobianOplus[0] = Jpose;
        _jacobianOplus[1] = Jline;
    }

protected:
    Eigen::Vector3d _measurement; // p_c_obs
};


// Edge: 连接 4 个顶点： pose (SE3), line (Plucker), depth0, depth1
// 残差：4-dim = [2 dims for endpoint0, 2 dims for endpoint1]
class EdgeLineEndpointOnPluckerLine
    : public g2o::BaseMultiEdge<4, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    // 构造时传入两个像素端点的方向向量（K^-1 * [u,v,1]），保持常量
    // dir0, dir1: Eigen::Vector3d (camera-frame ray directions, not normalized required)
    EdgeLineEndpointOnPluckerLine(const Eigen::Vector3d& dir0,
                                  const Eigen::Vector3d& dir1)
        : _dir0(dir0), _dir1(dir1)
    {
        resize(4); // pose, line, depth0, depth1
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // 设置观测（不需要额外 measurement，因为 dir 已包含像素位置）
    inline void setCameraIntrinsics(double fx,double fy,double cx,double cy) {
        fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
    }

    // compute two orthonormal basis vectors perpendicular to v (camera-frame)
    void build_orthonormal_basis(const Eigen::Vector3d& v,
                                 Eigen::Vector3d& e1,
                                 Eigen::Vector3d& e2) const
    {
        // choose a vector not parallel to v
        Eigen::Vector3d tmp = (std::abs(v.z()) < 0.9) ? Eigen::Vector3d(0,0,1) : Eigen::Vector3d(0,1,0);
        e1 = v.cross(tmp);
        if (e1.norm() < 1e-12) {
            tmp = Eigen::Vector3d(1,0,0);
            e1 = v.cross(tmp);
        }
        e1.normalize();
        e2 = v.cross(e1);
        e2.normalize();
    }

    void computeError() override
    {
        using g2o::VertexSE3Expmap;
        const VertexSE3Expmap* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine = static_cast<const VertexLinePlucker*>(_vertices[1]);
        const VertexDepth* vD0 = static_cast<const VertexDepth*>(_vertices[2]);
        const VertexDepth* vD1 = static_cast<const VertexDepth*>(_vertices[3]);

        g2o::SE3Quat Tcw = vPose->estimate();
        const Eigen::Matrix<double,6,1> Lw = vLine->estimate();
        const double d0 = vD0->estimate();
        const double d1 = vD1->estimate();

        Eigen::Vector3d n_w = Lw.head<3>(), v_w = Lw.tail<3>();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        // transform line to camera frame
        Eigen::Vector3d v_c = Rcw * v_w;
        Eigen::Vector3d n_c = Rcw * n_w + tcw.cross(v_c);

        // point on line p0 = (v x n) / (v.v)
        double s = v_c.dot(v_c);
        Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
        if (s > 1e-12) p0 = v_c.cross(n_c) / s;

        // endpoints in camera frame: p_ci = di * dir_i
        Eigen::Vector3d p0_c = d0 * _dir0;
        Eigen::Vector3d p1_c = d1 * _dir1;

        // two orthonormal basis e1,e2 perpendicular to v_c
        Eigen::Vector3d e1, e2;
        build_orthonormal_basis(v_c, e1, e2);

        // residuals: projection of (p_ci - p0) on e1,e2
        Eigen::Vector2d r0, r1;
        Eigen::Vector3d delta0 = p0_c - p0;
        Eigen::Vector3d delta1 = p1_c - p0;
        r0(0) = e1.dot(delta0);
        r0(1) = e2.dot(delta0);
        r1(0) = e1.dot(delta1);
        r1(1) = e2.dot(delta1);

        _error.resize(4);
        _error(0) = r0(0); _error(1) = r0(1);
        _error(2) = r1(0); _error(3) = r1(1);
    }

    void linearizeOplus() override
    {
        using g2o::VertexSE3Expmap;
        const VertexSE3Expmap* vPose = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexLinePlucker* vLine = static_cast<const VertexLinePlucker*>(_vertices[1]);
        const VertexDepth* vD0 = static_cast<const VertexDepth*>(_vertices[2]);
        const VertexDepth* vD1 = static_cast<const VertexDepth*>(_vertices[3]);

        g2o::SE3Quat Tcw = vPose->estimate();
        Eigen::Matrix<double,6,1> Lw = vLine->estimate();
        double d0 = vD0->estimate();
        double d1 = vD1->estimate();

        Eigen::Vector3d n_w = Lw.head<3>(), v_w = Lw.tail<3>();
        Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
        Eigen::Vector3d tcw = Tcw.translation();

        // intermediates (same as computeError)
        Eigen::Vector3d v_c = Rcw * v_w;
        Eigen::Vector3d n_c = Rcw * n_w + tcw.cross(v_c);
        double s = v_c.dot(v_c);
        Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
        if (s > 1e-12) p0 = v_c.cross(n_c) / s;

        Eigen::Vector3d p0_c = d0 * _dir0;
        Eigen::Vector3d p1_c = d1 * _dir1;

        Eigen::Vector3d e1, e2;
        build_orthonormal_basis(v_c, e1, e2);

        // matrix that projects a 3-vector r_vec into residual components for endpoint i:
        // res_i = [e1^T; e2^T] * (p_ci - p0)
        Eigen::Matrix<double,2,3> P;
        P.row(0) = e1.transpose();
        P.row(1) = e2.transpose();

        // ---------- partial wrt depths (analytic)
        // dr/d(d0) for endpoint0: P * dir0
        Eigen::Matrix<double,2,1> dr0_dd0 = P * _dir0;
        Eigen::Matrix<double,2,1> dr1_dd1 = P * _dir1; // but note endpoint1 uses p1_c - p0, p0 also depends on line

        // We'll compose chains: dr/dpose, dr/dline via d(p0)/d(v_c,n_c) and dr/dv_c,dn_c etc.
        // To reduce algebra errors, compute Jr_vc (3x3) and Jr_nc (3x3) numeric for function r_vec = p_ci - p0 (only p0 depends on v_c,n_c)
        // For both endpoints, rvec_i = p_ci - p0 (p_ci depends on depth only)
        // We compute numeric derivatives of p0 wrt v_c and n_c, then Dr = -d p0/d(...)

        // compute numeric derivatives of p0 wrt v_c and n_c
        Eigen::Matrix<double,3,3> dp0_dvc; dp0_dvc.setZero();
        Eigen::Matrix<double,3,3> dp0_dnc; dp0_dnc.setZero();
        const double eps = 1e-8;
        for (int k = 0; k < 3; ++k) {
            Eigen::Vector3d dv = Eigen::Vector3d::Zero(); dv(k) = eps;
            Eigen::Vector3d vcp = v_c + dv;
            Eigen::Vector3d ncp = n_c;
            double sp = vcp.dot(vcp);
            Eigen::Vector3d p0p = Eigen::Vector3d::Zero();
            if (sp > 1e-12) p0p = vcp.cross(ncp) / sp;
            dp0_dvc.col(k) = (p0p - p0) / eps;

            dv = Eigen::Vector3d::Zero(); dv(k) = eps;
            vcp = v_c; ncp = n_c + dv;
            sp = vcp.dot(vcp);
            p0p = Eigen::Vector3d::Zero();
            if (sp > 1e-12) p0p = vcp.cross(ncp) / sp;
            dp0_dnc.col(k) = (p0p - p0) / eps;
        }

        // For endpoint i: rvec_i = p_ci - p0
        // drvec_i/dv_c = - dp0_dvc  (3x3)
        // drvec_i/dn_c = - dp0_dnc

        Eigen::Matrix<double,3,6> drvec_dvn; drvec_dvn.setZero();
        drvec_dvn.block<3,3>(0,0) = -dp0_dvc;
        drvec_dvn.block<3,3>(0,3) = -dp0_dnc;

        // Now relate [v_c;n_c] to pose (6) and to world line parameters (n_w,v_w)
        // Compose pose jacobians numerically (directional derivatives) for [v_c; n_c] (6x6)
        Eigen::Matrix<double,6,6> J_vn_pose; J_vn_pose.setZero();
        {
            const double eps2 = 1e-7;
            Eigen::Matrix<double,6,1> base; base.block<3,1>(0,0)=v_c; base.block<3,1>(3,0)=n_c;
            for (int i = 0; i < 6; ++i) {
                Eigen::Matrix<double,6,1> xi = Eigen::Matrix<double,6,1>::Zero();
                xi(i) = eps2;
                Eigen::Matrix3d R2 = Rcw;
                Eigen::Vector3d t2 = tcw;
                if (i < 3) {
                    Eigen::Vector3d w = xi.segment<3>(0);
                    Eigen::Matrix3d W; W << 0,-w(2),w(1), w(2),0,-w(0), -w(1),w(0),0;
                    R2 = (Eigen::Matrix3d::Identity() + W) * Rcw; // left-mult approx
                } else {
                    t2(i-3) += eps2;
                }
                Eigen::Vector3d vcp = R2 * v_w;
                Eigen::Vector3d ncp = R2 * n_w + t2.cross(vcp);
                Eigen::Matrix<double,6,1> vp; vp.block<3,1>(0,0)=vcp; vp.block<3,1>(3,0)=ncp;
                J_vn_pose.col(i) = (vp - base) / eps2;
            }
        }

        // Compose line jacobians: [v_c; n_c] wrt line [n_w; v_w] (6x6)
        Eigen::Matrix<double,6,6> J_vn_line; J_vn_line.setZero();
        // ordering: rows (v_c(3); n_c(3)), cols (n_w(3); v_w(3))
        // dv_c/dn_w = 0 ; dv_c/dv_w = Rcw
        J_vn_line.block<3,3>(0,0).setZero();
        J_vn_line.block<3,3>(0,3) = Rcw;
        // dn_c/dn_w = Rcw ; dn_c/dv_w = skew(tcw) * Rcw
        Eigen::Matrix3d S_t; S_t << 0,-tcw(2),tcw(1), tcw(2),0,-tcw(0), -tcw(1),tcw(0),0;
        J_vn_line.block<3,3>(3,0) = Rcw;
        J_vn_line.block<3,3>(3,3) = S_t * Rcw;

        // Now assemble Jacobians for each endpoint
        // dr_i / dpose = P * drvec_i/d[ v_c;n_c ] * J_vn_pose
        Eigen::Matrix<double,2,6> Jr0_pose = P * drvec_dvn * J_vn_pose;
        Eigen::Matrix<double,2,6> Jr1_pose = P * drvec_dvn * J_vn_pose;

        // dr_i / dline = P * drvec * J_vn_line
        Eigen::Matrix<double,2,6> Jr0_line = P * drvec_dvn * J_vn_line;
        Eigen::Matrix<double,2,6> Jr1_line = P * drvec_dvn * J_vn_line;

        // dr0/dd0 = P * dir0 (analytic)
        Eigen::Matrix<double,2,1> Jr0_d0 = P * _dir0;
        // dr1/dd1 = P * dir1 (analytic)
        Eigen::Matrix<double,2,1> Jr1_d1 = P * _dir1;

        // Finally fill _jacobianOplus in the same vertex order
        // vertex 0: pose
        _jacobianOplus[0].setZero();
        _jacobianOplus[0].block<2,6>(0,0) = Jr0_pose;
        _jacobianOplus[0].block<2,6>(2,0) = Jr1_pose;

        // vertex 1: line (6)
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<2,6>(0,0) = Jr0_line;
        _jacobianOplus[1].block<2,6>(2,0) = Jr1_line;

        // vertex 2: depth0 (1)
        _jacobianOplus[2].setZero();
        _jacobianOplus[2].block<2,1>(0,0) = Jr0_d0;

        // vertex 3: depth1 (1)
        _jacobianOplus[3].setZero();
        _jacobianOplus[3].block<2,1>(2,0) = Jr1_d1;
    }

protected:
    Eigen::Vector3d _dir0, _dir1; // K^-1 * [u,v,1] (ray directions in camera frame)
    double fx_=0, fy_=0, cx_=0, cy_=0;
};


//这是迭代的line slam的方法，需要改成这个方法，试试方法
class EdgePointToPluckerLine : public g2o::BaseMultiEdge<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgePointToPluckerLine(const Eigen::Vector2d& px, const Eigen::Matrix3d& K_inv)
        : px_(px), K_inv_(K_inv)
    {
        resize(3); // vertex order: pose, depth, line
    }

    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }

    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0, -v.z(),  v.y(),
               v.z(),     0, -v.x(),
              -v.y(),  v.x(),     0;
        return S;
    }

    // -----------------------------------------------------------------------
    //                         Compute Error
    // -----------------------------------------------------------------------
    void computeError() override
    {
        const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);
        const auto* vLine  = static_cast<const VertexLinePlucker*>(_vertices[2]);

        double d = vDepth->estimate();

        Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0); // camera ray (unscaled)
        Eigen::Vector3d pc  = d * ray; // point in camera frame

        // Use explicit R and t of Twc = Tcw^{-1}: pw = R_wc * pc + t_wc
        Eigen::Matrix3d Rwc = vPose->estimate().inverse().rotation().toRotationMatrix();
        Eigen::Vector3d twc = vPose->estimate().inverse().translation();
        Eigen::Vector3d pw  = Rwc * pc + twc; // world point

        Eigen::Vector3d n = vLine->estimate().head<3>();
        Eigen::Vector3d v = vLine->estimate().tail<3>();
        Eigen::Vector3d vnorm = v.normalized();
        Eigen::Vector3d p0 = vnorm.cross(n); // a point on the line

        // residual = (pw - p0) x vnorm
        _error = (pw - p0).cross(vnorm);
    }

    // -----------------------------------------------------------------------
    //                         Jacobian
    // -----------------------------------------------------------------------
    void linearizeOplus() override
    {
        const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);
        const auto* vLine  = static_cast<const VertexLinePlucker*>(_vertices[2]);

        double d = vDepth->estimate();

        Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0);
        Eigen::Vector3d pc  = d * ray;

        // Twc = Tcw^{-1}
        Eigen::Matrix3d Rwc = vPose->estimate().inverse().rotation().toRotationMatrix();
        Eigen::Vector3d twc = vPose->estimate().inverse().translation();
        Eigen::Vector3d pw  = Rwc * pc + twc;

        Eigen::Vector3d n = vLine->estimate().head<3>();
        Eigen::Vector3d v = vLine->estimate().tail<3>();
        Eigen::Vector3d vnorm = v.normalized();
        Eigen::Vector3d p0 = vnorm.cross(n);

        // ---------------- 1) depth Jacobian ----------------
        // pw = Rwc * (d * ray) + twc  => dpw/dd = Rwc * ray
        Eigen::Vector3d dpw_dd = Rwc * ray;
        // dr/dd = (dpw/dd) x vnorm
        Eigen::Matrix<double,3,1> Jd = skew(dpw_dd) * vnorm; // same as dpw_dd.cross(vnorm)
        _jacobianOplus[1].setZero();
        _jacobianOplus[1].block<3,1>(0,0) = Jd;

        // ---------------- 2) pose Jacobian ----------------
        // We use small-left-multiplicative perturbation xi = [rho; phi] on Tcw.
        // For pw = Twc * pc = Rwc*pc + twc:
        // d(pw)/d(trans) = I
        // d(pw)/d(rot)   = - Rwc * skew(pc)  (standard result)
        Eigen::Matrix<double,3,6> Jpose;
        Jpose.setZero();

        // dr/dt = (d/dt (pw)) x vnorm = I x vnorm = skew(d t) * vnorm => linear map equals -skew(vnorm)
        // As linear operator mapping delta_t to delta_r: delta_r = delta_t x vnorm = skew(delta_t) * vnorm = -skew(vnorm) * delta_t
        Jpose.block<3,3>(0,0) = -skew(vnorm);

        // dr/domega = (d(pw)/domega) x vnorm = ( - Rwc * skew(pc) )_cols -> use matrix multiply
        Eigen::Matrix3d d_pw_domega = - Rwc * skew(pc); // 3x3
        // delta_r = S(delta_pw) * vnorm => for small parameter, J = S(vnorm) * ??? using identity:
        // S(A) vnorm = - S(vnorm) A  (applied column-wise). So J_omega = -skew(vnorm) * d_pw_domega
        Jpose.block<3,3>(0,3) = -skew(vnorm) * d_pw_domega;

        _jacobianOplus[0] = Jpose;

        // ---------------- 3) line Jacobian ----------------
        // In the alternating scheme we typically fix the Plücker line during pose+depth optimize.
        // So set zero. If you want line optimization too, implement analytic jacobian here.
        _jacobianOplus[2].setZero();

        // Note: if vertex 2 is NOT fixed and you implement its jacobian, g2o will use it and then call
        // the VertexLinePlucker::oplusImpl(...) where you already enforce n·v=0.
    }

private:
    Eigen::Vector2d px_;
    Eigen::Matrix3d K_inv_;
};


// EdgePointToPluckerLinePoseAndDepth
//  — 优化相机 pose (VertexSE3Expmap) 和 单个端点深度 (VertexDepth)
//  — Plücker 直线作为固定测量 (Eigen::Matrix<float,6,1>)
//  — 使用 BaseMultiEdge，vertices order: [pose, depth]
//  — 残差为 3D 向量 (point-to-line cross-product)
// 注意：适配你的 g2o 版本时，确保 VertexDepth 类型已定义且继承自 g2o::BaseVertex 或类似。
class EdgePointToPluckerLinePoseAndDepth : public g2o::BaseMultiEdge<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    /**
     * @param px   image endpoint (u,v) in pixels
     * @param K_inv 3x3 inverse camera intrinsics (double)
     * @param plucker_line 6x1 float: [n(3); v(3)]  (固定测量)
     */
    EdgePointToPluckerLinePoseAndDepth(
        const Eigen::Vector2d& px,
        const Eigen::Matrix3d& K_inv,
        const Eigen::Matrix<double,6,1>& plucker_line)
        : px_(px), K_inv_(K_inv), plucker_line_(plucker_line)
    {
        // 2 vertices: 0: pose (VertexSE3Expmap), 1: depth (VertexDepth)
        resize(2);
        // 初始化雅可比矩阵大小并清零
        _jacobianOplus[0].resize(3,6);
        _jacobianOplus[0].setZero();
        _jacobianOplus[1].resize(3,1);
        _jacobianOplus[1].setZero();
    }
    bool read(std::istream& /*is*/) override { return false; }
    bool write(std::ostream& /*os*/) const override { return false; }
    // skew / cross-product matrix
    inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
    {
        Eigen::Matrix3d S;
        S <<     0.0, -v.z(),  v.y(),
               v.z(),   0.0, -v.x(),
              -v.y(),  v.x(),   0.0;
        return S;
    }
    // -------------------- computeError --------------------
    void computeError() override
    {
        // vertex cast
        const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);
        // safe read depth
        double d = vDepth->estimate();
        //std::cerr <<"d: " << d << std::endl;
        //std::cerr << "plucker_line_: " << plucker_line_.transpose() << std::endl;
        if(!(d > 0.0 && std::isfinite(d))) d = 1e-3; // clamp to small positive
        // backproject ray (camera frame)
        Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0); // camera ray (double)
        Eigen::Vector3d pc  = d * ray; // point in camera frame
        // Twc = inverse(Tcw)
        g2o::SE3Quat Tcw_est = vPose->estimate();
        g2o::SE3Quat Twc = Tcw_est.inverse();
        Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Vector3d pw = Rwc * pc + twc; // world point
        // plucker (fixed) -> cast float->double for stable computation
        Eigen::Vector3d n = plucker_line_.head<3>();
        Eigen::Vector3d v = plucker_line_.tail<3>();
        // safety for direction vector v
        double vnorm_len = v.norm();
        if(!std::isfinite(vnorm_len) || vnorm_len < 1e-9) {
            // fallback to a safe direction (arbitrary but finite)
            v = Eigen::Vector3d(1.0, 0.0, 0.0);
            vnorm_len = 1.0;
        }
        Eigen::Vector3d vnorm = v / vnorm_len;
        Eigen::Vector3d p0 = vnorm.cross(n); // a point on the line (world)
        // residual: 3D vector — point-to-line cross product
        _error = (pw - p0).cross(vnorm);
        Eigen::Vector3d test_error = (pw - p0).cross(vnorm);
        //std::cerr <<"test_error:  " << test_error.transpose() << std::endl;
        // safety: if any NaN/Inf arises, zero the error (prevents g2o crash)
        if(!(std::isfinite(_error[0]) && std::isfinite(_error[1]) && std::isfinite(_error[2])))
        {
            _error.setZero();
        }
    }
    // -------------------- linearizeOplus --------------------
    // fill _jacobianOplus[0] (pose: 3x6) and _jacobianOplus[1] (depth: 3x1)
    void linearizeOplus() override
    {
        // defensive
        if(_vertices[0] == nullptr || _vertices[1] == nullptr) {
            _jacobianOplus[0].setZero();
            _jacobianOplus[1].setZero();
            return;
        }
        const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);
        // safe depth
        double d = vDepth->estimate();
        if(!(d > 0.0 && std::isfinite(d))) d = 1e-3;
        // ray and pc
        Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0);
        Eigen::Vector3d pc  = d * ray;
        // Twc, Rwc
        g2o::SE3Quat Tcw_est = vPose->estimate();
        g2o::SE3Quat Twc = Tcw_est.inverse();
        Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();
        // plucker line (fixed)
        Eigen::Vector3d n = plucker_line_.head<3>();
        Eigen::Vector3d v = plucker_line_.tail<3>();
        //std::cerr <<"n: " << n.transpose() << std::endl;
        //std::cerr <<"v: " << v.transpose() << std::endl;
        double vnorm_len = v.norm();
        if(!std::isfinite(vnorm_len) || vnorm_len < 1e-9) {
            v = Eigen::Vector3d(1.0, 0.0, 0.0);
            vnorm_len = 1.0;
        }
        Eigen::Vector3d vnorm = v / vnorm_len;
        // ----- depth jacobian (3x1) -----
        // dpw/dd = Rwc * ray
        Eigen::Vector3d dpw_dd = Rwc * ray;
        //std::cerr << "dpw_dd: " << dpw_dd.transpose() << std::endl;
        // dr/dd = (dpw/dd) x vnorm = skew(dpw_dd) * vnorm
        _jacobianOplus[1].setZero();
        Eigen::Matrix<double,3,1> Jd = skew(dpw_dd) * vnorm;
        //std::cerr << "JD: " << Jd.transpose() << std::endl;
        //_jacobianOplus[1].block<3,1>(0,0) = Jd;
        _jacobianOplus[1].col(0) = Jd;
        // _jacobianOplus[1](0,0) = Jd(0,0);
        // std::cerr << "_jacobianOplus[1](0,0): " << _jacobianOplus[1](0,0) << std::endl;
        // _jacobianOplus[1](1,0) = Jd(1,0);
        // _jacobianOplus[1](2,0) = Jd(2,0);
        //std::cerr << "JD: END" << std::endl;
        // ----- pose jacobian (3x6) -----
        // d(pw)/d(delta_t) = I
        // d(pw)/d(delta_phi) = - Rwc * skew(pc)
        Eigen::Matrix<double,3,6> Jpose; Jpose.setZero();
        //std::cerr << "111111" << std::endl;
        // translation part: dr/dt = [vnorm]_x  (we write as -skew(vnorm) for consistency)
        Jpose.block<3,3>(0,0) = -skew(vnorm);
        //std::cerr << "pc: " << pc.transpose() << std::endl;
        // rotation part:
        Eigen::Matrix3d d_pw_domega = - Rwc * skew(pc); // 3x3
        Jpose.block<3,3>(0,3) = -skew(vnorm) * d_pw_domega;
        //std::cerr << "d_pw_domega: " << d_pw_domega.transpose() << std::endl;
        //std::cerr << "Jpose: " << Jpose << std::endl;
        // assign Jpose → jacobianOplus[0] safely
        _jacobianOplus[0] = Jpose;
        //std::cerr << "Jpose End: " << Jpose << std::endl;
        // // Safety: if some jacobian entries are NaN/Inf, clamp to zero to avoid optimizer crash
        // for(int i=0;i<3;i++){
        //     for(int j=0;j<6;j++){
        //         if(!std::isfinite(_jacobianOplus[0](i,j))) _jacobianOplus[0](i,j) = 0.0;
        //     }
        //     if(!std::isfinite(_jacobianOplus[1](i,0))) _jacobianOplus[1](i,0) = 0.0;
        // }
        //std::cerr << "........all end......." << std::endl;
    }
private:
    Eigen::Vector2d px_;                       // image pixel endpoint (u,v)
    Eigen::Matrix3d K_inv_;                    // inverse intrinsics (double)
    Eigen::Matrix<double,6,1> plucker_line_;    // fixed plucker measurement (double)
};


class EdgeSimpleSE3Depth
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 g2o::VertexSE3Expmap,
                                 VertexDepth>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void computeError() override
    {
        const auto* v0 =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* v1 =
            static_cast<const VertexDepth*>(_vertices[1]);

        const Eigen::Vector3d t = v0->estimate().translation();
        const double d = v1->estimate();

        // 非常简单、稳定的 error
        _error = t + Eigen::Vector3d(d, d, d) - _measurement;
    }

    void linearizeOplus() override
    {
        // --- w.r.t. pose (6D) ---
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<3,3>(0,3).setIdentity(); // dt / dtranslation

        // --- w.r.t. depth (1D) ---
        _jacobianOplusXj.setZero();
        _jacobianOplusXj.col(0).setOnes();
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};


// void computeError() override
    // {
    //     auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(vertex(0));
    //     auto* vDepth = static_cast<const VertexDepth*>(vertex(1));
    //     // 确保顶点存在
    //     if(!vPose || !vDepth){ _error.setZero(); return; }
    //     double d = vDepth->estimate();
    //     // 深度检查
    //     if(!(d>0 && std::isfinite(d))) d = 1e-3;
    //     // 1. 投影射线 (归一化平面)
    //     Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_[0], px_[1], 1.0);
    //     // 2. 相机坐标系下的点
    //     Eigen::Vector3d pc = d * ray;
    //     // 3. 世界坐标系下的点
    //     g2o::SE3Quat Tcw = vPose->estimate();
    //     g2o::SE3Quat Twc = Tcw.inverse();
    //     Eigen::Vector3d pw = Twc.map(pc);
    //     // 4. Plucker 线参数
    //     Eigen::Vector3d n = Lw_.head<3>(); // 矩
    //     Eigen::Vector3d v = Lw_.tail<3>(); // 方向
    //     double vn = v.norm();
    //     if(vn<1e-9 || !std::isfinite(vn)){ v=Eigen::Vector3d(1,0,0); vn=1.0; } // 检查并默认
    //     Eigen::Vector3d vnorm = v/vn; // 方向单位向量
    //     // 5. 直线上的一个点 p0 = (v x n) / ||v||^2
    //     Eigen::Vector3d p0 = v.cross(n)/(vn*vn);
    //     // 6. 误差：点到直线的垂直距离 r = (pw - p0) x vnorm
    //     Eigen::Vector3d r = (pw - p0).cross(vnorm);
    //     if(!r.allFinite()) _error.setZero();
    //     else _error = r;
    // }

// =======================================================
// Edge: Point to Plucker Line (3D Error)
// Debug-safe version for Jacobian checking
// =======================================================
// class EdgePointToPluckerLinePoseAndDepthNew
//     : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
//                                  g2o::VertexSE3Expmap, VertexDepth>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     virtual ~EdgePointToPluckerLinePoseAndDepthNew() = default;

//     EdgePointToPluckerLinePoseAndDepthNew(
//         const Eigen::Vector2d& px,
//         const Eigen::Matrix3d& K_inv,
//         const Eigen::Matrix<double,6,1>& plucker)
//         : px_(px), K_inv_(K_inv), Lw_(plucker)
//     {
//         dbg_J_pose.setZero();
//         dbg_J_depth.setZero();
//         dbg_jac_updated = false;
//     }
//     bool read(std::istream&) override { return false; }
//     bool write(std::ostream&) const override { return false; }
//     Eigen::Matrix<double,3,6> dbg_J_pose;   //debug jacobian
//     Eigen::Matrix<double,3,1> dbg_J_depth;
//     bool dbg_jac_updated = false;
// private:
//     // ---- debug macro ----
// #ifndef JAC_CHECK_DEBUG
// #define JAC_CHECK_DEBUG 1
// #endif
// #if JAC_CHECK_DEBUG
// #define STEP(tag) \
//     do { std::cout << "[EDGE] " << (tag) << " @L" << __LINE__ << std::endl; } while(0)
// #else
// #define STEP(tag) do {} while(0)
// #endif
//     static inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
//     {
//         Eigen::Matrix3d S;
//         S << 0.0,   -v.z(),  v.y(),
//              v.z(),  0.0,   -v.x(),
//             -v.y(),  v.x(),  0.0;
//         return S;
//     }
// public:
//     void computeError() override
//     {
//         STEP("computeError.begin");
//         const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(vertex(0));
//         const auto* vDepth = static_cast<const VertexDepth*>(vertex(1));
//         if(!vPose || !vDepth){
//             _error.setZero();
//             STEP("computeError.nullVertex");
//             return;
//         }
//         double d = vDepth->estimate();
//         if(!(d > 0.0) || !std::isfinite(d)){
//             // 不要 assert 直接炸；调试时更希望继续跑完看到更多信息
//             d = 1e-3;
//         }
//         STEP("computeError.ray");
//         const Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_.x(), px_.y(), 1.0);
//         if(!ray.allFinite()){
//             _error.setZero();
//             STEP("computeError.ray.nan");
//             return;
//         }
//         STEP("computeError.pc");
//         const Eigen::Vector3d pc = d * ray;
//         STEP("computeError.Twc");
//         const g2o::SE3Quat Twc = vPose->estimate().inverse();
//         STEP("computeError.pw");
//         const Eigen::Vector3d pw = Twc.map(pc);
//         if(!pw.allFinite()){
//             _error.setZero();
//             STEP("computeError.pw.nan");
//             return;
//         }
//         STEP("computeError.line");
//         Eigen::Vector3d n = Lw_.head<3>();
//         Eigen::Vector3d v = Lw_.tail<3>();
//         if(!n.allFinite() || !v.allFinite()){
//             _error.setZero();
//             STEP("computeError.line.nan");
//             return;
//         }
//         double vn = v.norm();
//         if(!(vn > 1e-9) || !std::isfinite(vn)){
//             // 退化线：给一个默认方向，保证不炸
//             v  = Eigen::Vector3d(1,0,0);
//             vn = 1.0;
//         }
//         const Eigen::Vector3d vnorm = v / vn;
//         STEP("computeError.p0");
//         const Eigen::Vector3d p0 = v.cross(n) / (vn * vn);
//         STEP("computeError.r");
//         const Eigen::Vector3d r = (pw - p0).cross(vnorm);
//         if(!r.allFinite()){
//             _error.setZero();
//             STEP("computeError.r.nan");
//             return;
//         }
//         _error = r;
//         STEP("computeError.end");
//     }
//     void linearizeOplus() override
//     {
//         STEP("linearize.begin");
//         const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(vertex(0));
//         const auto* vDepth = static_cast<const VertexDepth*>(vertex(1));
//         if(!vPose || !vDepth){
//             _jacobianOplusXi.setZero();
//             _jacobianOplusXj.setZero();
//             STEP("linearize.nullVertex");
//             return;
//         }
//         // ---------- 强制检查 Jacobian 尺寸（非常关键） ----------
//         // g2o 正常情况下：Xi=3x6, Xj=3x1
//         if(_jacobianOplusXi.rows()!=3 || _jacobianOplusXi.cols()!=6 ||
//            _jacobianOplusXj.rows()!=3 || _jacobianOplusXj.cols()!=1)
//         {
// #if JAC_CHECK_DEBUG
//             std::cout << "[EDGE] JXi size=" << _jacobianOplusXi.rows()
//                       << "x" << _jacobianOplusXi.cols()
//                       << "  JXj size=" << _jacobianOplusXj.rows()
//                       << "x" << _jacobianOplusXj.cols() << std::endl;
// #endif
//             _jacobianOplusXi.setZero();
//             _jacobianOplusXj.setZero();
//             STEP("linearize.badJacSize");
//             return;
//         }
//         STEP("linearize.inputs");
//         double d = vDepth->estimate();
//         if(!(d > 0.0) || !std::isfinite(d)) d = 1e-3;
//         const Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_.x(), px_.y(), 1.0);
//         if(!ray.allFinite()){
//             _jacobianOplusXi.setZero();
//             _jacobianOplusXj.setZero();
//             STEP("linearize.ray.nan");
//             return;
//         }
//         const Eigen::Vector3d pc = d * ray;
//         STEP("linearize.Rwc");
//         const g2o::SE3Quat Twc = vPose->estimate().inverse();
//         const Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();
//         if(!Rwc.allFinite()){
//             _jacobianOplusXi.setZero();
//             _jacobianOplusXj.setZero();
//             STEP("linearize.Rwc.nan");
//             return;
//         }
//         STEP("linearize.vnorm");
//         Eigen::Vector3d v = Lw_.tail<3>();
//         double vn = v.norm();
//         if(!(vn > 1e-9) || !std::isfinite(vn)){
//             v  = Eigen::Vector3d(1,0,0);
//             vn = 1.0;
//         }
//         const Eigen::Vector3d vnorm = v / vn;
//         STEP("linearize.R");
//         const Eigen::Matrix3d R = skew(vnorm); // e = [v]x (pw - p0)
//         // ---------- Pose Jacobian ----------
//         STEP("linearize.dpw_dxi");
//         Eigen::Matrix<double,3,6> dpw_dxi;
//         dpw_dxi.block<3,3>(0,0) = -Rwc * skew(pc); // d pw / d omega
//         dpw_dxi.block<3,3>(0,3) = -Rwc;            // d pw / d rho
//         STEP("linearize.JXi");
//         _jacobianOplusXi.noalias() = R * dpw_dxi;
//         // ---------- Depth Jacobian ----------
//         STEP("linearize.dpw_dd");
//         const Eigen::Vector3d dpw_dd = Rwc * ray;
//         STEP("linearize.JXj");
//         _jacobianOplusXj.setZero();
//         _jacobianOplusXj.col(0).noalias() = R * dpw_dd; // = vnorm x dpw_dd
//         STEP("linearize.finiteCheck");
//         if(!_jacobianOplusXi.allFinite() || !_jacobianOplusXj.allFinite()){
//             _jacobianOplusXi.setZero();
//             _jacobianOplusXj.setZero();
//             STEP("linearize.nan");
//             return;
//         }
//         // ✅ 放在 finiteCheck 后
//         dbg_J_pose  = _jacobianOplusXi; //debug jacobian
//         dbg_J_depth = _jacobianOplusXj;
//         dbg_jac_updated = true;
//         STEP("linearize.end");
//     }
// private:
//     Eigen::Vector2d px_;
//     Eigen::Matrix3d K_inv_;
//     Eigen::Matrix<double,6,1> Lw_;
// };



class EdgePointToPluckerLinePoseAndDepthNew
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,
                                 g2o::VertexSE3Expmap, VertexDepth>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual ~EdgePointToPluckerLinePoseAndDepthNew() = default;

    EdgePointToPluckerLinePoseAndDepthNew(
        const Eigen::Vector2d& px,
        const Eigen::Matrix3d& K_inv,
        const Eigen::Matrix<double,6,1>& plucker)
        : px_(px), K_inv_(K_inv), Lw_(plucker)
    {
        dbg_J_pose.setZero();
        dbg_J_depth.setZero();
        dbg_jac_updated = false;
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // -------- debug output --------
    Eigen::Matrix<double,3,6> dbg_J_pose;
    Eigen::Matrix<double,3,1> dbg_J_depth;
    bool dbg_jac_updated = false;

private:
#ifndef JAC_CHECK_DEBUG
#define JAC_CHECK_DEBUG 0   
#endif

#if JAC_CHECK_DEBUG
#define STEP(tag) \
    do { std::cout << "[EDGE] " << (tag) << " @L" << __LINE__ << std::endl; } while(0)
#else
#define STEP(tag) do {} while(0)
#endif

    static inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d S;
        S <<  0.0,   -v.z(),  v.y(),
              v.z(),  0.0,   -v.x(),
             -v.y(),  v.x(),  0.0;
        return S;
    }

public:
    // =====================================================
    // computeError
    // =====================================================
    void computeError() override
    {
        STEP("computeError.begin");

        const auto* vPose  =
            static_cast<const g2o::VertexSE3Expmap*>(vertex(0));
        const auto* vDepth =
            static_cast<const VertexDepth*>(vertex(1));

        if(!vPose || !vDepth){
            _error.setZero();
            STEP("computeError.nullVertex");
            return;
        }

        double d = vDepth->estimate();
        if(!(d > 0.0) || !std::isfinite(d))
            d = 1e-3;

        const Eigen::Vector3d ray =
            K_inv_ * Eigen::Vector3d(px_.x(), px_.y(), 1.0);

        if(!ray.allFinite()){
            _error.setZero();
            STEP("computeError.ray.nan");
            return;
        }

        const Eigen::Vector3d pc = d * ray;

        const g2o::SE3Quat Twc = vPose->estimate().inverse();
        const Eigen::Vector3d pw = Twc.map(pc);

        if(!pw.allFinite()){
            _error.setZero();
            STEP("computeError.pw.nan");
            return;
        }

        Eigen::Vector3d n = Lw_.head<3>();
        Eigen::Vector3d v = Lw_.tail<3>();

        if(!n.allFinite() || !v.allFinite()){
            _error.setZero();
            STEP("computeError.line.nan");
            return;
        }

        double vn = v.norm();
        if(!(vn > 1e-9))
            v = Eigen::Vector3d(1,0,0), vn = 1.0;

        const Eigen::Vector3d vnorm = v / vn;
        const Eigen::Vector3d p0 = v.cross(n) / (vn * vn);

        const Eigen::Vector3d r = (pw - p0).cross(vnorm);
        if(!r.allFinite()){
            _error.setZero();
            STEP("computeError.r.nan");
            return;
        }

        _error = r;
        STEP("computeError.end");
    }

    // =====================================================
    // linearizeOplus  (✔ numerically verified)
    // =====================================================

    void linearizeOplus() override
    {
        STEP("linearize.begin");

        const auto* vPoseRaw  = vertex(0);
        const auto* vDepthRaw = vertex(1);

        if (!vPoseRaw || !vDepthRaw) {
            std::cerr << "[ERR] vertex null (vPoseRaw or vDepthRaw)" << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        auto* vPose  = dynamic_cast<const g2o::VertexSE3Expmap*>(vPoseRaw);
        auto* vDepth = dynamic_cast<const VertexDepth*>(vDepthRaw);

        if (!vPose || !vDepth) {
            std::cerr << "[ERR] vertex cast failed." << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        //std::cout << "[INFO] linearize vertex ids: pose=" << vPose->id()
        //        << ", depth=" << vDepth->id() << std::endl;

        double d = vDepth->estimate();
        if (!(d > 0.0) || !std::isfinite(d)) {
            std::cerr << "[WARN] invalid depth: " << d << std::endl;
            d = 1e-3;
        }

        const Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_.x(), px_.y(), 1.0);
        if (!ray.allFinite()) {
            std::cerr << "[ERR] ray is not finite: " << ray.transpose() << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }
        //std::cout << "[DBG] ray = " << ray.transpose() << std::endl;

        const Eigen::Vector3d pc = d * ray;
        if (!pc.allFinite()) {
            std::cerr << "[ERR] pc is not finite: " << pc.transpose() << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        const g2o::SE3Quat Twc = vPose->estimate().inverse();
        //std::cout << "[DBG] Twc translation = " << Twc.translation().transpose() << std::endl;
        const Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();
        if (!Rwc.allFinite()) {
            std::cerr << "[ERR] Rwc not finite!" << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        Eigen::Vector3d v = Lw_.tail<3>();
        double vn = v.norm();
        if (!(vn > 1e-9)) {
            std::cerr << "[WARN] v.norm() too small. Resetting." << std::endl;
            v = Eigen::Vector3d(1, 0, 0);
            vn = 1.0;
        }

        const Eigen::Vector3d vnorm = v / vn;
        if (!vnorm.allFinite()) {
            std::cerr << "[ERR] vnorm is not finite: " << vnorm.transpose() << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        const Eigen::Matrix3d R = -skew(vnorm);
        //std::cout << "[DBG] R = \n" << R << std::endl;
        if (!R.allFinite()) {
            std::cerr << "[ERR] R = skew(vnorm) is not finite!" << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        // --- Pose Jacobian ---
        Eigen::Matrix<double, 3, 6> dpw_dxi;
        dpw_dxi.block<3,3>(0,0) = Rwc * skew(pc);
        dpw_dxi.block<3,3>(0,3) = -Rwc;

        if (!dpw_dxi.allFinite()) {
            std::cerr << "[ERR] dpw_dxi not finite!" << std::endl;
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        _jacobianOplusXi.noalias() = R * dpw_dxi;

        // --- Depth Jacobian ---
        const Eigen::Vector3d dpw_dd = Rwc * ray;
        if (!dpw_dd.allFinite()) {
            std::cerr << "[ERR] dpw_dd not finite: " << dpw_dd.transpose() << std::endl;
            _jacobianOplusXj.setZero();
            return;
        }
        //std::cout << "[DBG] dpw_dd = " << dpw_dd.transpose() << std::endl;
        _jacobianOplusXj.col(0).noalias() = R * dpw_dd;
        //std::cout << "[DBG] J_depth = " << _jacobianOplusXj.col(0).transpose() << std::endl;

        // Save for debug
        dbg_J_pose  = _jacobianOplusXi;
        dbg_J_depth = _jacobianOplusXj;
        dbg_jac_updated = true;

        STEP("linearize.end");
    }


    // void linearizeOplus() override
    // {
    //     STEP("linearize.begin");

    //     const auto* vPose  =
    //         static_cast<const g2o::VertexSE3Expmap*>(vertex(0));
    //     const auto* vDepth =
    //         static_cast<const VertexDepth*>(vertex(1));

    //     if(!vPose || !vDepth){
    //         _jacobianOplusXi.setZero();
    //         _jacobianOplusXj.setZero();
    //         return;
    //     }

    //     double d = vDepth->estimate();
    //     if(!(d > 0.0) || !std::isfinite(d))
    //         d = 1e-3;

    //     const Eigen::Vector3d ray =
    //         K_inv_ * Eigen::Vector3d(px_.x(), px_.y(), 1.0);
    //     const Eigen::Vector3d pc = d * ray;

    //     const g2o::SE3Quat Twc = vPose->estimate().inverse();
    //     const Eigen::Matrix3d Rwc =
    //         Twc.rotation().toRotationMatrix();

    //     Eigen::Vector3d v = Lw_.tail<3>();
    //     double vn = v.norm();
    //     if(!(vn > 1e-9))
    //         v = Eigen::Vector3d(1,0,0), vn = 1.0;

    //     const Eigen::Vector3d vnorm = v / vn;

    //     // ⭐ 核心修正：负号
    //     const Eigen::Matrix3d R = -skew(vnorm);

    //     // ---- pose Jacobian ----
    //     Eigen::Matrix<double,3,6> dpw_dxi;
    //     //dpw_dxi.block<3,3>(0,0) = -Rwc * skew(pc);
    //     dpw_dxi.block<3,3>(0,0) = Rwc * skew(pc);
    //     dpw_dxi.block<3,3>(0,3) = -Rwc;

    //     _jacobianOplusXi.noalias() = R * dpw_dxi;

    //     // ---- depth Jacobian ----
    //     const Eigen::Vector3d dpw_dd = Rwc * ray;
    //     _jacobianOplusXj.col(0).noalias() = R * dpw_dd;

    //     dbg_J_pose  = _jacobianOplusXi;
    //     dbg_J_depth = _jacobianOplusXj;
    //     dbg_jac_updated = true;

    //     STEP("linearize.end");
    // }
    
public:
    inline const Eigen::Matrix<double,3,1>& JacobianDepth() const
    {
        return _jacobianOplusXj;
    }

    inline const Eigen::Vector3d& ErrorVec() const
    {
        return _error;
    }

    inline const Eigen::Matrix3d& InfoMat() const
    {
        return information(); // g2o BaseEdge 提供
    }
    // inside EdgePointToPluckerLinePoseAndDepthNew
    inline const Eigen::Vector2d& GetPixel() const { return px_; }
    inline const Eigen::Matrix3d& GetKinv() const { return K_inv_; }
    inline const Eigen::Matrix<double,6,1>& GetPlucker() const { return Lw_; }

private:
    Eigen::Vector2d px_;
    Eigen::Matrix3d K_inv_;
    Eigen::Matrix<double,6,1> Lw_;
};


// =======================================================
// 3. MANUAL JACOBIAN CHECKER
// =======================================================
// 检查 Vertex 的 Jacobian (通用函数)
bool checkJacobian_zdg(g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap, VertexDepth>* edge, int vertex_idx);


// class EdgePointToPluckerLinePoseAndDepth : public g2o::BaseMultiEdge<3, Eigen::Vector3d>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

//     /**
//      * @param px   image endpoint (u,v) in pixels
//      * @param K_inv 3x3 inverse camera intrinsics (double)
//      * @param plucker_line 6x1 double: [n(3); v(3)]  (固定测量)
//      */
//     EdgePointToPluckerLinePoseAndDepth(
//         const Eigen::Vector2d& px,
//         const Eigen::Matrix3d& K_inv,
//         const Eigen::Matrix<double,6,1>& plucker_line)
//         : px_(px), K_inv_(K_inv), plucker_line_(plucker_line)
//     {
//         // 2 vertices: 0 = pose, 1 = depth
//         resize(2);

//         // 初始化雅可比矩阵大小并清零
//         _jacobianOplus[0].resize(3,6);
//         _jacobianOplus[0].setZero();
//         _jacobianOplus[1].resize(3,1);
//         _jacobianOplus[1].setZero();
//     }

//     bool read(std::istream& /*is*/) override { return false; }
//     bool write(std::ostream& /*os*/) const override { return false; }

//     // skew / cross-product matrix
//     inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) const
//     {
//         Eigen::Matrix3d S;
//         S <<     0.0, -v.z(),  v.y(),
//                v.z(),   0.0, -v.x(),
//               -v.y(),  v.x(),   0.0;
//         return S;
//     }

//     void computeError() override
//     {
//         const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);
//         if(!vPose || !vDepth) { _error.setZero(); return; }

//         double d = vDepth->estimate();
//         if(!(d > 0.0 && std::isfinite(d))) d = 1e-3;

//         Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0);
//         Eigen::Vector3d pc  = d * ray;

//         g2o::SE3Quat Tcw = vPose->estimate();
//         g2o::SE3Quat Twc = Tcw.inverse();
//         Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();
//         Eigen::Vector3d t = Twc.translation();

//         Eigen::Vector3d pw = Rwc * pc + t;

//         Eigen::Vector3d n = plucker_line_.head<3>();
//         Eigen::Vector3d v = plucker_line_.tail<3>();
//         double vnorm_len = v.norm();
//         if(!std::isfinite(vnorm_len) || vnorm_len < 1e-9) v = Eigen::Vector3d(1.0,0,0), vnorm_len=1.0;
//         Eigen::Vector3d vnorm = v / vnorm_len;
//         Eigen::Vector3d p0 = vnorm.cross(n);

//         _error = (pw - p0).cross(vnorm);

//         if(!(_error.allFinite())) _error.setZero();
//     }

//     void linearizeOplus() override
//     {
//         if(!_vertices[0] || !_vertices[1]) {
//             _jacobianOplus[0].setZero();
//             _jacobianOplus[1].setZero();
//             return;
//         }

//         const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//         const auto* vDepth = static_cast<const VertexDepth*>(_vertices[1]);

//         double d = vDepth->estimate();
//         if(!(d>0.0 && std::isfinite(d))) d = 1e-3;

//         Eigen::Vector3d ray = K_inv_ * Eigen::Vector3d(px_(0), px_(1), 1.0);
//         Eigen::Vector3d pc  = d * ray;

//         g2o::SE3Quat Twc = vPose->estimate().inverse();
//         Eigen::Matrix3d Rwc = Twc.rotation().toRotationMatrix();

//         Eigen::Vector3d n = plucker_line_.head<3>();
//         Eigen::Vector3d v = plucker_line_.tail<3>();
//         double vnorm_len = v.norm();
//         if(!std::isfinite(vnorm_len) || vnorm_len < 1e-9) v = Eigen::Vector3d(1.0,0,0), vnorm_len=1.0;
//         Eigen::Vector3d vnorm = v / vnorm_len;

//         // ----- depth jacobian (3x1) -----
//         Eigen::Vector3d dpw_dd = Rwc * ray;
//         _jacobianOplus[1].setZero();
//         std::cerr < "dpw_dd: " << dpw_dd.transpose() << std::endl;
//         _jacobianOplus[1].col(0) = skew(dpw_dd) * vnorm;
//         std::cerr <<"end..." << std::endl;

//         // ----- pose jacobian (3x6) -----
//         Eigen::Matrix<double,3,6> Jpose;
//         Jpose.setZero();
//         Jpose.block<3,3>(0,0) = -skew(vnorm);
//         Eigen::Matrix3d d_pw_domega = -Rwc * skew(pc);
//         Jpose.block<3,3>(0,3) = -skew(vnorm) * d_pw_domega;
//         _jacobianOplus[0] = Jpose;

//         // 防止 NaN/Inf
//         for(int i=0;i<3;i++){
//             for(int j=0;j<6;j++) if(!std::isfinite(_jacobianOplus[0](i,j))) _jacobianOplus[0](i,j)=0.0;
//             if(!std::isfinite(_jacobianOplus[1](i,0))) _jacobianOplus[1](i,0)=0.0;
//         }
//     }

// private:
//     Eigen::Vector2d px_;
//     Eigen::Matrix3d K_inv_;
//     Eigen::Matrix<double,6,1> plucker_line_;
// };


}

#endif //ORB_SLAM3_OPTIMIZABLETYPES_H
