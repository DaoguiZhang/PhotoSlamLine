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

#include <Eigen/Geometry>
#include <include/CameraModels/GeometricCamera.h>


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


class EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine
    : public g2o::BaseUnaryEdge<4, Eigen::Vector4d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine(const Eigen::Vector3d& p1w,
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

class EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine
    : public g2o::BaseUnaryEdge<2, Eigen::Matrix<double,2,1>, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    EdgeSE3ProjectLineXYZOnlyPoseToBody_PointToLine()
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

}

#endif //ORB_SLAM3_OPTIMIZABLETYPES_H
