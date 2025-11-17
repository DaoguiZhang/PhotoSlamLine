#ifndef VERTEXTOLINEPLUCER_H
#define VERTEXTOLINEPLUCER_H
// vertex_line_plucker.h
// A g2o Vertex representing a 3D Plücker line as a 6-vector [n; v],
// where n = moment = p x v, v = direction.
// We enforce/encourage: v normalized (||v|| = 1) and n · v = 0
// (after every update we project to satisfy these).

//#include <Thirdparty/g2o/core/base_vertex.h>
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/types/sim3.h>
#include <Eigen/Core>
#include <cmath>

namespace ORB_SLAM3
{

class VertexLinePluckerOld : public g2o::BaseVertex<6, Eigen::Matrix<double,6,1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    VertexLinePluckerOld()
    {
        // default estimate: zero vector (invalid line) -- caller should set proper estimate
        _estimate.setZero();
    }
    // required by g2o: set the vertex estimate to identity / origin
    virtual void setToOriginImpl() override {
        _estimate.setZero();
    }
    // additive update: simply add the increment to the 6-vector and re-normalize/project
    // In many usages you may prefer to param in minimal form; here we use simple additive update.
    virtual void oplusImpl(const double* update_) override {
        Eigen::Map<const Eigen::Matrix<double,6,1>> upd(update_);
        _estimate += upd;
        enforceConstraints();
    }
    // read / write (very simple)
    virtual bool read(std::istream& is) override {
        for (int i=0;i<6;i++) is >> _estimate[i];
        enforceConstraints();
        return is.good();
    }
    virtual bool write(std::ostream& os) const override {
        for (int i=0;i<6;i++) {
            os << _estimate[i] << " ";
        }
        return os.good();
    }
    // --- helpers for user code ---
    // set from explicit n and v (will be projected to constraints)
    void setFromNV(const Eigen::Matrix<double,3,1>& n_in,
                   const Eigen::Matrix<double,3,1>& v_in)
    {
        _estimate.head<3>() = n_in;
        _estimate.tail<3>() = v_in;
        enforceConstraints();
    }
    // build plucker from two distinct points p1,p2: v = p2 - p1; n = p1 x v
    void setFromTwoPoints(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
        Eigen::Vector3d v = p2 - p1;
        Eigen::Vector3d n = p1.cross(v);
        _estimate.head<3>() = n;
        _estimate.tail<3>() = v;
        enforceConstraints();
    }
    // return direction (unit) v
    Eigen::Vector3d direction() const {
        return _estimate.tail<3>();
    }
    // return moment n
    Eigen::Vector3d moment() const {
        return _estimate.head<3>();
    }
    // return a point on the line (the point closest to origin)
    // formula: p0 = (v x n) / ||v||^2  (works even if v not unit, but we maintain ||v||=1)
    Eigen::Vector3d pointOnLineClosestToOrigin() const {
        Eigen::Vector3d n = moment();
        Eigen::Vector3d v = direction();
        double v2 = v.squaredNorm();
        if (v2 < 1e-12) return Eigen::Vector3d::Zero();
        return v.cross(n) / v2;
    }
    // utility: return two distinct points on the line (p0, p0 + v)
    void getTwoPoints(Eigen::Vector3d &p0, Eigen::Vector3d &p1) const {
        p0 = pointOnLineClosestToOrigin();
        p1 = p0 + direction();
    }
protected:
    // Enforce normalization & orthogonality constraints:
    //  - make v non-zero; normalize v to unit length
    //  - make n orthogonal to v by replacing n <- n - (n·v) v
    // This keeps parameters stable and removes scale ambiguity.
    void enforceConstraints() {
        Eigen::Vector3d n = _estimate.head<3>();
        Eigen::Vector3d v = _estimate.tail<3>();

        double vnorm = v.norm();
        if (vnorm < 1e-12) {
            // bad direction: leave v as small vector but avoid divide by zero
            // set to default direction to avoid NaN
            v = Eigen::Vector3d(1.0, 0.0, 0.0);
            vnorm = 1.0;
        }
        // normalize direction
        v /= vnorm;
        // enforce orthogonality: n <- n - (n·v) v
        double ndotv = n.dot(v);
        n = n - ndotv * v;
        _estimate.head<3>() = n;
        _estimate.tail<3>() = v;
    }
};

class VertexLinePlucker : public g2o::BaseVertex<6, Eigen::Matrix<double,6,1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
    VertexLinePlucker() {}
    virtual bool read(std::istream& is)
    {
        for(int i=0; i<6; i++) is >> _estimate(i);
        return true;
    }
    virtual bool write(std::ostream& os) const
    {
        for(int i=0; i<6; i++) os << _estimate(i) << " ";
        return os.good();
    }
    // --- 重置 ---
    virtual void setToOriginImpl()
    {
        _estimate.setZero();
    }
    // --- 增量更新 ---
    virtual void oplusImpl(const double* update)
    {
        Eigen::Matrix<double,6,1> upd;
        for(int i=0; i<6; i++) upd(i) = update[i];
        _estimate += upd;
        // (可选) 正交化 Plücker 线
        enforcePluckerConstraint();
    }
    // 维度
    virtual int dimension() const { return 6; }
    // ============= Plücker 线正交约束 =============
    // n⋅v = 0 必须成立，否则投影会漂
    inline void enforcePluckerConstraint()
    {
        Eigen::Vector3d n = _estimate.head<3>();
        Eigen::Vector3d v = _estimate.tail<3>();

        double nv = n.dot(v);
        if(std::fabs(nv) < 1e-9) return;

        // 简单正交化：从 n 中减去与 v 共线部分
        Eigen::Vector3d v_norm = v.normalized();
        n = n - nv * v_norm;
        _estimate.head<3>() = n;
    }
};


class VertexLinePluckerOnlyPose : public g2o::BaseVertex<6, Eigen::Matrix<double,6,1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)

    VertexLinePluckerOnlyPose() {}

    virtual bool read(std::istream& is)
    {
        for(int i=0; i<6; i++) is >> _estimate(i);
        return true;
    }

    virtual bool write(std::ostream& os) const
    {
        for(int i=0; i<6; i++) os << _estimate(i) << " ";
        return os.good();
    }

    virtual void setToOriginImpl()
    {
        // 不动线
    }

    // --- update 忽略 ---
    virtual void oplusImpl(const double* update)
    {
        // do nothing → line stays fixed
    }

    virtual int dimension() const { return 0; }  // 关键：维度是 0
};


}

#endif

