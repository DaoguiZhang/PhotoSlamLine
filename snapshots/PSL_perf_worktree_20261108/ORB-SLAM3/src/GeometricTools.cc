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

#include "GeometricTools.h"

#include "KeyFrame.h"

namespace ORB_SLAM3
{

Eigen::Matrix3f GeometricTools::ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2)
{
    Sophus::SE3<float> Tc1w = pKF1->GetPose();
    Sophus::Matrix3<float> Rc1w = Tc1w.rotationMatrix();
    Sophus::SE3<float>::TranslationMember tc1w = Tc1w.translation();

    Sophus::SE3<float> Tc2w = pKF2->GetPose();
    Sophus::Matrix3<float> Rc2w = Tc2w.rotationMatrix();
    Sophus::SE3<float>::TranslationMember tc2w = Tc2w.translation();

    Sophus::Matrix3<float> Rc1c2 = Rc1w * Rc2w.transpose();
    Eigen::Vector3f tc1c2 = -Rc1c2 * tc2w + tc1w;

    Eigen::Matrix3f tc1c2x = Sophus::SO3f::hat(tc1c2);

    const Eigen::Matrix3f K1 = pKF1->mpCamera->toK_();
    const Eigen::Matrix3f K2 = pKF2->mpCamera->toK_();

    return K1.transpose().inverse() * tc1c2x * Rc1c2 * K2.inverse();
}

bool GeometricTools::Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,Eigen::Matrix<float,3,4> &Tc1w ,Eigen::Matrix<float,3,4> &Tc2w , Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f A;
    A.block<1,4>(0,0) = x_c1(0) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(0,0);
    A.block<1,4>(1,0) = x_c1(1) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(1,0);
    A.block<1,4>(2,0) = x_c2(0) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(0,0);
    A.block<1,4>(3,0) = x_c2(1) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(1,0);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);

    Eigen::Vector4f x3Dh = svd.matrixV().col(3);

    if(x3Dh(3)==0)
        return false;

    // Euclidean coordinates
    x3D = x3Dh.head(3)/x3Dh(3);

    return true;
}

Eigen::Vector4f GeometricTools::ComputeLinePlane(
        const Eigen::Vector3f& x1,
        const Eigen::Vector3f& x2,
        const Eigen::Vector3f& Ow)
{
    Eigen::Vector3f n = x1.cross(x2).normalized();
    float d = -n.dot(Ow);
    return Eigen::Vector4f(n(0), n(1), n(2), d);
}

bool GeometricTools::IntersectPlanes(
        const Eigen::Vector4f& p1,
        const Eigen::Vector4f& p2,
        Eigen::Vector3f& point,
        Eigen::Vector3f& dir)
{
    Eigen::Vector3f n1 = p1.head<3>();
    Eigen::Vector3f n2 = p2.head<3>();

    dir = n1.cross(n2);
    if (dir.norm() < 1e-6)
        return false;

    Eigen::Matrix3f A;
    A.row(0) = n1;
    A.row(1) = n2;
    A.row(2) = dir;

    Eigen::Vector3f b(-p1[3], -p2[3], 0);

    point = A.colPivHouseholderQr().solve(b);
    return true;
}

Eigen::Vector3f GeometricTools::ProjectRayToLine(
        const Eigen::Vector3f& rayDir,
        const Eigen::Vector3f& camCenter,
        const Eigen::Vector3f& linePoint,
        const Eigen::Vector3f& lineDir)
{
    Eigen::Vector3f w0 = camCenter - linePoint;
    Eigen::Vector3f a = lineDir.cross(rayDir);
    float denom = a.squaredNorm();

    if (denom < 1e-6)
        return linePoint;

    float t = a.dot(lineDir.cross(w0)) / denom;
    return linePoint + t * lineDir;
}


} //namespace ORB_SLAM
