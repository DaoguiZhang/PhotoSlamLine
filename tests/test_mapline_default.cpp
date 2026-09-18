// ASan/UBSan test: a default-constructed MapLine (no map, no reference KF)
// must be safe to construct, query via public getters, and destroy.
//
// Regression for: MapLine::MapLine() dereferenced an uninitialized mpMap in the
// id-allocation mutex lock (fixed to null-guard; mpMap/mpRefKF are initialized
// to nullptr).

#include <iostream>
#include "MapLine.h"

using namespace ORB_SLAM3;

int main()
{
    {
        MapLine ml;  // default ctor, no map / no reference KF

        Map* m = ml.GetMap();                       // expect nullptr
        KeyFrame* ref = ml.GetReferenceKeyFrame();  // expect nullptr
        Eigen::Matrix<double,6,1> L = ml.GetPluckerLine();
        std::pair<Eigen::Vector3f, Eigen::Vector3f> ep = ml.GetLineWorldPos();
        bool bad = ml.isBad();
        bool retr = ml.isRetrived();
        int n = ml.Observations();
        std::map<KeyFrame*, std::tuple<int,int>> obs = ml.GetLineObservations();
        Eigen::Vector3f norm = ml.GetLineNormalVector();

        std::cout << "map=" << (m ? "non-null" : "null")
                  << " refKF=" << (ref ? "non-null" : "null")
                  << " bad=" << bad
                  << " retrived=" << retr
                  << " obs=" << n
                  << " plucker_finite=" << (L.allFinite() ? 1 : 0)
                  << " endpoints_finite=" << (ep.first.allFinite() && ep.second.allFinite() ? 1 : 0)
                  << " normal_finite=" << (norm.allFinite() ? 1 : 0)
                  << std::endl;

        // Getters must return finite ZERO values (no uninitialized read).
        bool ok = (m == nullptr) && (ref == nullptr) && L.allFinite() && L.norm() == 0.0
                  && ep.first.allFinite() && ep.first.norm() == 0.0f
                  && ep.second.allFinite() && ep.second.norm() == 0.0f
                  && norm.allFinite() && norm.norm() == 0.0f;
        if (!ok)
        {
            std::cerr << "FAIL: default-constructed geometry is not finite/zero" << std::endl;
            return 1;
        }
    }
    std::cout << "PASS (default-construct + getters + destroy, finite/zero, no UB)" << std::endl;
    return 0;
}
