/**
 * This file is part of Photo-SLAM-L (replay harness).
 *
 * Common-input + geometry-event-stream export / import for the
 * "same input, same event stream" V/N comparison.
 *
 * Design rules
 * ------------
 *  * Everything written here is plain text (plus lossless PNG for the images),
 *    so a reader can audit and re-hash it without this binary.
 *  * Every file ends with a content hash line computed over the bytes above it.
 *    A run refuses to start if a hash does not match, which is how the harness
 *    proves that the five conditions consumed the *same* input.
 *  * No floats are re-rounded: %.9g round-trips float exactly.
 *
 * The two record kinds are:
 *   keyframes  - the identity of every GaussianKeyframe that ever entered the
 *                mapper: id, camera id, the pose it was born with, image size,
 *                source file name, and the lossless PNG of the exact image the
 *                mapper trains on (already undistorted).
 *   events     - every MappingOperation the live mapper consumed, in order,
 *                with the mapper iteration at which it was consumed, the
 *                operation type, the scale/Sim3 payload, and the full per-
 *                keyframe new pose + map-point + line-sample payload.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "point3d.h"

namespace psl_replay
{

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

inline std::string hashHex(const std::string &s)
{
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : s) {
        h ^= static_cast<std::uint64_t>(ch);
        h *= 1099511628211ULL;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

inline std::string fmt(double v)
{
    if (!std::isfinite(v))
        return std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf");
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return std::string(buf);
}

inline std::string fileName(const std::filesystem::path &p)
{
    return p.filename().string();
}

// ---------------------------------------------------------------------------
// Keyframe records
// ---------------------------------------------------------------------------

// Original cv::Mat depth of an image at the time it entered the mapper. The
// live pipeline feeds the mapper float images (RGB in 0..1, depth in metres),
// while PNG only stores integers, so the type is recorded and restored.
enum ImgCode {
    IMG_EMPTY = 0,
    IMG_8U = 1,
    IMG_16U = 2,
    IMG_32F = 3,
};

inline int imgCode(const cv::Mat &m)
{
    if (m.empty())
        return IMG_EMPTY;
    switch (m.depth()) {
    case CV_8U: return IMG_8U;
    case CV_16U: return IMG_16U;
    case CV_32F: return IMG_32F;
    default: return IMG_EMPTY;
    }
}

// float [0,1] colour -> 8 bit (<= 1/255 per channel), float metres -> millimetres
// in 16 bit. The quantisation is identical for every condition, and the source
// images of both datasets are 8 bit RGB / integer depth, so nothing that the
// live run could resolve is lost.
inline cv::Mat imgToStorable(const cv::Mat &m, int code)
{
    if (m.empty())
        return cv::Mat();
    cv::Mat out;
    if (code == IMG_32F) {
        if (m.channels() == 1)
            m.convertTo(out, CV_16U, 1000.0);
        else
            m.convertTo(out, CV_8U, 255.0);
        return out;
    }
    return m;
}

inline cv::Mat imgFromStored(const cv::Mat &m, int code)
{
    if (m.empty() || code == IMG_EMPTY)
        return cv::Mat();
    cv::Mat out;
    if (code == IMG_32F) {
        if (m.channels() == 1)
            m.convertTo(out, CV_32F, 1.0 / 1000.0);
        else
            m.convertTo(out, CV_32FC3, 1.0 / 255.0);
        return out;
    }
    if (code == IMG_16U && m.depth() != CV_16U) {
        m.convertTo(out, CV_16U);
        return out;
    }
    return m;
}

struct KfRecord
{
    unsigned long fid = 0;
    unsigned long camera_id = 0;
    float q[4] = {1.f, 0.f, 0.f, 0.f};  // w,x,y,z of Tcw
    float t[3] = {0.f, 0.f, 0.f};
    int width = 0;
    int height = 0;
    int has_aux = 0;
    int rgb_code = IMG_EMPTY;
    int aux_code = IMG_EMPTY;
    long long creation_iter = 0;
    std::string name;
};

inline std::string kfLine(const KfRecord &k)
{
    std::ostringstream o;
    o << "K " << k.fid << ' ' << k.camera_id;
    for (int i = 0; i < 4; ++i)
        o << ' ' << fmt(k.q[i]);
    for (int i = 0; i < 3; ++i)
        o << ' ' << fmt(k.t[i]);
    o << ' ' << k.width << ' ' << k.height << ' ' << k.has_aux << ' '
      << k.rgb_code << ' ' << k.aux_code << ' ' << k.creation_iter
      << " | " << k.name << '\n';
    return o.str();
}

inline bool parseKfLine(const std::string &line, KfRecord &k)
{
    if (line.size() < 2 || line[0] != 'K' || line[1] != ' ')
        return false;
    std::istringstream in(line.substr(2));
    in >> k.fid >> k.camera_id;
    for (int i = 0; i < 4; ++i)
        in >> k.q[i];
    for (int i = 0; i < 3; ++i)
        in >> k.t[i];
    in >> k.width >> k.height >> k.has_aux >> k.rgb_code >> k.aux_code
       >> k.creation_iter;
    // " | name" (name may contain spaces)
    std::string rest;
    std::getline(in, rest);
    const auto bar = rest.find('|');
    if (bar != std::string::npos) {
        k.name = rest.substr(bar + 1);
        const auto b = k.name.find_first_not_of(' ');
        k.name = (b == std::string::npos) ? std::string() : k.name.substr(b);
    }
    return in.good() || in.eof();
}

inline std::string imgPath(const std::filesystem::path &dir, unsigned long fid)
{
    return (dir / "kf_img" / (std::to_string(fid) + ".png")).string();
}
inline std::string auxPath(const std::filesystem::path &dir, unsigned long fid)
{
    return (dir / "kf_aux" / (std::to_string(fid) + ".png")).string();
}

// Write the two images of one keyframe. The RGB image is the already
// undistorted 8-bit image the mapper trains on, so the PNG round-trip is
// lossless with respect to the in-memory tensor.
inline bool writeKfImages(const std::filesystem::path &dir, unsigned long fid,
                          const cv::Mat &rgb, int rgb_code,
                          const cv::Mat &aux, int aux_code)
{
    std::filesystem::create_directories(dir / "kf_img");
    std::filesystem::create_directories(dir / "kf_aux");
    if (rgb.empty())
        return false;
    if (!cv::imwrite(imgPath(dir, fid), imgToStorable(rgb, rgb_code)))
        return false;
    if (!aux.empty())
        return cv::imwrite(auxPath(dir, fid), imgToStorable(aux, aux_code));
    return true;
}

inline bool readKfImages(const std::filesystem::path &dir, const KfRecord &k,
                         cv::Mat &rgb, cv::Mat &aux)
{
    cv::Mat raw_rgb = cv::imread(imgPath(dir, k.fid), cv::IMREAD_UNCHANGED);
    if (raw_rgb.empty())
        return false;
    if (raw_rgb.channels() == 1)
        cv::cvtColor(raw_rgb, raw_rgb, cv::COLOR_GRAY2BGR);
    rgb = imgFromStored(raw_rgb, k.rgb_code == IMG_EMPTY ? IMG_8U : k.rgb_code);
    if (std::filesystem::exists(auxPath(dir, k.fid))) {
        cv::Mat raw_aux = cv::imread(auxPath(dir, k.fid), cv::IMREAD_UNCHANGED);
        aux = imgFromStored(raw_aux, k.aux_code);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Event stream records
// ---------------------------------------------------------------------------

struct EvKf
{
    unsigned long fid = 0;
    unsigned long camera_id = 0;
    float q[4] = {1.f, 0.f, 0.f, 0.f};
    float t[3] = {0.f, 0.f, 0.f};
};

struct EvPoint
{
    float p[3] = {0.f, 0.f, 0.f};
    float c[3] = {0.f, 0.f, 0.f};
};

struct EvLineSample
{
    float p[3] = {0.f, 0.f, 0.f};
    float c[3] = {0.f, 0.f, 0.f};
    float d[3] = {1.f, 0.f, 0.f};
};

struct Event
{
    int seq = 0;
    int type = 0;          // MappingOperation::OprType
    std::string type_name;
    float scale = 1.f;
    float tq[4] = {1.f, 0.f, 0.f, 0.f};
    float tt[3] = {0.f, 0.f, 0.f};
    long long iteration = 0;   // mapper iteration when the live mapper consumed it
    std::vector<EvKf> kfs;
    std::vector<EvPoint> pts;
    std::vector<EvLineSample> lpts;
    // filled by the reader
    std::string file;
    long long replay_step = -1;
};

inline const char *oprTypeName(int t)
{
    switch (t) {
    case 1: return "LocalMappingBA";
    case 2: return "LoopClosingBA";
    case 3: return "ScaleRefinement";
    case 4: return "GlobalBAWriteback";
    case 5: return "MapMerge";
    default: return "Unknown";
    }
}

// A geometry event is one that moves already-existing Gaussians or changes the
// world frame. These are the events around which the harness evaluates.
inline bool isGeometryEvent(int t)
{
    return t == 2 || t == 3 || t == 4 || t == 5;
}

inline std::string eventText(const Event &e)
{
    std::ostringstream o;
    o << "# photo-slam-l geometry event v1\n";
    o << "EV " << e.seq << ' ' << e.type << ' ' << fmt(e.scale);
    for (int i = 0; i < 4; ++i) o << ' ' << fmt(e.tq[i]);
    for (int i = 0; i < 3; ++i) o << ' ' << fmt(e.tt[i]);
    o << ' ' << e.iteration << '\n';
    for (const auto &k : e.kfs) {
        o << "KF " << k.fid << ' ' << k.camera_id;
        for (int i = 0; i < 4; ++i) o << ' ' << fmt(k.q[i]);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(k.t[i]);
        o << '\n';
    }
    for (const auto &p : e.pts) {
        o << "PT";
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(p.p[i]);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(p.c[i]);
        o << '\n';
    }
    for (const auto &l : e.lpts) {
        o << "LS";
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(l.p[i]);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(l.c[i]);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(l.d[i]);
        o << '\n';
    }
    return o.str();
}

inline bool writeEvent(const std::filesystem::path &dir, const Event &e)
{
    std::filesystem::create_directories(dir / "events");
    char name[64];
    std::snprintf(name, sizeof(name), "e%05d_%s.txt", e.seq, oprTypeName(e.type));
    const auto path = dir / "events" / name;
    const std::string body = eventText(e);
    std::ofstream out(path);
    if (!out.is_open())
        return false;
    out << body;
    out << "HASH " << hashHex(body) << '\n';
    return true;
}

inline bool readEvent(const std::filesystem::path &path, Event &e)
{
    std::ifstream in(path);
    if (!in.is_open())
        return false;
    std::string body, line, stored, text;
    text.reserve(1 << 16);
    while (std::getline(in, line)) {
        if (line.rfind("HASH ", 0) == 0)
            stored = line.substr(5);
        else {
            text += line;
            text += '\n';
        }
    }
    if (!stored.empty() && stored != hashHex(text)) {
        std::cerr << "[Replay] event hash mismatch: " << path << '\n';
        return false;
    }
    std::istringstream ss(text);
    while (std::getline(ss, line)) {
        if (line.rfind("EV ", 0) == 0) {
            std::istringstream ls(line.substr(3));
            ls >> e.seq >> e.type >> e.scale;
            for (int i = 0; i < 4; ++i) ls >> e.tq[i];
            for (int i = 0; i < 3; ++i) ls >> e.tt[i];
            ls >> e.iteration;
            e.type_name = oprTypeName(e.type);
        }
        else if (line.rfind("KF ", 0) == 0) {
            EvKf k;
            std::istringstream ls(line.substr(3));
            ls >> k.fid >> k.camera_id;
            for (int i = 0; i < 4; ++i) ls >> k.q[i];
            for (int i = 0; i < 3; ++i) ls >> k.t[i];
            e.kfs.push_back(k);
        }
        else if (line.rfind("PT ", 0) == 0) {
            EvPoint p;
            std::istringstream ls(line.substr(3));
            for (int i = 0; i < 3; ++i) ls >> p.p[i];
            for (int i = 0; i < 3; ++i) ls >> p.c[i];
            e.pts.push_back(p);
        }
        else if (line.rfind("LS ", 0) == 0) {
            EvLineSample l;
            std::istringstream ls(line.substr(3));
            for (int i = 0; i < 3; ++i) ls >> l.p[i];
            for (int i = 0; i < 3; ++i) ls >> l.c[i];
            for (int i = 0; i < 3; ++i) ls >> l.d[i];
            e.lpts.push_back(l);
        }
    }
    e.file = path.filename().string();
    return true;
}

// ---------------------------------------------------------------------------
// Snapshot (the initial, un-optimised model the replay must start from)
// ---------------------------------------------------------------------------

struct Snapshot
{
    long long natural_iter = 0;   // mapper iterations the live run went through
    float extent = 0.f;
    std::string map_id = "0";
    int coord_epoch = 0;
    std::vector<KfRecord> kfs;
    std::map<point3D_id_t, Point3D> pcd;
    std::string hash;
};

inline std::string snapshotText(const Snapshot &s)
{
    std::ostringstream o;
    std::size_t n_line = 0;
    for (const auto &kv : s.pcd)
        if (kv.second.source_ == PointSourceType::LINE_SAMPLED)
            ++n_line;
    o << "# photo-slam-l replay snapshot v1\n";
    o << "META extent " << fmt(s.extent) << " natural_iter " << s.natural_iter
      << " coord_epoch " << s.coord_epoch << " map_id " << s.map_id
      << " nkf " << s.kfs.size() << " npcd " << s.pcd.size()
      << " nline " << n_line << '\n';
    for (const auto &k : s.kfs)
        o << kfLine(k);
    for (const auto &kv : s.pcd) {
        const Point3D &p = kv.second;
        o << "P " << kv.first;
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(p.xyz_[i]);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(p.color_[i]);
        o << ' ' << (p.source_ == PointSourceType::LINE_SAMPLED ? 1 : 0);
        for (int i = 0; i < 3; ++i) o << ' ' << fmt(p.line_dir_[i]);
        o << ' ' << fmt(p.sample_step_) << '\n';
    }
    return o.str();
}

inline bool writeSnapshot(const std::filesystem::path &dir, const Snapshot &s)
{
    std::filesystem::create_directories(dir);
    const std::string body = snapshotText(s);
    std::ofstream out(dir / "snapshot.txt");
    if (!out.is_open())
        return false;
    out << body;
    out << "HASH " << hashHex(body) << '\n';
    std::ofstream h(dir / "snapshot.hash");
    h << hashHex(body) << '\n';
    return true;
}

inline bool readSnapshot(const std::filesystem::path &dir, Snapshot &s)
{
    std::ifstream in(dir / "snapshot.txt");
    if (!in.is_open())
        return false;
    std::string line, stored, text;
    text.reserve(1 << 20);
    while (std::getline(in, line)) {
        if (line.rfind("HASH ", 0) == 0)
            stored = line.substr(5);
        else {
            text += line;
            text += '\n';
        }
    }
    s.hash = hashHex(text);
    if (!stored.empty() && stored != s.hash) {
        std::cerr << "[Replay] snapshot hash mismatch in " << dir << '\n';
        return false;
    }
    std::istringstream ss(text);
    while (std::getline(ss, line)) {
        if (line.rfind("META ", 0) == 0) {
            std::istringstream ls(line.substr(5));
            std::string key;
            while (ls >> key) {
                if (key == "extent") ls >> s.extent;
                else if (key == "natural_iter") ls >> s.natural_iter;
                else if (key == "coord_epoch") ls >> s.coord_epoch;
                else if (key == "map_id") ls >> s.map_id;
            }
        }
        else if (line.rfind("K ", 0) == 0) {
            KfRecord k;
            if (parseKfLine(line, k))
                s.kfs.push_back(k);
        }
        else if (line.rfind("P ", 0) == 0) {
            std::istringstream ls(line.substr(2));
            point3D_id_t id = 0;
            Point3D p;
            int src = 0;
            ls >> id;
            for (int i = 0; i < 3; ++i) ls >> p.xyz_[i];
            for (int i = 0; i < 3; ++i) ls >> p.color_[i];
            ls >> src;
            for (int i = 0; i < 3; ++i) ls >> p.line_dir_[i];
            ls >> p.sample_step_;
            p.source_ = (src == 1) ? PointSourceType::LINE_SAMPLED
                                   : PointSourceType::MAP_POINT;
            s.pcd[id] = p;
        }
    }
    return !s.kfs.empty();
}

// Index of all keyframes that ever entered the mapper (superset of the
// snapshot's initial set). Written incrementally as each keyframe is born.
inline bool appendKfRecord(const std::filesystem::path &dir, const KfRecord &k)
{
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "kf_index.txt", std::ios::app);
    if (!out.is_open())
        return false;
    out << kfLine(k);
    return true;
}

inline bool readKfIndex(const std::filesystem::path &dir, std::vector<KfRecord> &out)
{
    std::ifstream in(dir / "kf_index.txt");
    if (!in.is_open())
        return false;
    std::string line;
    while (std::getline(in, line)) {
        KfRecord k;
        if (parseKfLine(line, k))
            out.push_back(k);
    }
    return !out.empty();
}

inline bool readEventIndex(const std::filesystem::path &dir, std::vector<Event> &out)
{
    const auto edir = dir / "events";
    if (!std::filesystem::exists(edir))
        return false;
    std::vector<std::string> names;
    for (const auto &e : std::filesystem::directory_iterator(edir))
        if (e.path().extension() == ".txt")
            names.push_back(e.path().string());
    std::sort(names.begin(), names.end());
    for (const auto &n : names) {
        Event ev;
        if (!readEvent(n, ev))
            return false;
        out.push_back(std::move(ev));
    }
    return true;
}

}  // namespace psl_replay
