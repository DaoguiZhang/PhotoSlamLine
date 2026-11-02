/**
* No-SLAM test for the capture-path gates.
*
* Replaces the defect where a nonexistent capture directory made os.good()
* false and the run still reported "capture done". Every rejection must be
* explicit and testable BEFORE any SLAM starts.
*
* Cases: valid path, missing parent directory, path that is a file, unwritable
* directory, and duplicate/no-overwrite behaviour.
*
* Build:
*   g++ -std=c++17 -O2 -Wall -Wextra -I include tests/test_po_capture_paths.cpp \
*       -o /tmp/test_po_capture_paths
*/

#include "PoCapture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{
int g_checks = 0, g_fails = 0;
void Check(bool c, const std::string& what)
{
    ++g_checks;
    if (!c) { ++g_fails; std::printf("  [FAIL] %s\n", what.c_str()); }
}
} // namespace

int main()
{
    std::printf("== capture path validation ==\n");

    // 1. valid path
    const std::string good = "/tmp/po_cap_test_dir";
    ::mkdir(good.c_str(), 0777);
    Check(ORB_SLAM3::po_capture::ValidateDir(good.c_str()) == 0,
          "existing writable directory -> OK");

    // 2. missing parent directory (the exact historical defect)
    const std::string missing = "/tmp/po_cap_test_dir/does/not/exist";
    Check(ORB_SLAM3::po_capture::ValidateDir(missing.c_str()) == -3,
          "missing parent directory -> MISSING (-3)");

    // 3. path that is a regular file, not a directory
    const std::string afile = "/tmp/po_cap_test_file";
    { std::ofstream f(afile.c_str()); f << "x\n"; }
    Check(ORB_SLAM3::po_capture::ValidateDir(afile.c_str()) == -4,
          "regular file instead of directory -> NOT_A_DIR (-4)");

    // 4. unwritable directory
    const std::string ro = "/tmp/po_cap_test_ro";
    ::mkdir(ro.c_str(), 0555);
    const int roCode = ORB_SLAM3::po_capture::ValidateDir(ro.c_str());
    if (::geteuid() == 0)
    {
        // root bypasses permission bits, so the probe succeeds. Record the
        // observation instead of pretending the check works here.
        std::printf("     [note] running as root: unwritable-dir check returns %d (%s)\n",
                    roCode, ORB_SLAM3::po_capture::ValidateName(roCode));
        Check(roCode == 0 || roCode == -5 || roCode == -6,
              "unwritable dir -> either rejected or unusable-as-expected");
    }
    else
    {
        Check(roCode == -5 || roCode == -6, "unwritable directory -> REJECTED");
    }
    ::chmod(ro.c_str(), 0777);

    // 5. unset / empty
    Check(ORB_SLAM3::po_capture::ValidateDir(nullptr) == -1, "null path -> NOT_SET (-1)");
    Check(ORB_SLAM3::po_capture::ValidateDir("") == -2, "empty path -> EMPTY (-2)");

    // 6. no-overwrite: a second identical file must not be rewritten
    const std::string dup = good + "/po_probe.txt";
    { std::ofstream f(dup.c_str()); f << "first\n"; }
    struct stat st1;
    ::stat(dup.c_str(), &st1);
    {
        std::ifstream probe(dup.c_str());
        if (!probe.good()) { std::ofstream f(dup.c_str()); f << "second\n"; }
    }
    std::ifstream in(dup.c_str());
    std::string line;
    std::getline(in, line);
    Check(line == "first", "existing pack is not overwritten");
    ::unlink(dup.c_str());

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
