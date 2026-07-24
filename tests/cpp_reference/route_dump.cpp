//
// Reference dumper for the Python CylinderRouter port.
//
// Links against the UNMODIFIED hardware controller source
// (basic_control/src/control/CylinderRouter.cpp) and evaluates whatever cases
// arrive on stdin, so tests/test_cylinder_router.py can compare the Python
// port against the real C++ implementation instead of transcribed numbers.
//
// The Python side owns the case list; this program only evaluates. One case
// per input line, whitespace separated:
//
//   enabled cx cy radius z_min z_max clearance tol  sx sy sz  tx ty tz
//
// One JSON object per output line: kind, target_adjusted, effective_target,
// length_m and the full waypoint list. All values are metres.
//

#include "control/CylinderRouter.h"

#include <iomanip>
#include <iostream>

namespace
{
    void PrintVector(const Eigen::Vector3d& v)
    {
        std::cout << '[' << v.x() << ',' << v.y() << ',' << v.z() << ']';
    }
} // namespace

int main()
{
    // 17 significant digits round-trips an IEEE-754 double exactly, so the
    // comparison tolerance in the Python test measures algorithm agreement
    // rather than text formatting loss.
    std::cout << std::setprecision(17);

    int enabled = 0;
    double cx = 0.0, cy = 0.0, radius = 0.0, z_min = 0.0, z_max = 0.0;
    double clearance = 0.0, tolerance = 0.0;
    double sx = 0.0, sy = 0.0, sz = 0.0, tx = 0.0, ty = 0.0, tz = 0.0;

    while (std::cin >> enabled >> cx >> cy >> radius >> z_min >> z_max
                    >> clearance >> tolerance >> sx >> sy >> sz
                    >> tx >> ty >> tz) {
        CylinderKeepout keepout;
        keepout.enabled = enabled != 0;
        keepout.center_xy_m = Eigen::Vector2d(cx, cy);
        keepout.radius_m = radius;
        keepout.z_min_m = z_min;
        keepout.z_max_m = z_max;
        keepout.clearance_m = clearance;
        keepout.waypoint_tolerance_m = tolerance;

        const CylinderRouter router(keepout);
        const Eigen::Vector3d start(sx, sy, sz);
        const Eigen::Vector3d target(tx, ty, tz);
        const CylinderRoute route = router.Plan(start, target);

        std::cout << "{\"kind\":\"" << CylinderRouteKindName(route.kind)
                  << "\",\"target_adjusted\":"
                  << (route.target_adjusted ? "true" : "false")
                  << ",\"size\":" << route.size
                  << ",\"length_m\":" << route.length_m
                  << ",\"effective_target\":";
        PrintVector(route.effective_target);
        std::cout << ",\"segment_intersects\":"
                  << (router.SegmentIntersects(start, target) ? "true" : "false")
                  << ",\"waypoints\":[";
        for (std::size_t i = 0; i < route.size; ++i) {
            if (i > 0)
                std::cout << ',';
            PrintVector(route.waypoints[i]);
        }
        std::cout << "]}\n";
    }
    return 0;
}
