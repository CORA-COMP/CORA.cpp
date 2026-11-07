// test_capsule_vertices_3d - Capsule vertices_3d: points on the surface of a 3D capsule
//
// CORA: contSet/@capsule/plot3D.m
// PROPERTY: contSet/@capsule/plot3D.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The points are in the capsule, and the outer halves of the two spheres are on its boundary.
void vertices_3d_property(const std::string &b) {
    Rng rng(79);
    const Capsule C = test::random_capsule(rng, 3);
    const std::vector<Point3> V = C.vertices_3d()[0];
    check(V.size() == 400, b + ": 200 per sphere");
    bool inside = true;
    int boundary = 0;
    for (const Point3 &p : V) {
        const double distance = test::axis_distance(C, {p[0], p[1], p[2]});
        inside = inside && distance <= C.r().item() + 1e-9;
        boundary += close(distance, C.r().item(), 1e-5) ? 1 : 0;
    }
    check(inside && boundary >= 150,
          b + ": every point is in the capsule, the outer halves on its boundary");
}

/// With the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0, 3.0}), column({3.0, 4.0, 5.0}), 0.5);
    ChecksScope off(false);
    check(C.vertices_3d()[0].size() == 400, "vertices_3d: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { vertices_3d_property(b); });
    checks_and_their_switch();
    return test::finish("capsule vertices_3d");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
