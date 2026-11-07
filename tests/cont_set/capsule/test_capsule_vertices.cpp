// test_capsule_vertices - Capsule vertices: the outline of a planar capsule
//
// CORA: contSet/@capsule/vertices_.m
// PROPERTY: contSet/@capsule/vertices_.m
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

/// A point gives one vertex; the outline of a planar capsule is closed and at the radius from the
/// axis.
void vertices_property(const std::string &b) {
    const std::vector<Polygon> point = Capsule(column({1.0, 2.0})).vertices();
    check(point.size() == 1 && point[0].size() == 1, b + ": a point");
    Rng rng(73);
    for (int k = 0; k < 3; ++k) {
        const Capsule C = test::random_capsule(rng, 2);
        const Polygon V = C.vertices()[0];
        check(V.size() == 1001 && V.front() == V.back(),
              b + ": two half circles of 500 and the first vertex");
        bool on = true;
        for (const Point &p : V)
            on = on && close(test::axis_distance(C, {p[0], p[1]}), C.r().item(), 1e-5);
        check(on, b + ": every vertex is at the radius from the axis");
    }
}

/// A set of another dimension is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0, 3.0}), column({3.0, 4.0, 5.0}), 0.5);
    check(test::raises([&] { C.vertices(); }, Error::not_supported), "vertices: 2D only");
    const Capsule D(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    ChecksScope off(false);
    check(D.vertices()[0].size() == 1001, "vertices: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { vertices_property(b); });
    checks_and_their_switch();
    return test::finish("capsule vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
