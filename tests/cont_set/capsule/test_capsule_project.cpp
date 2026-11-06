// test_capsule_project - Capsule project: the center and the generator are projected
//
// CORA: contSet/@capsule/project.m
// PROPERTY: contSet/@capsule/project.m
// CORA authors: Mark Wetzlinger
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

/// Hand-computed: dimensions in the order given, the radius kept.
void project_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0, 3.0}), column({4.0, 5.0, 6.0}), 0.5);
    const Capsule P = C.project({2, 0}).as<Capsule>();
    check(close(P.c(), {3, 1}, 0) && close(P.generator(), {6, 4}, 0) && close(P.r(), {0.5}, 0),
          b + ": {2, 0}");
    check(C.project({1})->dim() == 1, b + ": one dimension");
}

/// Stated property: the projection of every point of the capsule is in the projected capsule.
void project_property(const std::string &b) {
    Rng rng(53);
    for (int k = 0; k < 5; ++k) {
        const Capsule C = test::random_capsule(rng, 4);
        const Tensor P = test::sample_points(C, rng, 30);
        const Capsule R = C.project({3, 1}).as<Capsule>();
        check(R.contains(P.take({3, 1}, 0)), b + ": the projected points are inside");
    }
}

/// A dimension outside the set is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0, 3.0}), column({4.0, 5.0, 6.0}), 0.5);
    check(test::raises([&] { C.project({3}); }, Error::wrong_value), "project: out of range");
    ChecksScope off(false);
    check(C.project({0, 1})->dim() == 2, "project: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        project_known(b);
        project_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
