// test_ellipsoid_project - Ellipsoid project: the rows of the coordinates kept
//
// CORA: contSet/@ellipsoid/project.m
// PROPERTY: contSet/@ellipsoid/project.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a selection and an order; the empty ellipsoid takes the new dimension.
void project_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}),
                      column({1.0, 2.0, 3.0}));
    const Ellipsoid P = E.project({2, 0}).as<Ellipsoid>();
    check(close(P.Q(), {3, 0, 0, 4}, 0) && close(P.q(), {3, 1}, 0), b + ": {2, 0}");
    check(E.project({1})->dim() == 1 && Ellipsoid::empty(3).project({0, 1})->dim() == 2,
          b + ": one dimension, empty");
}

/// Stated property: projected points are in the projection.
void project_property(const std::string &b) {
    Rng rng(101);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 4, k == 3 ? 2 : -1);
        const Ellipsoid R = E.project({3, 1}).as<Ellipsoid>();
        check(test::all_inside(R, test::ellipsoid_points(E, rng, 30).take({3, 1}, 0), 1e-6),
              b + ": the projected points are inside");
    }
}

/// A dimension outside is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor::eye(3), column({1.0, 2.0, 3.0}));
    check(test::raises([&] { E.project({3}); }, Error::wrong_value), "project: out of range");
    ChecksScope off(false);
    check(E.project({0, 1})->dim() == 2, "project: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        project_known(b);
        project_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
