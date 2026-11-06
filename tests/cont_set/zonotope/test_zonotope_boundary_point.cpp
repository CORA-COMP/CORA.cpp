// test_zonotope_boundary_point - zonotope boundary_point: where a ray leaves the zonotope
//
// CORA: contSet/@zonotope/boundaryPoint.m
// PROPERTY: contSet/@zonotope/boundaryPoint.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Zonotope diamond() { return Zonotope(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}})); }

/// Hand-computed rays: from the center, from another start, and in one dimension.
void known_rays(const std::string &b) {
    const Zonotope Z = diamond();
    check(close(Z.boundary_point(column({1.0, 0.0})), std::vector<double>{2, 0}, 1e-9),
          b + ": from the center along x");
    check(close(Z.boundary_point(column({3.0, 3.0})), std::vector<double>{1, 1}, 1e-9),
          b + ": the direction need not be a unit vector");
    check(close(Z.boundary_point(column({1.0, 1.0}), column({1.0, 0.0})),
                std::vector<double>{1.5, 0.5}, 1e-9),
          b + ": from (1, 0) along (1, 1)");
    const Zonotope Z1(column({3.0}), Tensor({{1.0, -2.0}}));
    check(close(Z1.boundary_point(column({-1.0})), std::vector<double>{0}, 0), b + ": 1-D, left");
    check(close(Z1.boundary_point(column({2.0})), std::vector<double>{6}, 0), b + ": 1-D, right");
}

/// Stated property: the point is in the zonotope, a little further out it is not, and it is on the
/// ray from the start.
void the_point_is_on_the_boundary(const std::string &b) {
    cora::Rng rng(27);
    for (int trial = 0; trial < 8; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 5);
        const std::vector<double> dir = test::random_direction(rng, 3);
        const Tensor d = column(dir);
        const Tensor &c = Z.center();
        const Tensor start = c + (Z.rand_point(1, rng) - c) * 0.5; // an interior point
        for (const Tensor &s : {c, start}) {
            const Tensor x = Z.boundary_point(d, s);
            check(Z.contains(x, "exact", 1e-6), b + ": the boundary point is in the set");
            check(!Z.contains(s + (x - s) * 1.001), b + ": a little further out it is not");
            const Tensor along = x - s;
            const std::vector<double> a = along.data();
            const double t = a[0] / dir[0];
            check(t > 0 && close(a[1], t * dir[1], 1e-8) && close(a[2], t * dir[2], 1e-8),
                  b + ": on the ray from the start");
        }
    }
}

/// A zero direction, a start outside, and wrong shapes are refused; with the checks off the
/// result is the same.
void checks_and_their_switch() {
    const Zonotope Z = diamond();
    check(test::raises([&] { Z.boundary_point(column({0.0, 0.0})); }, Error::wrong_value),
          "boundary_point: the zero direction");
    check(test::raises([&] { Z.boundary_point(column({1.0, 0.0}), column({5.0, 0.0})); },
                       Error::wrong_value),
          "boundary_point: a start outside");
    check(
        test::raises([&] { Z.boundary_point(column({1.0, 0.0, 0.0})); }, Error::dimension_mismatch),
        "boundary_point: a direction of dimension 3");
    check(test::raises([&] { Z.boundary_point(column({1.0, 0.0}), column({0.0})); },
                       Error::dimension_mismatch),
          "boundary_point: a start of dimension 1");
    check(test::raises([&] { Zonotope::stack({Z, Z}).boundary_point(column({1.0, 0.0})); },
                       Error::not_supported),
          "boundary_point: a batch");
    const Tensor on = Z.boundary_point(column({1.0, 0.0}));
    ChecksScope off(false);
    check(close(Z.boundary_point(column({1.0, 0.0})), on, 0), "boundary_point: checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        known_rays(b);
        the_point_is_on_the_boundary(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope boundary_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
