// test_ellipsoid_ellipsoid_norm - Ellipsoid::ellipsoid_norm: sqrt(|p' Q^+ p|), Inf off the range
//
// CORA: contSet/@ellipsoid/ellipsoidNorm.m
// PROPERTY: contSet/@ellipsoid/ellipsoidNorm.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Hand-computed: full rank by the solve, degenerate by the decomposition, batches by member.
void ellipsoid_norm_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::finite_near(E.ellipsoid_norm(column({2.0, 0.0})), 1, 1e-12),
          b + ": half the axis is 1");
    check(test::finite_near(E.ellipsoid_norm(column({0.0, 3.0})), 3, 1e-12),
          b + ": three times the axis");
    check(test::finite_near(E.ellipsoid_norm(column({2.0, 3.0})), std::sqrt(10.0), 1e-12),
          b + ": mixed");
    const Ellipsoid segment(Tensor({{1.0, 0.0}, {0.0, 0.0}}));
    check(test::finite_near(segment.ellipsoid_norm(column({2.0, 0.0})), 2, 1e-12),
          b + ": along a segment");
    check(segment.ellipsoid_norm(column({0.0, 1.0})).item() == kInf, b + ": off a segment: Inf");
    // an entry below the threshold of the method (1e-10) is cut to zero
    check(test::finite_near(segment.ellipsoid_norm(column({2.0, 1e-12})), 2, 1e-12),
          b + ": 1e-12 is zero");
    check(segment.ellipsoid_norm(column({2.0, 1e-8})).item() == kInf, b + ": 1e-8 is not");
    const Ellipsoid point(Tensor::zeros({2, 2}));
    check(point.ellipsoid_norm(column({0.0, 0.0})).item() == 0.0 &&
              point.ellipsoid_norm(column({1.0, 0.0})).item() == kInf,
          b + ": a point: 0 at itself, Inf elsewhere");
    // a batch decides per member: a full-rank and a degenerate one
    const Tensor Qs =
        Tensor::stack({Tensor({{4.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 0.0}, {0.0, 0.0}})}, 0);
    const Tensor ps = Tensor::stack({column({0.0, 3.0}), column({0.0, 1.0})}, 0);
    const Tensor norms = Ellipsoid(Qs).ellipsoid_norm(ps);
    check(norms.shape() == Shape{2, 1, 1} && test::finite_near(norms.slice(0, 0, 1), 3.0, 1e-12) &&
              norms.slice(0, 1, 2).item() == kInf,
          b + ": a batch of a full-rank and a degenerate member");
}

/// Stated property: a norm: ||t p|| = t ||p||, and the boundary points have norm 1.
void ellipsoid_norm_property(const std::string &b) {
    cora::Rng rng(23);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k % 3 == 2 ? 2 : -1);
        const Tensor P = test::ellipsoid_points(E, rng, 5, true);
        for (int64_t j = 0; j < 5; ++j) {
            const Tensor p = P.slice(1, j, j + 1) - E.q();
            check(test::finite_near(E.ellipsoid_norm(p), 1.0), b + ": the boundary has norm 1");
            check(test::finite_near(E.ellipsoid_norm(p * 2.5), 2.5), b + ": homogeneous");
            check(E.ellipsoid_norm(-p).item() == E.ellipsoid_norm(p).item(), b + ": symmetric");
        }
    }
}

/// A wrong p and an empty ellipsoid are refused; with the checks off the norm is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.ellipsoid_norm(column({1.0})); }, Error::wrong_value),
          "ellipsoid_norm: p has the dimension of E");
    check(test::raises([&] { E.ellipsoid_norm(Tensor({{1.0, 2.0}})); }, Error::wrong_value),
          "ellipsoid_norm: p is a column");
    check(test::raises([] { Ellipsoid::empty(2).ellipsoid_norm(column({0.0, 0.0})); },
                       Error::empty_set),
          "ellipsoid_norm: the empty ellipsoid has no norm");
    const Tensor on = E.ellipsoid_norm(column({2.0, 3.0}));
    ChecksScope off(false);
    check(close(E.ellipsoid_norm(column({2.0, 3.0})), on, 0),
          "ellipsoid_norm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        ellipsoid_norm_known(b);
        ellipsoid_norm_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid ellipsoid_norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
