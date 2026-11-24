// test_specification_robustness - the signed distance of points to a specification
//
// CORA: specification/@specification/robustness.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/specification/specification.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [lo1, hi1] x [lo2, hi2] as a polytope.
Polytope box(double lo1, double hi1, double lo2, double hi2) {
    return {Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
            vec({hi1, -lo1, hi2, -lo2})};
}


Polytope cube() {
    const Tensor I = Tensor::eye(3);
    return {Tensor::concat({I, -I}, 0), vec({1.0, 1.0, 1.0, 0.0, 0.0, 0.0})};
}

/// Inside: the distance to the nearest facet; outside: to the one hyperplane, the vertex, or in
/// the 1-norm; an unsafe set reverses the signs (the values CORALean's coverage row cites).
void values(const std::string &b) {
    const Specification safe = Specification::safe_set(box(0, 1, 0, 1));
    const Specification unsafe = Specification::unsafe_set(box(0, 1, 0, 1));
    const Tensor P({{0.25, 0.5}, {2.0, 0.5}, {2.0, 2.0}, {0.5, -0.75}});
    check(test::close(robustness(safe, P), {0.25, -1.0, -std::sqrt(2.0), -0.75}),
          b + ": a safe set: inside, one hyperplane, a vertex");
    check(test::close(robustness(unsafe, P), {-0.25, 1.0, std::sqrt(2.0), 0.75}),
          b + ": an unsafe set: the signs reversed");
    check(test::close(robustness(Specification::invariant(box(0, 1, 0, 1)), P),
                      {0.25, -1.0, -std::sqrt(2.0), -0.75}),
          b + ": an invariant as a safe set");
    const Specification c = Specification::safe_set(cube());
    check(test::close(robustness(c, Tensor({{2.0, 2.0, 0.5}})), {-2.0}),
          b + ": the 1-norm distance (two of three rows violated)");
    check(test::close(robustness(c, Tensor({{2.0, 2.0, 2.0}})), {-std::sqrt(3.0)}),
          b + ": the vertex of three rows");
    check(test::close(robustness({safe, unsafe}, P), {-0.25, -1.0, -std::sqrt(2.0), -0.75}),
          b + ": the least over a list");
}

/// A set of a lower dimension: CORA ignores the equalities and gives 0 for a point off the set; the
/// sign must say violated (tests/ docs/cora-differences.md).
void equalities(const std::string &b) {
    const Polytope segment(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), vec({1.0, 0.0}), Tensor({{0.0, 1.0}}),
                           vec({0.0}));
    const Specification safe = Specification::safe_set(segment);
    check(test::close(robustness(safe, Tensor({{0.5, 1.0}})), {-1.0}),
          b + ": a point off a segment is outside, by 1");
    check(test::close(robustness(safe, Tensor({{0.5, 0.0}})), {0.0}), b + ": a point on it is 0");
    check(test::close(robustness(Specification::unsafe_set(segment), Tensor({{0.5, 1.0}})), {1.0}),
          b + ": the unsafe segment is cleared by 1");
}

void trajectory(const std::string &b) {
    const Specification spec = Specification::safe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0);
    const Tensor P({{0.5, 0.5}, {2.0, 0.5}, {5.0, 5.0}});
    check(test::close(robustness(spec, P, {0.0, 1.5, 3.0}), -1.0),
          b + ": only the point in the window");
    check(test::close(robustness(Specification::safe_set(box(0, 1, 0, 1)), P, {0.0, 1.5, 3.0}),
                      -std::sqrt(32.0 + 0.0)),
          b + ": untimed: the least over all (distance 4 + 4 to (5,5): vertex (1,1))");
    // no sample in the window: interpolated at its center 1.5 between t = 1 and t = 2
    const Tensor Q({{0.0, 0.5}, {1.0, 0.5}});
    check(test::close(robustness(Specification::safe_set(box(0, 1, 0, 1)).with_time(1.2, 1.8), Q,
                                 {1.0, 2.0}),
                      0.5),
          b + ": interpolated at the center of the window");
    check(test::raises([&] { robustness(spec, P, {0.0, 0.5, 0.7}); }, Error::special_error),
          b + ": no point in the window and none to interpolate");
    check(test::raises([&] { robustness(spec, P); }, Error::special_error), b + ": times needed");
    check(test::raises([&] { robustness(spec, P, {0.0, 1.5}); }, Error::special_error),
          b + ": one time per point");
    check(test::raises(
              [&] { robustness(Specification::custom([](const ContSet &) { return true; }), P); },
              Error::not_supported),
          b + ": custom");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        values(b);
        equalities(b);
        trajectory(b);
    });
    return test::finish("test_specification_robustness");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
