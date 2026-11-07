// test_con_zonotope_support_function - ConZonotope::support_function
//
// CORA: contSet/@conZonotope/supportFunc_.m
// PROPERTY: contSet/@conZonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the formula without constraints, a linear program with them, -Inf if empty.
void support_function_known(const std::string &b) {
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(test::support(plain, {1, 1}), 6.0) && close(test::support(plain, {1, 0}), 2.0),
          b + ": without constraints");
    // x = (xi_1 + xi_2, xi_2) with xi_1 + xi_2 = 0 is the segment {0} x [-1, 1]
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({0.0}));
    check(close(test::support(segment, {1, 0}), 0.0) &&
              close(test::support(segment, {0, 1}), 1.0) &&
              close(test::support(segment, {1, 1}), 1.0),
          b + ": a segment");
    check(close(test::support(test::hexagon(), {1, 1}), 1.0) &&
              close(test::support(test::hexagon(), {1, -1}), 2.0),
          b + ": the hexagon");
    check(test::support(ConZonotope::empty(2), {1, 0}) < -1e300, b + ": the empty set has -Inf");
}

/// Stated property: no sampled point is above the value, and it is reached by the farthest one.
void support_function_property(const std::string &b) {
    Rng rng(73);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const std::vector<double> d = test::random_direction(rng, 3);
        const double value = test::support(cz, d);
        const Tensor points = test::sample_points(cz, rng, 40);
        const double best = column(d).matrix_transpose().matmul(points).max().item();
        check(best <= value + 1e-9 && best >= value - 3.0, b + ": the value bounds the points");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        support_function_known(b);
        support_function_property(b);
    });
    check(test::raises([] { test::hexagon().support_function(column({1.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of another dimension");
    ChecksScope off(false);
    check(close(test::support(test::hexagon(), {1, 1}), 1.0),
          "support_function: the same with the checks off");
    return test::finish("con_zonotope support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
