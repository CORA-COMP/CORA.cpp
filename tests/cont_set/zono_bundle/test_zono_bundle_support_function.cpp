// test_zono_bundle_support_function - ZonoBundle::support_function
//
// CORA: contSet/@zonoBundle/supportFunc_.m
// PROPERTY: contSet/@zonoBundle/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The members [-2, 2] x [-1, 1] and [-1, 3] x [-1, 1] meet in [-1, 2] x [-1, 1].
void support_function_known(const std::string &b) {
    const Zonotope A(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const Zonotope B(column({1.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const ZonoBundle zono_bundle({A, B});
    check(close(test::support(zono_bundle, {1, 0}), 2.0) &&
              close(test::support(zono_bundle, {-1, 0}), 1.0) &&
              close(test::support(zono_bundle, {0, 1}), 1.0) &&
              close(test::support(zono_bundle, {1, 1}), 3.0),
          b + ": the box [-1, 2] x [-1, 1]");
    check(test::support(ZonoBundle::empty(2), {1, 0}) < -1e300, b + ": the empty set has -Inf");
}

/// Stated property: the value bounds d'x of every point of the bundle and is reached by a vertex
/// of the box when the directions are the axes.
void support_function_property(const std::string &b) {
    Rng rng(68);
    for (int k = 0; k < 3; ++k) {
        Tensor common;
        const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 4, &common);
        const std::vector<double> d = test::random_direction(rng, 3);
        const double value = test::support(zono_bundle, d);
        const Tensor points = test::bundle_points(zono_bundle, rng, 30, common);
        const double best = column(d).matrix_transpose().matmul(points).max().item();
        check(best <= value + 1e-9, b + ": no point is above the value");
        check(value <= test::support(zono_bundle.zonotopes()[0], d) + 1e-9,
              b + ": below a member's value");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        support_function_known(b);
        support_function_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.support_function(column({1.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of another dimension");
    ChecksScope off(false);
    check(close(test::support(zono_bundle, {1, 0}),
                test::support(zono_bundle.con_zonotope(), {1, 0})),
          "support_function: the same with the checks off");
    return test::finish("zono_bundle support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
