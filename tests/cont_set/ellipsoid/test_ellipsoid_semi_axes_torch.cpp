// test_ellipsoid_semi_axes_torch - Ellipsoid semi_axes, volume, distance, norm, zonotope: gradients
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(227);
    // the semi-axes, the ball, the volume, the distance, the norm and the zonotope: Q's factors
    test::gradcheck(
        "semi_axes", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).semi_axes(2)}; },
        test::random_ellipsoid_factors(3));
    test::gradcheck(
        "ball_radius", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).ball_radius()}; },
        test::random_ellipsoid_factors(3));
    test::gradcheck(
        "volume", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).volume()}; },
        test::random_ellipsoid_factors(3));
    // the distance of points, the norm and the zonotope
    std::vector<torch::Tensor> p = test::random_ellipsoid_factors(3);
    // four points, one per row
    p.push_back(torch::randn({3, 4}, torch::kDouble).t().contiguous());
    test::gradcheck(
        "distance", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).distance(x[2])}; },
        p);
    test::gradcheck(
        "norm",
        [](const Inputs &x) {
            const Ellipsoid E = test::ellipsoid_of(x);
            return Inputs{Ellipsoid(E.Q()).norm()};
        },
        test::random_ellipsoid_factors(3));
    test::gradcheck(
        "zonotope", [](const Inputs &x) { return test::parts(test::ellipsoid_of(x).zonotope()); },
        test::random_ellipsoid_factors(3));
    return test::finish("ellipsoid semi_axes (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
