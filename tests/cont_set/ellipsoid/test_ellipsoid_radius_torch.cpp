// test_ellipsoid_radius_torch - Ellipsoid radius, volume, distance, norm, zonotope: gradients
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(227);
    // the semi-axes, the volume, the distance, the norm and the zonotope come from Q's factors
    test::gradcheck(
        "radius", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).radius(2)}; },
        test::random_ellipsoid_factors(3));
    test::gradcheck(
        "volume", [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).volume()}; },
        test::random_ellipsoid_factors(3));
    // the distance of points, the norm and the zonotope
    std::vector<torch::Tensor> p = test::random_ellipsoid_factors(3);
    p.push_back(torch::randn({3, 4}, torch::kDouble));
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
    return test::finish("ellipsoid radius (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
