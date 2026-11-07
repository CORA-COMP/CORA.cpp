// test_ellipsoid_support_function_torch - Ellipsoid support_function, maps, plus: gradients
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
    torch::manual_seed(223);
    // the support function, the maps and the sum against central differences
    std::vector<torch::Tensor> d = test::random_ellipsoid_factors(3);
    d.push_back(torch::randn({3, 1}, torch::kDouble));
    test::gradcheck(
        "support_function",
        [](const Inputs &x) { return Inputs{test::ellipsoid_of(x).support_function(x[2])}; }, d);
    // the maps and the sums
    std::vector<torch::Tensor> m = test::random_ellipsoid_factors(3);
    m.push_back(torch::randn({2, 3}, torch::kDouble));
    test::gradcheck(
        "linear_map",
        [](const Inputs &x) { return test::parts(test::ellipsoid_of(x).linear_map(x[2])); }, m);
    test::gradcheck(
        "project",
        [](const Inputs &x) {
            return test::parts(test::ellipsoid_of(x).project({2, 0}).as<Ellipsoid>());
        },
        test::random_ellipsoid_factors(3));
    // the sum with a vector (that of two ellipsoids holds the weight of Halder as a host number)
    std::vector<torch::Tensor> v = test::random_ellipsoid_factors(3);
    v.push_back(torch::randn({3, 1}, torch::kDouble));
    test::gradcheck(
        "plus vector",
        [](const Inputs &x) { return test::parts(test::ellipsoid_of(x).plus(x[2])); }, v);
    return test::finish("ellipsoid support_function (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
