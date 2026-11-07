// test_capsule_project_torch - Capsule project, support_function, zonotope: torch gradients
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(217);
    test::gradcheck(
        "project",
        [](const Inputs &x) { return test::parts(test::capsule(x).project({2, 0}).as<Capsule>()); },
        test::random_capsule_parts(3));
    std::vector<torch::Tensor> d = test::random_capsule_parts(3);
    d.push_back(torch::randn({3, 1}, torch::kDouble));
    test::gradcheck(
        "support_function",
        [](const Inputs &x) { return Inputs{test::capsule(x).support_function(x[3])}; }, d);
    test::gradcheck(
        "zonotope", [](const Inputs &x) { return test::parts(test::capsule(x).zonotope()); },
        test::random_capsule_parts(3));
    return test::finish("capsule project (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
