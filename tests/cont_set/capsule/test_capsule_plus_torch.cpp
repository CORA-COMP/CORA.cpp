// test_capsule_plus_torch - Capsule plus: gradients through torch
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
    torch::manual_seed(213);
    std::vector<torch::Tensor> pair = test::random_capsule_parts(3);
    for (const torch::Tensor &t : test::random_capsule_parts(3)) pair.push_back(t);
    test::gradcheck(
        "plus",
        [](const Inputs &x) { return test::parts(test::capsule(x).plus(test::capsule(x, 3))); },
        pair);
    test::gradcheck(
        "plus vector", [](const Inputs &x) { return test::parts(test::capsule(x).plus(x[3])); },
        [] {
            std::vector<torch::Tensor> with_vector = test::random_capsule_parts(3);
            with_vector.push_back(torch::randn({3, 1}, torch::kDouble));
            return with_vector;
        }());
    return test::finish("capsule plus (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
