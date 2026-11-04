// test_capsule_lift_torch - Capsule lift: gradients through torch
//
// lift is a permutation of the rows: differentiable.

#include "capsule_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(36);
    test::gradcheck(
        "lift", [](const Inputs &x) { return test::parts(test::capsule(x).lift(3, {2, 0, 1})); },
        test::random_capsule_parts(3));
    return test::finish("capsule lift (torch)");
}
