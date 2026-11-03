// test_zonotope_minnorm_torch - zonotope minnorm: gradients through torch
//
// minnorm is piecewise: the gradient follows the nearest facet, away from ties.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(41);
    test::gradcheck(
        "minnorm",
        [](const Inputs &x) {
            const auto [value, point] = Zonotope(x[0], x[1]).minnorm();
            return Inputs{value, point};
        },
        test::random_zonotope(2, 4));
    return test::finish("zonotope minnorm (torch)");
}
