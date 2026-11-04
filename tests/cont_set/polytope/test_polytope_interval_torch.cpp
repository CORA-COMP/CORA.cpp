// test_polytope_interval_torch - Polytope interval and box: gradients through torch
//
// The bounds of vertices are their coordinate-wise extremes: piecewise, away from ties.

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    const std::vector<torch::Tensor> V{torch::randn({3, 5}, torch::kDouble)};
    test::gradcheck(
        "interval",
        [](const Inputs &x) {
            const Interval I = Polytope(x[0]).interval();
            return Inputs{I.inf(), I.sup()};
        },
        V);
    test::gradcheck("box", [](const Inputs &x) { return test::parts(Polytope(x[0]).box()); }, V);
    return test::finish("polytope interval (torch)");
}
