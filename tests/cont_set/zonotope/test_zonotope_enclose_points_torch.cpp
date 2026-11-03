// test_zonotope_enclose_points_torch - zonotope enclose_points: gradients through torch
//
// enclose_points in a given frame is piecewise: the box follows the points that attain the
// extremes.

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(43);
    const torch::Tensor points = torch::randn({3, 12}, torch::kDouble);
    const torch::Tensor U = std::get<0>(torch::linalg_qr(torch::randn({3, 3}, torch::kDouble))).contiguous();
    test::gradcheck(
        "enclose_points",
        [](const Inputs &x) { return test::parts(Zonotope::enclose_points(x[0], x[1])); },
        {points, U});
    return test::finish("zonotope enclose_points (torch)");
}
