// test_zonotope_cubic_map_torch - zonotope cubic_map: gradients through torch, and a batch

#include "zonotope_gradients.h"

using namespace cora;
using test::check;
using test::Inputs;

namespace {

/// The map is a polynomial in c, G and T: differentiable in all of them.
void cubic_map_gradient() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 2);
    const torch::Tensor T = torch::randn({2, 2, 2, 2}, torch::kDouble);
    const auto grid = [](const Tensor &t) {
        std::vector<std::vector<Tensor>> out(2);
        for (int64_t i = 0; i < 2; ++i)
            for (int64_t k = 0; k < 2; ++k) out[i].push_back(from_torch(to_torch(t)[i][k]));
        return out;
    };
    test::gradcheck(
        "cubic_map",
        [&](const Inputs &x) { return test::parts(Zonotope(x[0], x[1]).cubic_map(grid(x[2]))); },
        {Z[0], Z[1], T});
    test::gradcheck("cubic_map (mixed)",
                    [&](const Inputs &x) {
                        const Zonotope Z2(x[0] * 0.5, x[1] * 0.7);
                        return test::parts(Zonotope(x[0], x[1]).cubic_map(Z2, Z2, grid(x[2])));
                    },
                    {Z[0], Z[1], T});
}

/// A batch on torch is the map of each member.
void cubic_map_of_a_batch() {
    const std::vector<torch::Tensor> Z = test::random_zonotope(2, 3);
    const torch::Tensor T = torch::randn({1, 2, 2, 2}, torch::kDouble);
    std::vector<std::vector<Tensor>> grid(1);
    for (int64_t k = 0; k < 2; ++k) grid[0].push_back(from_torch(T[0][k]));
    const Zonotope batch(from_torch(torch::stack({Z[0], 2 * Z[0]})),
                         from_torch(torch::stack({Z[1], Z[1] * 0.5})));
    const torch::Tensor G = to_torch(batch.cubic_map(grid).generators());
    const Zonotope member(from_torch(Z[0]), from_torch(Z[1]));
    const torch::Tensor first = to_torch(member.cubic_map(grid).generators());
    check(torch::allclose(G[0], first), "cubic_map: member 1 of a torch batch");
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(35);
    cubic_map_gradient();
    cubic_map_of_a_batch();
    return test::finish("zonotope cubic_map (torch)");
}
