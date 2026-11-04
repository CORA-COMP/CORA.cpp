// test_polytope_convex_hull_torch - Polytope convex_hull: gradients through torch
//
// The extreme vertices are chosen without gradient; the result takes them from the operands.

#include "polytope_gradients.h"

using namespace cora;
using test::Inputs;

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(31);
    Rng rng(31);
    const Tensor first = test::sphere_points(rng, 2, 5);
    const Tensor second = test::sphere_points(rng, 2, 5, 2.0);
    test::gradcheck(
        "convex_hull",
        [](const Inputs &x) { return test::parts(Polytope(x[0]).convex_hull(Polytope(x[1]))); },
        {to_torch(first), to_torch(second)});
    test::gradcheck("convex_hull, point",
                    [](const Inputs &x) { return test::parts(Polytope(x[0]).convex_hull(x[1])); },
                    {to_torch(first), to_torch(Tensor({{3.0}, {3.0}}))});
    return test::finish("polytope convex_hull (torch)");
}
