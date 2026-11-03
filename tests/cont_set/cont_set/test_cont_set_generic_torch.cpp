// test_cont_set_generic_torch - the generic operations of ContSet on a zonotope: gradients through torch
//
// linear_map, enlarge, minus, project_high_dim, project_on_hyperplane and decompose are linear or
// smooth in the center and generators: differentiable. reduce keeps the longest generators and
// boxes the rest: piecewise, checked away from the ties.
//
// CORA: contSet/@contSet/mtimes.m
// CORA: contSet/@contSet/enlarge.m
// CORA: contSet/@contSet/minus.m
// CORA: contSet/@contSet/projectHighDim_.m
// CORA: contSet/@contSet/projectOnHyperplane.m
// CORA: contSet/@contSet/decompose.m
// CORA: contSet/@contSet/reduce.m

#include "cont_set/zonotope/zonotope_gradients.h"

#include "cora/cont_set/set.h"

using namespace cora;
using test::Inputs;

namespace {

/// The center and generators of the Zonotope that a Set holds.
Inputs aux_parts(const Set &S) { return test::parts(S.as<Zonotope>()); }

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(41);
    const Tensor M = Tensor::from_data({1.0, 2.0, -1.0, 0.5, 1.5, 3.0}, {2, 3});
    const Tensor p = Tensor::from_data({0.5, -1.0, 2.0}, {3, 1});
    const Tensor c = Tensor::from_data({1.0, 2.0, 2.0}, {3, 1});
    const Tensor d = Tensor::from_data({0.7}, {1, 1});
    const auto zonotope = [](const Inputs &x) { return Zonotope(x[0], x[1]); };
    const auto S = [&](const Zonotope &Z) -> const ContSet & { return Z; };
    test::gradcheck("linear_map", [&](const Inputs &x) {
        return aux_parts(S(zonotope(x)).linear_map(M));
    }, test::random_zonotope(3, 4));
    test::gradcheck("enlarge", [&](const Inputs &x) {
        return aux_parts(S(zonotope(x)).enlarge(Tensor::from_data({2.0, 0.5, 3.0}, {3, 1})));
    }, test::random_zonotope(3, 4));
    test::gradcheck("minus", [&](const Inputs &x) { return aux_parts(S(zonotope(x)).minus(p)); },
                    test::random_zonotope(3, 4));
    test::gradcheck("minus from the left", [&](const Inputs &x) {
        return aux_parts(ContSet::minus(p, S(zonotope(x))));
    }, test::random_zonotope(3, 4));
    test::gradcheck("project_high_dim", [&](const Inputs &x) {
        return aux_parts(S(zonotope(x)).project_high_dim(5, {4, 0, 2}));
    }, test::random_zonotope(3, 4));
    test::gradcheck("project_on_hyperplane", [&](const Inputs &x) {
        return aux_parts(S(zonotope(x)).project_on_hyperplane(c, d));
    }, test::random_zonotope(3, 4));
    test::gradcheck("decompose", [&](const Inputs &x) {
        Inputs out;
        for (const Set &part : S(zonotope(x)).decompose({{0, 1}, {2, 2}}))
            for (const Tensor &t : aux_parts(part)) out.push_back(t);
        return out;
    }, test::random_zonotope(3, 4));
    test::gradcheck("reduce", [&](const Inputs &x) {
        return aux_parts(S(zonotope(x)).reduce(1));
    }, test::random_zonotope(2, 5));
    return test::finish("cont_set generic (torch)");
}
