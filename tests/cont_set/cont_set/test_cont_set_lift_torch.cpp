// test_cont_set_lift_torch - the generic lift, norm and enclose of a zonotope: gradients through
//
// torch
//
// lift permutes rows and enclose combines centers and generators linearly: differentiable. norm
// is the norm of the box around the zonotope, made of abs and max: piecewise, checked away from
// the ties.
//
// CORA: contSet/@contSet/lift.m
// CORA: contSet/@contSet/lift_.m
// CORA: contSet/@contSet/norm.m
// CORA: contSet/@contSet/norm_.m
// CORA: contSet/@contSet/enclose.m
// CORA authors: Tobias Ladner, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/zonotope_gradients.h"

#include "cora/cont_set/set.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(43);
    const auto zonotope = [](const Inputs &x) { return Zonotope(x[0], x[1]); };
    const auto S = [](const Zonotope &Z) -> const ContSet & { return Z; };
    test::gradcheck(
        "lift",
        [&](const Inputs &x) {
            return test::parts(
                S(zonotope(x)).lift(3, std::vector<int64_t>{2, 0, 1}).as<Zonotope>());
        },
        test::random_zonotope(3, 4));
    for (const Tensor::Norm type : {Tensor::Norm::one, Tensor::Norm::two, Tensor::Norm::inf})
        test::gradcheck(
            "norm", [&](const Inputs &x) { return Inputs{S(zonotope(x)).norm(type)}; },
            test::random_zonotope(3, 4));
    test::gradcheck(
        "enclose",
        [&](const Inputs &x) {
            const Zonotope Z2(x[0] + x[0].ones_like(), x[1]);
            // gradient of enclose
            return test::parts(S(zonotope(x)).enclose(Z2).as<Zonotope>());
        },
        test::random_zonotope(3, 4));
    return test::finish("cont_set lift, norm and enclose (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
