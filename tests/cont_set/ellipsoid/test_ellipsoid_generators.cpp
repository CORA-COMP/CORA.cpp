// test_ellipsoid_generators - Ellipsoid::generators: G with G G' = Q
//
// CORA: contSet/@ellipsoid/generators.m
// PROPERTY: contSet/@ellipsoid/generators.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a diagonal Q has the roots of its entries as generators, up to order and sign.
void generators_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    const Tensor G = E.generators();
    check(G.shape() == Shape{2, 2}, b + ": a square matrix");
    check(close(G.matmul(G.matrix_transpose()), Tensor({{4.0, 0.0}, {0.0, 1.0}}), 1e-12),
          b + ": G G' is Q");
    check(close(Ellipsoid(Tensor::zeros({2, 2})).generators(), Tensor::zeros({2, 2}), 0),
          b + ": a point has zero generators");
    check(Ellipsoid::empty(2).generators().shape() == Shape{2, 0}, b + ": empty: no columns");
}

/// Stated property: G G' = Q for full-rank, degenerate and batched ellipsoids.
void generators_property(const std::string &b) {
    cora::Rng rng(3);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 4, k % 3 == 0 ? 2 : -1);
        const Tensor G = E.generators();
        check(close(G.matmul(G.matrix_transpose()), E.Q(), 1e-10), b + ": G G' is Q");
    }
    const Tensor Q =
        Tensor::stack({Tensor({{2.0, 1.0}, {1.0, 2.0}}), Tensor({{1.0, 0.0}, {0.0, 3.0}})}, 0);
    const Tensor G = Ellipsoid(Q).generators();
    check(close(G.matmul(G.matrix_transpose()), Q, 1e-12), b + ": a batch");
}

/// No input to get wrong; with the checks off the call is the same.
void switch_off() {
    const Ellipsoid E(Tensor({{2.0, 1.0}, {1.0, 2.0}}));
    const Tensor on = E.generators();
    ChecksScope off(false);
    check(close(E.generators(), on, 0), "generators: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        generators_known(b);
        generators_property(b);
    });
    switch_off();
    return test::finish("ellipsoid generators");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
