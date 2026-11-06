// test_polytope_cart_prod - Polytope::cart_prod: the Cartesian product
//
// CORA: contSet/@polytope/cartProd_.m
// PROPERTY: contSet/@polytope/cartProd_.m
//
// Property: (x, y) is in the product iff x is in the first and y in the second set
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Halfspaces: the block diagonal; vertices: all pairs; mixed forms; boxes; points.
void products(const std::string &b) {
    const Polytope H1(Tensor({{1.0}, {-1.0}}), column({1.0, 0.0}));
    const Polytope H2(Tensor({{1.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}), column({1.0, 0.0, 0.0}));
    const Polytope P = H1.cart_prod(H2);
    check(P.dim() == 3 && close(P.A(), {1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, -1}, 0),
          b + ": block diagonal rows");
    check(close(P.b(), {1, 0, 1, 0, 0}, 0), b + ": stacked offsets");
    check(P.contains(column({0.5, 0.2, 0.2})) && !P.contains(column({1.5, 0.2, 0.2})) &&
              !P.contains(column({0.5, 0.8, 0.8})),
          b + ": the points");
    const Polytope V1(Tensor({{0.0, 1.0}})), V2(Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    const Polytope Q = V1.cart_prod(V2);
    check(Q.is_v_rep() && Q.V().shape() == Shape{3, 6}, b + ": vertices: all pairs");
    check(close(Q.V().take({0}, -1), {0, 0, 0}, 0) && close(Q.V().take({5}, -1), {1, 0, 1}, 0),
          b + ": vertices: the order, second varies fastest");
    check(H1.cart_prod(V2).contains(column({0.5, 0.2, 0.2})), b + ": mixed forms");
    check(Polytope(Tensor::zeros({2, 0})).cart_prod(V2).V().shape() == Shape{4, 0},
          b + ": an empty vertex set");
    const Polytope boxed = H1.cart_prod(Interval(column({0.0, 0.0}), column({1.0, 2.0})));
    // the product with a box
    check(boxed.dim() == 3 && boxed.contains(column({0.5, 0.5, 1.5})) &&
              !boxed.contains(column({0.5, 1.5, 1.5})),
          b + ": with a box");
    const Polytope pt = H1.cart_prod(column({3.0, 4.0}));
    check(pt.dim() == 3 && pt.contains(column({0.5, 3.0, 4.0})) &&
              !pt.contains(column({0.5, 3.0, 5.0})),
          b + ": with a point");
    const Polytope front = Polytope::cart_prod(column({3.0}), H2);
    check(front.dim() == 3 && front.contains(column({3.0, 0.5, 0.2})) &&
              !front.contains(column({2.0, 0.5, 0.2})),
          b + ": a point first");
    check(Polytope::inf(1).cart_prod(H1).dim() == 2 && Polytope::empty(1).cart_prod(H1).dim() == 2,
          b + ": R^n and the empty set");
}

/// Batches multiply member by member.
void batches(const std::string &b) {
    const Polytope A(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1}));
    const Polytope P = A.cart_prod(A);
    check(P.A().shape() == Shape{3, 4, 4}, b + ": a batch of three");
    const Polytope V(Tensor::ones({3, 2, 4}));
    check(V.cart_prod(V).V().shape() == Shape{3, 4, 16}, b + ": a batch of vertices");
}

/// What is refused; checks off.
void refusals() {
    const Polytope A(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1}));
    const Polytope S(Tensor::ones({1, 2}), Tensor::ones({1, 1}));
    check(test::raises([&] { A.cart_prod(S); }, Error::dimension_mismatch),
          "cart_prod: batch shapes");
    check(test::raises([&] { S.cart_prod(Tensor::ones({2, 3})); }, Error::not_supported),
          "cart_prod: only single points");
    check(test::raises([&] { S.cart_prod(Tensor::zeros({2})); }, Error::dimension_mismatch),
          "cart_prod: the point is a column");
    ChecksScope off(false);
    check(S.cart_prod(S).dim() == 4, "cart_prod: the same with the checks off");
}

/// (x, y) is in the product iff x is in the first and y in the second set.
void properties(const std::string &b) {
    Rng rng(4);
    const Polytope P1 = test::random_polytope(rng, 2, 4), P2 = test::random_polytope(rng, 3, 5);
    const Polytope product = P1.cart_prod(P2);
    bool same = true;
    for (int k = 0; k < 60; ++k) {
        std::vector<double> x(5);
        rng.uniform(x.data(), x.size(), -3.5, 3.5);
        const Tensor p = column(x);
        same = same && product.contains(p) ==
                           (P1.contains(p.take({0, 1}, 0)) && P2.contains(p.take({2, 3, 4}, 0)));
    }
    check(same, b + ": the product holds exactly the pairs");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        properties(b);
        products(b);
        batches(b);
    });
    refusals();
    return test::finish("polytope cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
