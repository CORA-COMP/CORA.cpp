// test_con_zonotope_cart_prod - ConZonotope::cart_prod: the product with a set or a point
//
// CORA: contSet/@conZonotope/cartProd_.m
// PROPERTY: contSet/@conZonotope/cartProd_.m

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the blocks of the generators and of the constraints; a zonotope, a box and a
/// point.
void cart_prod_known(const std::string &b) {
    const ConZonotope cz1(column({1.0}), Tensor({{2.0}}), Tensor({{1.0}}), column({0.5}));
    const ConZonotope cz2(column({3.0, 4.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope P = cz1.cart_prod(cz2);
    check(close(P.c(), {1, 3, 4}, 0), b + ": the centers stack");
    check(close(P.generators(), {2, 0, 0, 0, 1, 0, 0, 0, 1}, 0), b + ": block diagonal generators");
    check(close(P.A(), {1, 0, 0}, 0) && close(P.b(), {0.5}, 0), b + ": the constraints stay");
    // the constraints of the second set come second
    const ConZonotope Q = cz2.cart_prod(cz1);
    check(close(Q.A(), {0, 0, 1}, 0), b + ": its constraints on its own factors");
    const ConZonotope with_zonotope = cz1.cart_prod(Zonotope(column({5.0}), Tensor({{6.0}})));
    check(close(with_zonotope.c(), {1, 5}, 0) && with_zonotope.A().shape() == Shape{1, 2},
          b + ": a zonotope");
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    check(close(cz1.cart_prod(I).c(), {1, 1, 0}, 0), b + ": a box");
    const ConZonotope with_point = cz1.cart_prod(column({9.0}));
    check(close(with_point.c(), {1, 9}, 0) && with_point.generators().shape() == Shape{2, 1},
          b + ": a point adds a zero row");
    const ConZonotope point_first = ConZonotope::cart_prod(column({9.0}), cz1);
    check(close(point_first.c(), {9, 1}, 0) && close(point_first.generators(), {0, 2}, 0),
          b + ": a point first");
    const Set via_set = cart_prod(cz1, I);
    check(via_set.is<ConZonotope>() && via_set.as<ConZonotope>().dim() == 3,
          b + ": through the registry");
}

/// Stated property: the product of the points of two sets is in the product, and the box is the
/// product of the boxes.
void cart_prod_property(const std::string &b) {
    cora::Rng rng(25);
    const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1),
                      B = test::random_con_zonotope(rng, 3, 5, 2);
    const ConZonotope P = A.cart_prod(B);
    const Tensor pa = test::sample_points(A, rng, 20), pb = test::sample_points(B, rng, 20);
    check(P.contains(Tensor::concat({pa, pb}, 0), "exact", 1e-7),
          b + ": the points of both are in the product");
    const Interval ia = A.interval(), ib = B.interval(), ip = P.interval();
    check(close(ip.inf(), Tensor::concat({ia.inf(), ib.inf()}, 0), 1e-9) &&
              close(ip.sup(), Tensor::concat({ia.sup(), ib.sup()}, 0), 1e-9),
          b + ": the box of the product is the product of the boxes");
}

/// A batch of products, member by member.
void cart_prod_of_batches(const std::string &b) {
    const ConZonotope A(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 2}),
                        Tensor::zeros({2, 1, 1}));
    const ConZonotope B(Tensor::ones({2, 2, 1}), Tensor::ones({2, 2, 1}));
    const ConZonotope P = A.cart_prod(B);
    check(P.generators().shape() == Shape{2, 3, 3} && P.A().shape() == Shape{2, 1, 3},
          b + ": batch shapes");
    check(close(P.A().slice(0, 1, 2).squeeze(0), {1, 1, 0}, 0), b + ": member 2");
}

/// Wrong operands are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope cz(column({1.0}), Tensor({{2.0}}));
    check(test::raises([&] { cz.cart_prod(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "cart_prod: a point must be a column");
    check(test::raises([&] { ConZonotope::cart_prod(Tensor({{1.0, 2.0}}), cz); },
                       Error::dimension_mismatch),
          "cart_prod: a point first must be a column");
    check(test::raises(
              [&] { cz.cart_prod(ConZonotope(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 1}))); },
              Error::dimension_mismatch),
          "cart_prod: a batch and a single set");
    const Tensor on = cz.cart_prod(cz).generators();
    ChecksScope off(false);
    check(close(cz.cart_prod(cz).generators(), on, 0), "cart_prod: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cart_prod_known(b);
        cart_prod_property(b);
        cart_prod_of_batches(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope cart_prod");
}
