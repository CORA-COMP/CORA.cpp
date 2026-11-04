// test_ellipsoid_cart_prod - Ellipsoid::cart_prod: exact with a point, the box product otherwise
//
// CORA: contSet/@ellipsoid/cartProd_.m
// PROPERTY: contSet/@ellipsoid/cartProd_.m

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: a point pads Q with zeros; two ellipsoids give k diag(r^2) with the radii r of
/// the boxes (5e-3 at least) and k the number of nonzero radii.
void cart_prod_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0}}), column({1.0}));
    const Ellipsoid right = E.cart_prod(column({5.0, 6.0}));
    check(close(right.Q(), {4, 0, 0, 0, 0, 0, 0, 0, 0}, 0) && close(right.q(), {1, 5, 6}, 0),
          b + ": E x point: Q padded, centers stacked");
    const Ellipsoid left = Ellipsoid::cart_prod(column({5.0}), E);
    check(close(left.Q(), {0, 0, 0, 4}, 0) && close(left.q(), {5, 1}, 0), b + ": point x E");
    // the boxes [-1, 3] and [-1, 1]: radii 2 and 1, k = 2
    const Ellipsoid unit(Tensor({{1.0}}));
    const Ellipsoid both = E.cart_prod(unit);
    check(close(both.Q(), {8, 0, 0, 2}, 1e-12) && close(both.q(), {1, 0}, 1e-12),
          b + ": E x E: 2 diag(r^2) around the center of the box");
    // a radius of 1e-3 is widened to 5e-3, a radius of 0 stays
    const Ellipsoid thin(Tensor({{1e-6}}));
    check(close(E.cart_prod(thin).Q(), {8, 0, 0, 2 * 25e-6}, 1e-9), b + ": a thin box is widened");
    const Ellipsoid dot(Tensor::zeros({1, 1}), column({2.0}));
    check(close(E.cart_prod(dot).Q(), {4, 0, 0, 0}, 1e-12) &&
              close(E.cart_prod(dot).q(), {1, 2}, 0),
          b + ": a point has a zero radius: k = 1");
    check(close(dot.cart_prod(dot).Q(), Tensor::zeros({2, 2}), 0), b + ": two points");
    // a batch of ellipsoids with a batch of points: exact for every member
    const Ellipsoid batch(Tensor::stack({Tensor({{4.0}}), Tensor({{9.0}})}, 0),
                          Tensor::stack({column({1.0}), column({2.0})}, 0));
    const Ellipsoid with_points = batch.cart_prod(Tensor::stack({column({7.0}), column({8.0})}, 0));
    check(with_points.Q().shape() == Shape{2, 2, 2} &&
              close(with_points.q().slice(0, 1, 2), {2, 8}, 0),
          b + ": a batch");
}

/// Stated property: the product of two points, one from each ellipsoid, is in the result.
void cart_prod_property(const std::string &b) {
    cora::Rng rng(7);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 2),
                        S = test::random_ellipsoid(rng, 3, k % 2 ? 2 : -1);
        const Ellipsoid R = E.cart_prod(S);
        const Tensor x = test::ellipsoid_points(E, rng, 40, true),
                     y = test::ellipsoid_points(S, rng, 40, true);
        check(R.dim() == 5 && test::all_inside(R, Tensor::concat({x, y}, 0)),
              b + ": the product of boundary points is in the result");
        const Tensor p = test::normal_matrix(rng, 3, 1);
        const Ellipsoid exact = E.cart_prod(p);
        check(test::all_inside(exact, Tensor::concat({x, p.broadcast_to({3, 40})}, 0)) &&
                  !exact.is_full_dim(),
              b + ": with a point the product is exact (degenerate)");
    }
}

/// The unsupported modes, a point of the wrong shape, an empty ellipsoid and a batch mismatch are
/// refused; with the checks off the call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0}}), column({1.0}));
    check(test::raises([&] { E.cart_prod(E, "inner"); }, Error::not_supported),
          "cart_prod: inner is not supported");
    check(test::raises([&] { E.cart_prod(E, "exact"); }, Error::not_supported),
          "cart_prod: exact is not supported");
    check(test::raises([&] { E.cart_prod(E, "sloppy"); }, Error::wrong_value),
          "cart_prod: an unknown mode");
    check(test::raises([&] { E.cart_prod(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "cart_prod: the point is a column");
    check(test::raises([&] { Ellipsoid::cart_prod(Tensor({{1.0, 2.0}}), E); },
                       Error::dimension_mismatch),
          "cart_prod: the first point is a column");
    check(test::raises([&] { E.cart_prod(Ellipsoid::empty(1)); }, Error::empty_set),
          "cart_prod: the empty ellipsoid");
    check(
        test::raises(
            [&] {
                Ellipsoid(Tensor::zeros({2, 1, 1})).cart_prod(Ellipsoid(Tensor::zeros({3, 1, 1})));
            },
            Error::dimension_mismatch),
        "cart_prod: one batch shape");
    const Ellipsoid on = E.cart_prod(E);
    ChecksScope off(false);
    check(close(E.cart_prod(E).Q(), on.Q(), 0), "cart_prod: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cart_prod_known(b);
        cart_prod_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid cart_prod");
}
