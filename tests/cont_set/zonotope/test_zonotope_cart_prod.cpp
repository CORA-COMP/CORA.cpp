// test_zonotope_cart_prod - zonotope cart_prod: the product with a zonotope, a box or a point
//
// CORA: contSet/@zonotope/cartProd_.m
// PROPERTY: contSet/@zonotope/cartProd_.m

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Centers stack, generators sit on the block diagonal; a point and a box are zonotopes first.
void cart_prod_of_known_sets(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 2.0}, {3.0, 4.0}}));
    const Zonotope Z2(column({5.0}), Tensor({{6.0, 7.0, 8.0}}));
    const Zonotope P = Z.cart_prod(Z2);
    check(close(P.center(), std::vector<double>{1, 2, 5}, 0), b + ": centers stack");
    check(close(P.generators(), std::vector<double>{1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 6, 7, 8}, 0),
          b + ": block diagonal generators");

    const Zonotope with_point = Z.cart_prod(column({9.0}));
    check(close(with_point.generators(), std::vector<double>{1, 2, 3, 4, 0, 0}, 0),
          b + ": a point adds a zero row");
    const Zonotope point_first = Zonotope::cart_prod(column({9.0}), Z);
    check(close(point_first.center(), std::vector<double>{9, 1, 2}, 0), b + ": a point first");
    check(close(point_first.generators(), std::vector<double>{0, 0, 1, 2, 3, 4}, 0),
          b + ": a point first: zero row on top");

    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    const Zonotope with_box = Z.cart_prod(I);
    check(close(with_box.center(), std::vector<double>{1, 2, 1, 0}, 0), b + ": a box: its center");
    check(with_box.generators().shape() == std::vector<int64_t>({4, 4}), b + ": a box: two axes");

    const Set via_sets = cart_prod(Z, I);
    check(via_sets.is<Zonotope>() && via_sets.as<Zonotope>().dim() == 4, b + ": through the registry");
}

/// A batch of products, member by member.
void cart_prod_of_batches(const std::string &b) {
    cora::Rng rng(15);
    const Zonotope A1 = Zonotope::generate_random(rng, 2, std::nullopt, 3), A2 = Zonotope::generate_random(rng, 2, std::nullopt, 3);
    const Zonotope B1 = Zonotope::generate_random(rng, 3, std::nullopt, 2), B2 = Zonotope::generate_random(rng, 3, std::nullopt, 2);
    const Zonotope P = Zonotope::stack({A1, A2}).cart_prod(Zonotope::stack({B1, B2}));
    check(P.generators().shape() == std::vector<int64_t>({2, 5, 5}), b + ": batch shape");
    check(close(P.generators().slice(0, 1, 2).squeeze(0), A2.cart_prod(B2).generators(), 0),
          b + ": member 2");
}

/// Stated property: the support function of the product along (d1; d2) is the sum of the parts.
void support_function_adds(const std::string &b) {
    cora::Rng rng(16);
    const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 4), Z2 = Zonotope::generate_random(rng, 3, std::nullopt, 5);
    const Zonotope P = Z.cart_prod(Z2);
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 5);
        const std::vector<double> d1(d.begin(), d.begin() + 2), d2(d.begin() + 2, d.end());
        check(close(test::support(P, d), test::support(Z, d1) + test::support(Z2, d2), 1e-9),
              b + ": support function of the product");
    }
}

/// Wrong operands are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {3.0}}));
    check(test::raises([&] { Z.cart_prod(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "cart_prod: a point must be a column");
    check(test::raises([&] { Z.cart_prod(Zonotope::stack({Z, Z})); }, Error::dimension_mismatch),
          "cart_prod: a batch and a single zonotope");
    const Tensor on = Z.cart_prod(Z).generators();
    ChecksScope off(false);
    check(test::close(Z.cart_prod(Z).generators(), on, 0), "cart_prod: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cart_prod_of_known_sets(b);
        cart_prod_of_batches(b);
        support_function_adds(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope cart_prod");
}
