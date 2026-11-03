// test_con_zonotope_empty - ConZonotope::empty: the set without points
//
// CORA: contSet/@conZonotope/empty.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The empty set has the dimension, no generators and a constraint no factor satisfies.
void empty_set_data() {
    const ConZonotope E = ConZonotope::empty(3);
    check(E.dim() == 3 && E.generators().shape() == Shape{3, 0}, "empty: dimension, no generators");
    check(close(E.A(), Tensor::ones({1, 0}), 0) && close(E.b(), {1}, 0),
          "empty: the constraint 0 = 1");
    check(ConZonotope::empty().dim() == 0, "empty: the dimension 0 by default");
    check(E.interval().is_empty(), "empty: its box is empty");
    check(!E.is_full_dim(), "empty: it has no interior");
    check(!E.contains(column({0.0, 0.0, 0.0})), "empty: it contains no point");
    check(E.center().shape() == Shape{3, 0}, "empty: its center has no column");
    check(Set(E).is<ConZonotope>(), "empty: a Set holds it");
}

/// The empty set stays empty through the operations that CORA's own empty mishandles.
void empty_stays_empty() {
    const ConZonotope E = ConZonotope::empty(1);
    const ConZonotope box(column({0.0}), Tensor({{1.0}}));
    check(E.intersect(box).interval().is_empty(), "empty: its intersection with a box is empty");
    check(E.cart_prod(box).interval().is_empty(), "empty: its product with a box is empty");
    check(E.convex_hull(box).interval().is_empty() == false, "empty: the hull with a box is not");
}

/// A negative dimension is refused; the same call with the checks off.
void checks_and_their_switch() {
    check(test::raises([] { ConZonotope::empty(-1); }, Error::wrong_value),
          "empty: a negative dimension");
    ChecksScope off(false);
    check(ConZonotope::empty(2).dim() == 2, "empty: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &) {
        empty_set_data();
        empty_stays_empty();
    });
    checks_and_their_switch();
    return test::finish("con_zonotope empty");
}
