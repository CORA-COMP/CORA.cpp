// test_zonotope_isequal - zonotope isequal: equality of sets up to a tolerance
//
// CORA: contSet/@zonotope/isequal.m
// PROPERTY: contSet/@zonotope/isequal.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// Hand-computed: order and sign of generators do not matter, parallel ones merge, centers differ.
void isequal_known(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(Z.isequal(Z), b + ": a zonotope equals itself");
    check(Z.isequal(Zonotope(column({1.0, 2.0}), Tensor({{0.0, -1.0}, {-3.0, 0.0}}))),
          b + ": generators in another order and with another sign");
    check(!Z.isequal(Zonotope(column({1.0, 2.5}), Z.generators())), b + ": another center");
    check(!Z.isequal(Zonotope(Z.center(), Tensor({{1.0, 0.0}, {0.0, 2.0}}))),
          b + ": another generator");
    check(Zonotope(column({0.0}), Tensor({{1.0, 1.0}}))
              .isequal(Zonotope(column({0.0}), Tensor({{2.0}}))),
          b + ": [1, 1] and [2] generate the same interval");
    check(Z.isequal(
              Zonotope(Z.center(), Tensor::concat({Z.generators(), Tensor::zeros({2, 1})}, -1))),
          b + ": a zero generator does not matter");
    check(!Z.isequal(Zonotope(column({1.0}), Tensor({{1.0}}))), b + ": another dimension is false");
    check(Z.isequal(Zonotope(column({1.0 + 1e-9, 2.0}), Z.generators()), 1e-6), b + ": within tol");
    check(!Z.isequal(Zonotope(column({1.0 + 1e-9, 2.0}), Z.generators())), b + ": not within eps");
}

/// A box and a point are compared as zonotopes.
void isequal_other_sets(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({2.0, 1.5}));
    const Zonotope box(column({1.0, 1.25}), Tensor({{1.0, 0.0}, {0.0, 0.25}}));
    check(box.isequal(I), b + ": a box");
    check(!box.isequal(Interval(column({0.0, 1.0}), column({2.0, 2.0}))), b + ": another box");
    const Zonotope point(column({1.0, 2.0}), Tensor::zeros({2, 0}));
    check(point.isequal(column({1.0, 2.0})), b + ": a point");
    check(!point.isequal(column({1.0, 2.5})), b + ": another point");
    check(!box.isequal(column({1.0, 1.25})), b + ": a box is not a point");
}

/// Stated property: isequal is reflexive and symmetric, and a permuted copy of random
/// generators equals the zonotope.
void isequal_property(const std::string &b) {
    cora::Rng rng(30);
    for (int k = 0; k < 10; ++k) {
        const Zonotope Z = Zonotope::generate_random(3, 5, rng);
        const Tensor flipped =
            Z.generators().flip(-1).multiply(Tensor::like(Z.center(), {-1, 1, -1, 1, 1}, {1, 5}));
        const Zonotope W(Z.center(), flipped);
        check(Z.isequal(Z) && Z.isequal(W) && W.isequal(Z), b + ": a reordered, re-signed copy");
        const Zonotope V(Z.center() + column({1e-3, 0.0, 0.0}), Z.generators());
        check(!Z.isequal(V) && !V.isequal(Z), b + ": a shifted center");
    }
}

/// A batch and a negative tol are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).isequal(Z); }, Error::not_supported),
          "isequal: a batch");
    check(test::raises([&] { Z.isequal(Z, -1.0); }, Error::wrong_value), "isequal: tol < 0");
    ChecksScope off(false);
    check(Z.isequal(Z), "isequal: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        isequal_known(b);
        isequal_other_sets(b);
        isequal_property(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope isequal");
}
