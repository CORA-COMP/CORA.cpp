// test_cont_set_is_intersecting - is_intersecting of two sets: tol and the dimensions are checked
// dimensions are checked
//
// CORA: contSet/@contSet/isIntersecting.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;

namespace {

/// A negative tol and different dimensions are refused; a pair without an entry is noops.
void the_tolerance_is_checked() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { is_intersecting(Z, I, -1.0); }, Error::wrong_value), "tol -1");
    check(test::raises([&] { is_intersecting(Z, Z3); }, Error::dimension_mismatch),
          "dimensions 2 and 3");
    const test::UnregisteredSet U;
    check(test::raises([&] { is_intersecting(Z, U); }, Error::noops), "no pair: noops");
    check(is_intersecting(Z, I), "zonotope and box: the pair is registered");
    ChecksScope off(false);
    check(test::raises([&] { is_intersecting(Z, U, -1.0); }, Error::noops),
          "the checks off skip tol");
}

} // namespace

int main() {
    the_tolerance_is_checked();
    return test::finish("cont_set is_intersecting");
}
