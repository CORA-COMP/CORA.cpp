// test_cont_set_contains - contains of two sets with a tolerance: tol and the dimensions are
// checked

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// A negative tol and different dimensions are refused; no pair implements it yet.
void the_tolerance_is_checked() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { contains(Z, I, -1.0); }, Error::wrong_value), "tol -1");
    check(test::raises([&] { contains(Z, Z3); }, Error::dimension_mismatch), "dimensions 2 and 3");
    check(test::raises([&] { contains(Z, I); }, Error::noops), "no pair: noops");
    ChecksScope off(false);
    check(test::raises([&] { contains(Z, I, -1.0); }, Error::noops), "the checks off skip tol");
}

} // namespace

int main() {
    the_tolerance_is_checked();
    return test::finish("cont_set contains");
}
