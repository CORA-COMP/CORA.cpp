// test_cont_set_contains - contains of two sets with a tolerance: tol and the dimensions are
//
// checked
//
// CORA: contSet/@contSet/contains.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A negative tol and different dimensions are refused; a pair without an entry is noops.
void the_tolerance_is_checked() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}}));
    const Interval I = Interval(test::vec({0, -1}), test::vec({1, 1}));
    check(test::raises([&] { contains(Z, I, -1.0); }, Error::wrong_value), "tol -1");
    check(test::raises([&] { contains(Z, Z3); }, Error::dimension_mismatch), "dimensions 2 and 3");
    const test::UnregisteredSet U;
    check(test::raises([&] { contains(Z, U); }, Error::noops), "no pair: noops");
    ChecksScope off(false);
    check(test::raises([&] { contains(Z, U, -1.0); }, Error::noops), "the checks off skip tol");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    the_tolerance_is_checked();
    return test::finish("cont_set contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
