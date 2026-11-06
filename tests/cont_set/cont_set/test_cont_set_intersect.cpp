// test_cont_set_intersect - the intersection of two sets (CORA's and): a pair without an entry
//
// raises noops
//
// CORA: contSet/@contSet/and.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
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

/// A pair without an entry is noops, with the checks on and off; the dimensions are checked.
void an_unknown_pair_is_noops() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const test::UnregisteredSet U;
    check(test::raises([&] { intersect(Z, Z3); }, Error::dimension_mismatch), "dimensions 2 and 3");
    check(test::raises([&] { intersect(Z, U); }, Error::noops), "zonotope and unregistered set");
    ChecksScope off(false);
    check(test::raises([&] { intersect(U, Z); }, Error::noops), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    an_unknown_pair_is_noops();
    return test::finish("cont_set intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
