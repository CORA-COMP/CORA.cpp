// test_cont_set_cart_prod - the Cartesian product of two sets: any dimensions; a pair without an
//
// entry raises noops
//
// CORA: contSet/@contSet/cartProd.m
// CORA: contSet/@contSet/cartProd_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
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

/// Sets of different dimensions are fine for a product; a pair without an entry is noops.
void an_unknown_pair_is_noops() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const test::UnregisteredSet U;
    check(test::raises([&] { cart_prod(Z, U); }, Error::noops), "no pair: noops");
    check(cart_prod(Z, Z3).as<Zonotope>().dim() == 5, "dimensions 2 and 3: a zonotope of 5");
    ChecksScope off(false);
    check(test::raises([&] { cart_prod(U, Z3); }, Error::noops), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    an_unknown_pair_is_noops();
    return test::finish("cont_set cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
