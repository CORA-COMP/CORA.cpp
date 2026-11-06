// test_cont_set_interval - ContSet::interval: the box of any set
//
// CORA: contSet/@contSet/interval.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box of a zonotope; interval takes no input to check, and the checks off change nothing.
void the_box_of_a_set() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const ContSet &S = Z;
    check(test::close(S.interval().sup(), std::vector<double>{1.5, 0.5}), "the upper bound");
    ChecksScope off(false);
    check(test::close(S.interval().inf(), std::vector<double>{0.5, -0.5}), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    the_box_of_a_set();
    return test::finish("cont_set interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
