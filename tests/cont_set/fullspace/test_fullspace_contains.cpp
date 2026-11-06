// test_fullspace_contains - fullspace contains: the whole space contains every set
//
// CORA: contSet/@fullspace/contains_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void contains_every_set(const std::string &b) {
    const Fullspace F(2);
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    check(F.contains(Z) && F.contains(Fullspace(2)) && F.contains(Interval::unbounded(2)),
          b + ": a zonotope, the space and the unbounded box");
    check(F.contains(Interval(column({kInf, 0.0}), column({kInf, 1.0}))), b + ": a box at Inf");
    check(F.contains(Interval::empty(2)), b + ": the empty interval");
    check(F.contains(Z, 1e-3) && F.contains(Z, 0.0), b + ": any tol");
    check(contains(F, Z) && contains(F, Fullspace(2)), b + ": through the registry");
}

/// R^0, another dimension and a negative tol are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([] { Fullspace(0).contains(Fullspace(0)); }, Error::not_supported), "R^0");
    check(test::raises([&] { F.contains(Fullspace(3)); }, Error::dimension_mismatch), "dimension");
    check(test::raises([&] { F.contains(Fullspace(2), -1.0); }, Error::wrong_value),
          "negative tol");
    ChecksScope off(false);
    check(F.contains(Fullspace(2)), "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(contains_every_set);
    checks_and_their_switch();
    return test::finish("fullspace contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
