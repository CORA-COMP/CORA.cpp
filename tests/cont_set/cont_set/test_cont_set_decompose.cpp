// test_cont_set_decompose - ContSet::decompose: the projections onto blocks of coordinates
//
// CORA: contSet/@contSet/decompose.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A three-dimensional set splits into a block of two coordinates and one of one; a single block
/// is a projection too.
void decompose_projects_each_block(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({1.0, 2.0, 5.0}));
    const Zonotope Z(column({1.0, 2.0, 3.0}), Tensor::from_data({1, 0, 0, 1, 0, 1}, {3, 2}));
    const std::vector<Set> parts = static_cast<const ContSet &>(I).decompose({{0, 1}, {2, 2}});
    check(parts.size() == 2 && parts[0].as<Interval>().dim() == 2 &&
              test::close(parts[0].as<Interval>().sup(), {1.0, 2.0}) &&
              test::close(parts[1].as<Interval>().inf(), {2.0}),
          b + ": the blocks of a box");
    const std::vector<Set> zs = static_cast<const ContSet &>(Z).decompose({{1, 2}, {0, 0}});
    check(zs.size() == 2 && test::close(zs[0].as<Zonotope>().center(), {2.0, 3.0}) &&
              test::close(zs[1].as<Zonotope>().center(), {1.0}) &&
              zs[0].as<Zonotope>().generators().shape()[0] == 2,
          b + ": the blocks of a zonotope");
    const std::vector<Set> one = static_cast<const ContSet &>(I).decompose({{0, 2}});
    check(one.size() == 1 && one[0].as<Interval>().dim() == 3, b + ": one block");
}

/// Blocks outside the set or none are refused; the checks off skip the check.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0, 0.0}), column({1.0, 1.0, 1.0}));
    const ContSet &S = I;
    check(test::raises([&] { S.decompose({{1, 3}}); }, Error::wrong_value), "block past the end");
    check(test::raises([&] { S.decompose({{2, 1}}); }, Error::wrong_value), "first after last");
    check(test::raises([&] { S.decompose({}); }, Error::wrong_value), "no block");
    ChecksScope off(false);
    check(S.decompose({{0, 1}}).size() == 1, "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(decompose_projects_each_block);
    errors_and_their_switch();
    return test::finish("cont_set decompose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
