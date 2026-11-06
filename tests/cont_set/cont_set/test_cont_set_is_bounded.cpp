// test_cont_set_is_bounded - ContSet::is_bounded: whether a set is bounded
//
// CORA: contSet/@contSet/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Boxes and zonotopes through the base class, and a batch.
void bounded_or_not(const std::string &b) {
    const Interval box(column({-1.0, 0.0}), column({1.0, 2.0})),
        ray(column({-1.0, 0.0}), column({kInf, 2.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const ContSet &A = box, &C = ray, &D = Z;
    check(A.is_bounded() && !C.is_bounded() && D.is_bounded(), b + ": bounded or not");
    check(Interval::stack({box, box}).is_bounded() && !Interval::stack({box, ray}).is_bounded(),
          b + ": a batch is bounded only if every member is");
}

/// A class without the query raises noops, and the checks off change nothing.
void noops_and_the_switch() {
    check(test::raises([] { test::UnregisteredSet().is_bounded(); }, Error::noops), "noops");
    const Interval I(column({0.0}), column({1.0}));
    ChecksScope off(false);
    check(static_cast<const ContSet &>(I).is_bounded(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(bounded_or_not);
    noops_and_the_switch();
    return test::finish("cont_set is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
