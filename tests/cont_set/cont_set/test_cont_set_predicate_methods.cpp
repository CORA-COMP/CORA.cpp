// test_cont_set_predicate_methods - contains and is_intersecting take a method on every class
//
// CORA.cpp only: CORA's classes check the type argument in their own files.
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/polytope/polytope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The method "exact" and "approx" answer as without it; another method is refused.
// one predicate call by the two methods
template <class F>
void accepted_and_refused(const std::string &what, F call) {
    check(call("exact") == call("approx"), what + ": exact and approx agree");
    check(test::raises([&] { call("fast"); }, Error::wrong_value), what + ": another method");
}

void predicates(const std::string &b) {
    const Interval I(vec({0.0, 0.0}), vec({1.0, 1.0}));
    const Capsule C(vec({0.0, 0.0}), vec({1.0, 0.0}), 1.0);
    const Ellipsoid E(vec({0.0, 0.0}), Tensor::eye(2));
    const Tensor p = vec({0.5, 0.5});
    // the next step of the same case
    accepted_and_refused(b + ": Capsule::contains(p)",
                         [&](const char *m) { return static_cast<bool>(C.contains(p, m)); });
    accepted_and_refused(b + ": Capsule::contains(Interval)",
                         [&](const char *m) { return C.contains(I, m); });
    accepted_and_refused(b + ": Capsule::is_intersecting(Capsule)",
                         [&](const char *m) { return C.is_intersecting(C, m); });
    accepted_and_refused(b + ": Interval::is_intersecting(Interval)",
                         [&](const char *m) { return static_cast<bool>(I.is_intersecting(I, m)); });
    accepted_and_refused(b + ": Interval::is_intersecting(p)",
                         [&](const char *m) { return static_cast<bool>(I.is_intersecting(p, m)); });
    // the next step of the same case
    accepted_and_refused(b + ": Ellipsoid::is_intersecting(p)",
                         [&](const char *m) { return static_cast<bool>(E.is_intersecting(p, m)); });
    accepted_and_refused(b + ": EmptySet::contains", [&](const char *m) {
        return EmptySet::empty(2).contains(EmptySet::empty(2), m);
    });
    accepted_and_refused(b + ": Fullspace::contains", [&](const char *m) {
        return Fullspace::inf(2).contains(static_cast<const ContSet &>(I), m);
    });
    accepted_and_refused(b + ": Fullspace::is_intersecting", [&](const char *m) {
        return Fullspace::inf(2).is_intersecting(static_cast<const ContSet &>(I), m);
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { predicates(b); });
    return test::finish("predicate methods");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
