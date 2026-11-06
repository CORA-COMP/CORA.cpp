// test_cont_set_compact - ContSet::compact: a set without redundancy, through the class's compact_
//
// CORA: contSet/@contSet/compact.m
// CORA: contSet/@contSet/compact_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A zonotope drops its zero generators by default, merges the parallel ones with "all" and keeps
/// them with "aligned"; the entry gives the same sets as the class's own compact.
void compact_of_a_zonotope(const std::string &b) {
    // the generators (1, 0), (0, 0), (2, 0) and (0, 1) as columns
    const Zonotope Z(column({0.0, 0.0}),
                     Tensor::from_data({1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0}, {2, 4}));
    const ContSet &S = Z;
    const Zonotope zeros = S.compact().as<Zonotope>(), all = S.compact("all").as<Zonotope>(),
                   aligned = S.compact("aligned").as<Zonotope>();
    check(zeros.generators().shape() == std::vector<int64_t>({2, 3}), b + ": zeros dropped");
    check(test::close(zeros.generators(), Z.compact("zeros").generators(), 0), b + ": the default");
    check(all.generators().shape() == std::vector<int64_t>({2, 2}), b + ": parallel merged");
    check(test::close(all.generators(), Z.compact("all").generators(), 0),
          b + ": all, as the class");
    check(test::close(aligned.generators(), Z.generators(), 0), b + ": aligned keeps them");
    // tol: a generator of length 0.5 is zero for tol 0.6
    const Zonotope small(column({0.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 1.0}, {2, 2}));
    check(static_cast<const ContSet &>(small)
                  .compact("zeros", 0.6)
                  .as<Zonotope>()
                  .generators()
                  .size(-1) == 1,
          b + ": tol 0.6");
}

/// A box, the empty set and the whole space are returned as they are, whatever the arguments.
void compact_of_the_others() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    check(
        test::close(static_cast<const ContSet &>(I).compact("anything", -1.0).as<Interval>().sup(),
                    I.sup(), 0),
        "a box");
    check(static_cast<const ContSet &>(EmptySet(2)).compact().is<EmptySet>(), "the empty set");
    check(static_cast<const ContSet &>(Fullspace(2)).compact().is<Fullspace>(), "the whole space");
}

/// An unknown method and a negative tol are refused, a batch with "all" is not supported, a
/// class without compact_ raises noops; the checks off skip the checks.
void errors_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const ContSet &S = Z;
    check(test::raises([&] { S.compact("nonsense"); }, Error::wrong_value), "an unknown method");
    check(test::raises([&] { S.compact("zeros", -1.0); }, Error::wrong_value), "a negative tol");
    check(test::raises([] { test::UnregisteredSet().compact(); }, Error::noops), "noops");
    ChecksScope off(false);
    check(S.compact("zeros", 0.0).is<Zonotope>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(compact_of_a_zonotope);
    compact_of_the_others();
    errors_and_their_switch();
    return test::finish("cont_set compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
