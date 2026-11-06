// test_capsule_lift - Capsule lift: the coordinates permuted
//
// CORA: contSet/@capsule/lift_.m
// CORA: contSet/@contSet/lift.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// State i goes to position dims[i]; the radius is kept; the identity changes nothing.
void lift_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0, 3.0}), column({4.0, 5.0, 6.0}), 0.5);
    const Capsule L = C.lift(3, {1, 2, 0});
    check(close(L.c(), {3, 1, 2}, 0) && close(L.generator(), {6, 4, 5}, 0) &&
              close(L.r(), {0.5}, 0),
          b + ": coordinate i becomes coordinate dims[i]");
    check(close(C.lift(3, {0, 1, 2}).c(), C.c(), 0), b + ": the identity");
    // through the set interface, with and without the default positions
    const Set S = C;
    check(close(S->lift(3, std::vector<int64_t>{2, 1, 0}).as<Capsule>().c(), {3, 2, 1}, 0),
          b + ": ContSet::lift");
    check(close(S->lift(3).as<Capsule>().c(), C.c(), 0), b + ": the default positions");
    const Tensor c = Tensor::from_data({1, 2, 3, 4}, {2, 2, 1});
    check(close(Capsule(c, c, 0.1).lift(2, {1, 0}).c(), {2, 1, 4, 3}, 0), b + ": a batch");
}

/// Inverse permutations of lift and project differ; the lift of the lift undoes it.
void lift_property(const std::string &b) {
    Rng rng(17);
    const Capsule C = test::random_capsule(rng, 4);
    const Capsule back = C.lift(4, {2, 0, 3, 1}).lift(4, {1, 3, 0, 2});
    check(back.isequal(C, 0.0), b + ": a permutation and its inverse");
}

/// New dimensions would be unbounded (not_defined); a position used twice, out of range or too
/// few are refused; with the checks off a valid call gives the same set.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0, 3.0}), column({4.0, 5.0, 6.0}), 0.5);
    check(test::raises([&] { C.lift(4, {0, 1, 2}); }, Error::not_defined),
          "lift: more dimensions than the capsule has");
    check(test::raises([&] { C.lift(3, {0, 0, 1}); }, Error::wrong_value),
          "lift: a position used twice");
    check(test::raises([&] { C.lift(3, {0, 1, 3}); }, Error::wrong_value),
          "lift: a position out of range");
    check(test::raises([&] { C.lift(3, {0, 1}); }, Error::wrong_value), "lift: too few positions");
    const Capsule on = C.lift(3, {1, 2, 0});
    ChecksScope off(false);
    check(close(C.lift(3, {1, 2, 0}).c(), on.c(), 0), "lift: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        lift_known(b);
        lift_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
