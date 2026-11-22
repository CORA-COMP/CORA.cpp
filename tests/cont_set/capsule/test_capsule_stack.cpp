// test_capsule_stack - Capsule::stack: a batch of capsules, with its empty members
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The parts stack along a new first axis, and each member keeps its emptiness.
void stack_known(const std::string &b) {
    const Capsule A(vec({1.0, 2.0}), vec({3.0, 4.0}), 0.5),
        B(vec({0.0, 1.0}), vec({1.0, 0.0}), 2.0);
    const Capsule S = Capsule::stack({A, B});
    check(S.c().shape() == Shape{2, 2} && S.r().shape() == Shape{2}, b + ": the batch shape (2)");
    check(close(S.c(), {1, 2, 0, 1}, 0) && close(S.generator(), {3, 4, 1, 0}, 0) &&
              close(S.r(), {0.5, 2.0}, 0),
          b + ": the members in order");
    check(!S.empty_mask().any(), b + ": no member is empty");
    const Capsule mixed = Capsule::stack({A, Capsule::empty(2)});
    const std::vector<uint8_t> flags = mixed.empty_mask().flags(Shape{2});
    check(flags == std::vector<uint8_t>{0, 1} && !mixed.is_empty(), b + ": the second is empty");
    check(!mixed.contains(vec({1.0, 2.0})).all(), b + ": the empty member contains no point");
    check(Capsule::stack({Capsule::empty(2), Capsule::empty(2)}).is_empty(), b + ": all empty");
}

/// No capsules, or capsules of different shapes, are refused; with the checks off a valid call
/// gives the same batch.
void checks_and_their_switch() {
    const Capsule A(vec({1.0, 2.0}), vec({3.0, 4.0}), 0.5);
    check(test::raises([] { Capsule::stack({}); }, Error::wrong_value), "stack: no capsules");
    check(
        test::raises([&] { Capsule::stack({A, Capsule(vec({1.0}))}); }, Error::dimension_mismatch),
        "stack: one shape");
    const Tensor on = Capsule::stack({A, A}).c();
    ChecksScope off(false);
    check(close(Capsule::stack({A, A}).c(), on, 0), "stack: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { stack_known(b); });
    checks_and_their_switch();
    return test::finish("capsule stack");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
