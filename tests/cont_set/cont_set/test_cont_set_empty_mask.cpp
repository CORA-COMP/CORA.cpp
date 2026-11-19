// test_cont_set_empty_mask - EmptyMask: the flags of the empty members of a batch of sets (#163)
//
// CORA.cpp only: CORA stores an empty set in its data (zeros(n, 0), an infeasible constraint).
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       19-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_mask.h"
#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// No member, every member, or some members: the three states and their flags.
void states_and_flags() {
    const EmptyMask none, all = EmptyMask::all_members();
    check(!none.any() && !none.all() && all.any() && all.all(), "none and all");
    check(none.flags({2}) == std::vector<uint8_t>({0, 0}) &&
              all.flags({2, 1}) == std::vector<uint8_t>({1, 1}),
          "the flags of none and all at any batch shape");
    const EmptyMask some({3}, {0, 1, 0});
    check(some.any() && !some.all(), "some members");
    check(some.flags({2, 3}) == std::vector<uint8_t>({0, 1, 0, 0, 1, 0}),
          "the flags broadcast as numpy broadcasts");
    check(EmptyMask({2}, {1, 1}) == all && EmptyMask({2}, {0, 0}) == none,
          "uniform flags are the uniform masks");
}

/// The masks of a result: empty where either operand is, or where both are.
void either_and_both() {
    const EmptyMask a({2}, {1, 0}), b({2}, {0, 1}), none, all = EmptyMask::all_members();
    check(a.either(b) == all && a.both(b) == none, "either and both of two masks");
    check(a.either(none) == a && a.both(all) == a && a.either(all) == all && a.both(none) == none,
          "with the uniform masks");
    const EmptyMask column({2, 1}, {1, 0});
    check(column.either(EmptyMask({1, 2}, {0, 1})).flags({2, 2}) ==
              std::vector<uint8_t>({1, 1, 0, 1}),
          "masks of two batch shapes broadcast");
    const EmptyMask stacked = EmptyMask::stack({a, none}, {2});
    check(stacked.flags({2, 2}) == std::vector<uint8_t>({1, 0, 0, 0}), "a stack of masks");
}

/// A set keeps its mask: the empty interval has one, a box none; the shape keeps n.
void sets_carry_the_mask() {
    const Interval E = Interval::empty(3), I(vec({0.0, 1.0}), vec({1.0, 2.0}));
    check(E.empty_mask().all() && E.is_empty() && E.inf().shape() == std::vector<int64_t>({3}),
          "the empty interval keeps its dimension");
    check(!I.empty_mask().any() && !I.is_empty(), "a box is not empty");
    const Interval batch = Interval::stack({I, I.intersect(I + 5.0)});
    check(batch.empty_mask().flags(batch.batch_shape()) == std::vector<uint8_t>({0, 1}),
          "a stack of a box and an empty one");
}

/// Flags that do not match the batch, or batches that do not broadcast, are refused, with the
/// checks on and off alike.
void refusals() {
    check(test::raises([] { EmptyMask({3}, {0, 1}); }, Error::dimension_mismatch),
          "two flags for three members");
    check(test::raises([] { EmptyMask({3}, {0, 1, 0}).flags({2}); }, Error::dimension_mismatch),
          "a batch of 3 does not broadcast to 2");
    ChecksScope off(false);
    check(test::raises([] { EmptyMask({3}, {0, 1}); }, Error::dimension_mismatch),
          "the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    states_and_flags();
    either_and_both();
    sets_carry_the_mask();
    refusals();
    return test::finish("cont_set empty_mask");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
