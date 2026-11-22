// test_interval_masks - the predicates of a batch of boxes answer one flag per member
//
// CORA.cpp only: the batch of sets is CORA.cpp's, CORA's predicates answer one logical.
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The boxes [0, 1]^2 and [0, 2]^2 as one batch.
Interval two_boxes() {
    return Interval::stack(
        {Interval(vec({0.0, 0.0}), vec({1.0, 1.0})), Interval(vec({0.0, 0.0}), vec({2.0, 2.0}))});
}

void contains_and_intersecting(const std::string &b) {
    const Interval I = two_boxes();
    const Mask in = I.contains(vec({1.5, 1.5}));
    check(test::close(in.tensor(), {0.0, 1.0}), b + ": contains, one flag per member");
    check(!in.all() && in.any() && !static_cast<bool>(in), b + ": all, any and the bool of a mask");
    // the points of a batch meet the boxes member by member
    const Tensor points({{0.5, 0.5}, {3.0, 3.0}});
    check(test::close(I.contains(points).tensor(), {1.0, 0.0}), b + ": a batch of points");
    check(test::close(I.is_intersecting(vec({1.5, 1.5})).tensor(), {0.0, 1.0}),
          b + ": is_intersecting with a point");
    const Interval far(vec({1.5, 1.5}), vec({3.0, 3.0}));
    check(test::close(I.is_intersecting(far).tensor(), {0.0, 1.0}),
          b + ": is_intersecting with a box");
    // one set: a 0-d mask that is the old bool
    const Interval one(vec({0.0, 0.0}), vec({1.0, 1.0}));
    check(one.contains(vec({0.5, 0.5})).tensor().ndim() == 0, b + ": one set, a 0-d mask");
    check(one.contains(vec({0.5, 0.5})) && !one.contains(vec({2.0, 0.5})), b + ": one set, bool");
}

void equality_and_shape_predicates(const std::string &b) {
    const Interval I = two_boxes();
    check(test::close(I.isequal(I).tensor(), {1.0, 1.0}), b + ": isequal itself");
    const Interval other = Interval::stack(
        {Interval(vec({0.0, 0.0}), vec({1.0, 1.0})), Interval(vec({0.0, 0.0}), vec({3.0, 3.0}))});
    check(test::close(I.isequal(other).tensor(), {1.0, 0.0}), b + ": isequal member by member");
    const Interval flat = Interval::stack(
        {Interval(vec({0.0, 0.0}), vec({1.0, 0.0})), Interval(vec({0.0, 0.0}), vec({1.0, 1.0}))});
    check(test::close(flat.is_full_dim().tensor(), {0.0, 1.0}), b + ": is_full_dim");
    const double inf = std::numeric_limits<double>::infinity();
    const Interval wide = Interval::stack(
        {Interval(vec({0.0, 0.0}), vec({1.0, inf})), Interval(vec({0.0, 0.0}), vec({1.0, 1.0}))});
    check(test::close(wide.is_bounded().tensor(), {0.0, 1.0}), b + ": is_bounded");
}

/// An empty member is bounded, equal to another empty one, and in nothing.
void an_empty_member(const std::string &b) {
    const Interval I =
        Interval::stack({Interval(vec({0.0, 0.0}), vec({1.0, 1.0})), Interval::empty(2)});
    check(test::close(I.contains(vec({0.5, 0.5})).tensor(), {1.0, 0.0}), b + ": contains");
    check(test::close(I.is_bounded().tensor(), {1.0, 1.0}), b + ": is_bounded");
    check(test::close(I.is_full_dim().tensor(), {1.0, 0.0}), b + ": is_full_dim");
    check(test::close(I.isequal(I).tensor(), {1.0, 1.0}), b + ": isequal itself");
    check(test::close(I.is_intersecting(I).tensor(), {1.0, 0.0}), b + ": is_intersecting itself");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        contains_and_intersecting(b);
        equality_and_shape_predicates(b);
        an_empty_member(b);
    });
    return test::finish("interval masks");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
