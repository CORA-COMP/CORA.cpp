// test_interval_is_full_dim - interval is_full_dim: whether a box has a nonempty interior
//
// CORA: contSet/@interval/isFullDim.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

void is_full_dim_looks_for_flat_coordinates(const std::string &b) {
    check(Interval(column({0.0, 1.0}), column({1.0, 3.0})).is_full_dim(), b + ": a box");
    check(!Interval(column({0.0, 1.0}), column({1.0, 1.0})).is_full_dim(),
          b + ": a zero-width coordinate");
    check(!Interval(column({0.0, 1.0}), column({1.0, 1.0 + 1e-9})).is_full_dim(),
          b + ": a width within 1e-8 counts as zero, as CORA's withinTol");
    check(Interval(column({0.0, 1.0}), column({1.0, 1.0 + 1e-6})).is_full_dim(),
          b + ": a width of 1e-6 does not");
    check(Interval::unbounded(2).is_full_dim(), b + ": the whole space");
    check(!Interval::empty(2).is_full_dim(), b + ": the empty interval");
    const Interval full(column({0.0}), column({1.0})), flat(column({1.0}), column({1.0}));
    check(Interval::stack({full, full}).is_full_dim() && !Interval::stack({full, flat}).is_full_dim(),
          b + ": a batch is full-dimensional only if every member is");
}

/// is_full_dim has no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 3.0}));
    ChecksScope off(false);
    check(I.is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(is_full_dim_looks_for_flat_coordinates);
    checks_and_their_switch();
    return test::finish("interval is_full_dim");
}
