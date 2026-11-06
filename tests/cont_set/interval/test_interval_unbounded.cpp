// test_interval_unbounded - interval unbounded: the box of all vectors, CORA's interval.Inf
//
// CORA: contSet/@interval/Inf.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void unbounded_has_infinite_bounds(const std::string &b) {
    const double kInf = std::numeric_limits<double>::infinity();
    const Interval I = Interval::unbounded(3);
    check(I.dim() == 3 && I.inf().shape() == std::vector<int64_t>({3, 1}), b + ": a column of 3");
    check(I.inf().data() == std::vector<double>(3, -kInf) &&
              I.sup().data() == std::vector<double>(3, kInf),
          b + ": (-Inf, Inf) in every coordinate");
    check(I.contains(test::column({1e300, -1e300, 0})), b + ": it contains every point");
    check(Interval::unbounded(0).dim() == 0, b + ": dimension 0, CORA's default");
}

/// A negative dimension is refused; with the checks off the box is the same.
void checks_and_their_switch() {
    check(test::raises([] { Interval::unbounded(-1); }, Error::wrong_value),
          "unbounded: a negative dimension");
    ChecksScope off(false);
    check(Interval::unbounded(2).sup().data() ==
              std::vector<double>(2, std::numeric_limits<double>::infinity()),
          "unbounded: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { unbounded_has_infinite_bounds(b); });
    checks_and_their_switch();
    return test::finish("interval unbounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
