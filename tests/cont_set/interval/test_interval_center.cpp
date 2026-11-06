// test_interval_center - interval center, rad and dim, for boxes and for interval matrices
//
// CORA: contSet/@interval/interval.m
// CORA: contSet/@interval/center.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void basics(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({3.0, 0.5, 2.0}));
    check(I.dim() == 3, b + ": dim");
    check(close(I.center(), std::vector<double>{1.0, 0.25, 2.0}), b + ": center");
    check(close(I.rad(), std::vector<double>{2.0, 0.25, 0.0}), b + ": radius");
    check(close(I.interval().inf(), I.inf()) && close(I.interval().sup(), I.sup()),
          b + ": its own hull");
}

/// As in CORA an interval can hold matrices; centre and radius are then elementwise.
void bounds_can_be_matrices(const std::string &b) {
    const Interval M(Tensor({{-1.0, 0.0}, {2.0, -3.0}}), Tensor({{1.0, 4.0}, {2.0, 0.0}}));
    check(close(M.center(), std::vector<double>{0.0, 2.0, 2.0, -1.5}), b + ": matrix centre");
    check(close(M.rad(), std::vector<double>{1.0, 2.0, 0.0, 1.5}), b + ": matrix radius");
    check(M.center().shape() == std::vector<int64_t>({2, 2}), b + ": matrix shape");
}

/// An infinite center is NaN, the whole space has the origin, an empty box has no center, and a
/// batch is handled member by member.
void center_of_unbounded_and_empty_boxes(const std::string &b) {
    const double kInf = std::numeric_limits<double>::infinity();
    const Interval half(column({0.0, -kInf}), column({kInf, 2.0}));
    const std::vector<double> c = half.center().data();
    check(std::isnan(c[0]) && std::isnan(c[1]), b + ": infinite centers are NaN");
    const Interval all = Interval::unbounded(2);
    check(close(all.center(), std::vector<double>{0.0, 0.0}), b + ": the whole space: the origin");
    const Tensor none = all.inf().zeros_like({2, 0});
    check(Interval(none, none).center().shape() == std::vector<int64_t>({2, 0}),
          b + ": an empty box: (n, 0)");
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    const std::vector<double> batch = Interval::stack({I, all}).center().data();
    check(close(batch[0], 1.0) && close(batch[1], 2.0) && batch[2] == 0.0 && batch[3] == 0.0,
          b + ": a batch of a box and the whole space");
}

/// center has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({3.0, 0.5}));
    const Tensor on = I.center();
    ChecksScope off(false);
    check(close(I.center(), on, 0), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        basics(b);
        bounds_can_be_matrices(b);
        center_of_unbounded_and_empty_boxes(b);
    });
    checks_and_their_switch();
    return test::finish("interval center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
