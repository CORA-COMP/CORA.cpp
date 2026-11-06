// test_interval_enclose_points - interval enclose_points: the smallest box around points
//
// CORA: contSet/@interval/enclosePoints.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void enclose_points_takes_min_and_max(const std::string &b) {
    const Tensor P = Tensor::from_data({1, -2, 3, 0, 5, -1, 4, 4}, {2, 4});
    const Interval I = Interval::enclose_points(P);
    check(close(I.inf(), std::vector<double>{-2, -1}) && close(I.sup(), std::vector<double>{3, 5}),
          b + ": the extremes of the rows");
    check(I.inf().shape() == std::vector<int64_t>({2, 1}), b + ": a box of columns");
    check(I.contains(P.slice(-1, 2, 3)), b + ": a point is inside");
    // a batch of two point clouds
    const Tensor B = Tensor::concat({P.reshape({1, 2, 4}), (P * 2.0).reshape({1, 2, 4})}, 0);
    const Interval J = Interval::enclose_points(B);
    check(J.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(J.sup(), std::vector<double>{3, 5, 6, 10}),
          b + ": a batch member by member");
    check(close(Interval::enclose_points(P.slice(-1, 0, 1)).sup(), P.slice(-1, 0, 1)),
          b + ": one point is a point box");
}

/// No point is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor P = Tensor::from_data({1, -2, 3, 0}, {2, 2});
    check(
        test::raises([&] { Interval::enclose_points(Tensor::zeros({2, 0})); }, Error::wrong_value),
        "enclose_points: no points");
    const Tensor on = Interval::enclose_points(P).sup();
    ChecksScope off(false);
    check(close(Interval::enclose_points(P).sup(), on, 0),
          "enclose_points: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(enclose_points_takes_min_and_max);
    checks_and_their_switch();
    return test::finish("interval enclose_points");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
