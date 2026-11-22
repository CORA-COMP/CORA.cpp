// test_interval_mink_diff - interval mink_diff: the box of x with x + S in I
//
// CORA: contSet/@interval/minkDiff.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void mink_diff_of_boxes_and_points(const std::string &b) {
    const Interval I(vec({0.0, 0.0}), vec({10.0, 4.0}));
    const Interval R = I.mink_diff(Interval(vec({1.0, -1.0}), vec({3.0, 1.0})));
    check(close(R.inf(), std::vector<double>{-1, 1}) && close(R.sup(), std::vector<double>{7, 3}),
          b + ": [inf - inf_S, sup - sup_S]");
    check(!R.is_empty(), b + ": nonempty");
    const Interval P = I.mink_diff(vec({1.0, 2.0}));
    check(close(P.inf(), std::vector<double>{-1, -2}) && close(P.sup(), std::vector<double>{9, 2}),
          b + ": a point shifts the box by -p");
    const Interval F = I.mink_diff(Interval(vec({-1.0, 0.0}), vec({1.0, 0.0})));
    check(close(F.inf(), std::vector<double>{1, 0}) && close(F.sup(), std::vector<double>{9, 4}),
          b + ": a flat box shrinks one coordinate");
}

void mink_diff_can_be_empty(const std::string &b) {
    const Interval I(vec({0.0, 0.0}), vec({1.0, 4.0}));
    const Interval E = I.mink_diff(Interval(vec({-1.0, 0.0}), vec({1.0, 1.0})));
    check(E.is_empty() && E.dim() == 2, b + ": a bound would cross: the empty interval");
    const Interval B = Interval::stack({I, I});
    // a batch
    check(B.mink_diff(Interval::stack({Interval(vec({-1.0, 0.0}), vec({1.0, 1.0})),
                                       Interval(vec({-1.0, 0.0}), vec({1.0, 1.0}))}))
              .is_empty(),
          b + ": a batch of empty differences");
}

void mink_diff_of_another_set(const std::string &b) {
    const Interval I(vec({0.0, 0.0}), vec({4.0, 4.0}));
    const Zonotope Z(vec({0.0, 0.0}), Tensor::from_data({0.5, 0, 0, 0.5}, {2, 2}));
    const Interval R = I.mink_diff(Z);
    check(close(R.inf(), std::vector<double>{0.5, 0.5}) &&
              close(R.sup(), std::vector<double>{3.5, 3.5}),
          b + ": the support function bounds each coordinate");
    const Interval Q = I.mink_diff(Z, std::string("inner"));
    check(close(Q.inf(), R.inf()) && close(Q.sup(), R.sup()), b + ": type inner is the default");
    const Zonotope W(vec({0.0, 0.0}), Tensor::from_data({3, 0, 0, 3}, {2, 2}));
    check(I.mink_diff(W).is_empty(), b + ": a set that does not fit");
}

/// A wrong dimension, an unknown type, and "exact" for a set are refused; the checks off give the
/// same.
void checks_and_their_switch() {
    const Interval I(vec({0.0, 0.0}), vec({4.0, 4.0}));
    const Zonotope Z(vec({0.0, 0.0}), Tensor::from_data({0.5, 0, 0, 0.5}, {2, 2}));
    check(test::raises([&] { I.mink_diff(Interval(vec({0.0}), vec({1.0}))); },
                       Error::dimension_mismatch),
          "mink_diff: a box of dimension 1");
    check(test::raises([&] { I.mink_diff(vec({1.0})); }, Error::dimension_mismatch),
          "mink_diff: a point of dimension 1");
    check(test::raises([&] { I.mink_diff(Z, std::string("exact")); }, Error::no_exact_alg),
          "mink_diff: no exact algorithm for a zonotope");
    check(test::raises([&] { I.mink_diff(Z, std::string("outer")); }, Error::wrong_value),
          "mink_diff: an unknown type");
    const Tensor on = I.mink_diff(Z).sup();
    ChecksScope off(false);
    check(close(I.mink_diff(Z).sup(), on, 0), "mink_diff: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        mink_diff_of_boxes_and_points(b);
        mink_diff_can_be_empty(b);
        mink_diff_of_another_set(b);
    });
    checks_and_their_switch();
    return test::finish("interval mink_diff");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
