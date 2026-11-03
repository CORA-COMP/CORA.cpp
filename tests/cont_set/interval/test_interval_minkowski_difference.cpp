// test_interval_minkowski_difference - interval minkowski_difference: the box of x with x + S in I
//
// CORA: contSet/@interval/minkDiff.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void minkowski_difference_of_boxes_and_points(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({10.0, 4.0}));
    const Interval R = I.minkowski_difference(Interval(column({1.0, -1.0}), column({3.0, 1.0})));
    check(close(R.inf(), std::vector<double>{-1, 1}) && close(R.sup(), std::vector<double>{7, 3}),
          b + ": [inf - inf_S, sup - sup_S]");
    check(!R.is_empty(), b + ": nonempty");
    const Interval P = I.minkowski_difference(column({1.0, 2.0}));
    check(close(P.inf(), std::vector<double>{-1, -2}) && close(P.sup(), std::vector<double>{9, 2}),
          b + ": a point shifts the box by -p");
    const Interval F = I.minkowski_difference(Interval(column({-1.0, 0.0}), column({1.0, 0.0})));
    check(close(F.inf(), std::vector<double>{1, 0}) && close(F.sup(), std::vector<double>{9, 4}),
          b + ": a flat box shrinks one coordinate");
}

void minkowski_difference_can_be_empty(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({1.0, 4.0}));
    const Interval E = I.minkowski_difference(Interval(column({-1.0, 0.0}), column({1.0, 1.0})));
    check(E.is_empty() && E.dim() == 2, b + ": a bound would cross: the empty interval");
    const Interval B = Interval::stack({I, I});
    check(
        B.minkowski_difference(Interval::stack({Interval(column({-1.0, 0.0}), column({1.0, 1.0})),
                                                Interval(column({-1.0, 0.0}), column({1.0, 1.0}))}))
            .is_empty(),
        b + ": a batch of empty differences");
}

void minkowski_difference_of_another_set(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({4.0, 4.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({0.5, 0, 0, 0.5}, {2, 2}));
    const Interval R = I.minkowski_difference(Z);
    check(close(R.inf(), std::vector<double>{0.5, 0.5}) &&
              close(R.sup(), std::vector<double>{3.5, 3.5}),
          b + ": the support function bounds each coordinate");
    const Interval Q = I.minkowski_difference(Z, std::string("inner"));
    check(close(Q.inf(), R.inf()) && close(Q.sup(), R.sup()), b + ": type inner is the default");
    const Zonotope W(column({0.0, 0.0}), Tensor::from_data({3, 0, 0, 3}, {2, 2}));
    check(I.minkowski_difference(W).is_empty(), b + ": a set that does not fit");
}

/// A wrong dimension, an unknown type, and "exact" for a set are refused; the checks off give the
/// same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({4.0, 4.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({0.5, 0, 0, 0.5}, {2, 2}));
    check(test::raises([&] { I.minkowski_difference(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "minkowski_difference: a box of dimension 1");
    check(test::raises([&] { I.minkowski_difference(column({1.0})); }, Error::dimension_mismatch),
          "minkowski_difference: a point of dimension 1");
    check(
        test::raises([&] { I.minkowski_difference(Z, std::string("exact")); }, Error::no_exact_alg),
        "minkowski_difference: no exact algorithm for a zonotope");
    check(
        test::raises([&] { I.minkowski_difference(Z, std::string("outer")); }, Error::wrong_value),
        "minkowski_difference: an unknown type");
    const Tensor on = I.minkowski_difference(Z).sup();
    ChecksScope off(false);
    check(close(I.minkowski_difference(Z).sup(), on, 0),
          "minkowski_difference: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        minkowski_difference_of_boxes_and_points(b);
        minkowski_difference_can_be_empty(b);
        minkowski_difference_of_another_set(b);
    });
    checks_and_their_switch();
    return test::finish("interval minkowski_difference");
}
