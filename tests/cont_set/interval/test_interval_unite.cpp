// test_interval_unite - interval unite: the smallest box around a box and a box or a point (or)
//
// CORA: contSet/@interval/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void unite_is_the_hull(const std::string &b) {
    const Interval I(vec({0.0, 0.0}), vec({1.0, 1.0})), J(vec({2.0, -1.0}), vec({3.0, 0.5}));
    const Interval R = I.unite(J);
    check(close(R.inf(), std::vector<double>{0, -1}) && close(R.sup(), std::vector<double>{3, 1}),
          b + ": [min(inf, inf2), max(sup, sup2)]");
    check(close(I.unite(Interval::empty(2)).sup(), I.sup()) &&
              close(Interval::empty(2).unite(J).inf(), J.inf()),
          b + ": an empty operand returns the other");
    const Interval P = I.unite(vec({-1.0, 3.0}));
    check(close(P.inf(), std::vector<double>{-1, 0}) && close(P.sup(), std::vector<double>{1, 3}),
          b + ": a point grows the box");
    check(close(Interval::empty(2).unite(vec({1.0, 2.0})).sup(), std::vector<double>{1, 2}),
          b + ": a point and the empty interval");
    const Set S = unite(I, J);
    check(close(S.as<Interval>().sup(), R.sup()), b + ": the registry pair");
    const Interval B = Interval::stack({I, J}).unite(J);
    // a batch
    check(B.inf().shape() == std::vector<int64_t>({2, 2}) &&
              close(B.sup().slice(0, 0, 1).reshape({2}), R.sup()),
          b + ": a batch");
}

/// A box of another dimension and a point of another shape are refused; the checks off give the
/// same.
void checks_and_their_switch() {
    const Interval I(vec({0.0, 0.0}), vec({1.0, 1.0}));
    check(
        test::raises([&] { I.unite(Interval(vec({0.0}), vec({1.0}))); }, Error::dimension_mismatch),
        "unite: a box of dimension 1");
    check(test::raises([&] { I.unite(vec({1.0})); }, Error::dimension_mismatch),
          "unite: a point of dimension 1");
    const Tensor on = I.unite(I).sup();
    ChecksScope off(false);
    check(close(I.unite(I).sup(), on, 0), "unite: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(unite_is_the_hull);
    checks_and_their_switch();
    return test::finish("interval unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
