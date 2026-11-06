// test_interval_vertcat - interval vertcat: intervals stacked
//
// CORA: contSet/@interval/vertcat.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void vertcat_stacks_the_rows(const std::string &b) {
    const Interval A(column({0.0, 1.0}), column({1.0, 2.0})), B(column({5.0}), column({6.0}));
    const Interval R = Interval::vertcat({A, B, A});
    check(R.inf().shape() == std::vector<int64_t>({5, 1}) &&
              close(R.inf(), std::vector<double>{0, 1, 5, 0, 1}) &&
              close(R.sup(), std::vector<double>{1, 2, 6, 1, 2}),
          b + ": the columns stacked");
    check(close(Interval::vertcat({A}).inf(), A.inf(), 0), b + ": one interval is itself");
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    const Interval M(lo, lo + 1.0), V = Interval::vertcat({M, M});
    check(V.inf().shape() == std::vector<int64_t>({4, 2}) &&
              close(V.sup().slice(-2, 2, 4), M.sup(), 0),
          b + ": matrices with the same columns");
    check(close(Interval::cat(1, {A, B}).inf(), Interval::vertcat({A, B}).inf(), 0),
          b + ": the same as cat along 1");
    const Interval S = Interval::vertcat({Interval::stack({A, A}), Interval::stack({B, B})});
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 1}), b + ": a batch stacks per member");
}

/// The intervals must agree in the columns and there must be one; with the checks off a valid call
/// gives the same result.
void checks_and_their_switch() {
    const Interval A(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    check(test::raises([&] { Interval::vertcat({A, Interval(lo, lo + 1.0)}); },
                       Error::dimension_mismatch),
          "vertcat: different numbers of columns");
    check(test::raises([] { Interval::vertcat({}); }, Error::wrong_value), "vertcat: no interval");
    const Tensor on = Interval::vertcat({A, A}).sup();
    ChecksScope off(false);
    check(close(Interval::vertcat({A, A}).sup(), on, 0), "vertcat: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(vertcat_stacks_the_rows);
    checks_and_their_switch();
    return test::finish("interval vertcat");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
