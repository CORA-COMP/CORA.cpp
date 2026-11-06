// test_interval_cat - interval cat: intervals joined along a dimension
//
// CORA: contSet/@interval/cat.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
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

void cat_joins_rows_and_columns(const std::string &b) {
    const Interval I1(column({-0.5, 0.3}), column({0.5, 0.9})),
        I2(column({2.0, 3.0}), column({2.5, 4.0}));
    const Interval V = Interval::cat(1, {I1, I2});
    check(V.dim() == 4 && close(V.inf(), std::vector<double>{-0.5, 0.3, 2.0, 3.0}),
          b + ": dim 1 stacks");
    const Interval H = Interval::cat(2, {I1, I2, I1});
    check(H.inf().shape() == std::vector<int64_t>({2, 3}) &&
              close(H.sup(), std::vector<double>{0.5, 2.5, 0.5, 0.9, 4.0, 0.9}),
          b + ": dim 2 joins three columns");
    check(close(Interval::cat(1, {I1}).sup(), I1.sup()), b + ": one interval is itself");
}

/// A batch is joined member by member.
void cat_of_a_batch(const std::string &b) {
    const Interval A(column({0.0}), column({1.0})), B(column({2.0}), column({3.0}));
    const Interval J = Interval::cat(1, {Interval::stack({A, B}), Interval::stack({B, A})});
    check(J.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(J.inf(), std::vector<double>{0, 2, 2, 0}),
          b + ": member by member");
}

/// Wrong dim, no interval, or shapes that do not fit are refused; checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0})), J(column({0.0}), column({1.0}));
    check(test::raises([&] { Interval::cat(0, {I}); }, Error::wrong_value), "cat: dim 0");
    check(test::raises([&] { Interval::cat(3, {I}); }, Error::not_supported), "cat: dim 3");
    check(test::raises([&] { Interval::cat(1, {}); }, Error::wrong_value), "cat: no interval");
    check(test::raises([&] { Interval::cat(2, {I, J}); }, Error::dimension_mismatch),
          "cat: columns of 2 and 1 rows side by side");
    const Tensor on = Interval::cat(1, {I, I}).sup();
    ChecksScope off(false);
    check(close(Interval::cat(1, {I, I}).sup(), on, 0), "cat: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        cat_joins_rows_and_columns(b);
        cat_of_a_batch(b);
    });
    checks_and_their_switch();
    return test::finish("interval cat");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
