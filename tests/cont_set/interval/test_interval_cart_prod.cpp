// test_interval_cart_prod - interval cart_prod: boxes stacked, a point as a degenerate box
//
// CORA: contSet/@interval/cartProd_.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void cart_prod_stacks_columns(const std::string &b) {
    const Interval I1(column({-1.0, -3.0}), column({1.0, 6.0})),
        I2(column({-2.0, 1.0}), column({4.0, 2.0}));
    const Interval P = I1.cart_prod(I2);
    check(P.dim() == 4 && close(P.inf(), std::vector<double>{-1, -3, -2, 1}) &&
              close(P.sup(), std::vector<double>{1, 6, 4, 2}),
          b + ": the columns are stacked");
    const Set S = cart_prod(I1, I2);
    check(close(S.as<Interval>().inf(), P.inf()), b + ": the registry gives the same box");
}

/// Two rows are joined side by side; a column with a row is refused.
void cart_prod_joins_rows(const std::string &b) {
    const Interval R1(Tensor({{0.0, 1.0}}), Tensor({{1.0, 2.0}})),
        R2(Tensor({{5.0}}), Tensor({{6.0}}));
    const Interval P = R1.cart_prod(R2);
    check(P.inf().shape() == std::vector<int64_t>({1, 3}) &&
              close(P.sup(), std::vector<double>{1, 2, 6}),
          b + ": the rows are joined");
    check(test::raises([&] { R1.cart_prod(Interval(column({0.0, 1.0}), column({1.0, 2.0}))); },
                       Error::dimension_mismatch),
          b + ": a row and a column of 2 entries");
}

/// An empty operand returns the other; a point is the box [p, p], on either side.
void cart_prod_with_empty_and_with_a_point(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor none = I.inf().zeros_like({2, 0});
    const Interval E(none, none);
    check(close(E.cart_prod(I).inf(), I.inf()) && close(I.cart_prod(E).sup(), I.sup()),
          b + ": the empty interval is neutral");
    const Tensor p = column({7.0});
    check(close(I.cart_prod(p).inf(), std::vector<double>{0, 1, 7}) &&
              close(I.cart_prod(p).sup(), std::vector<double>{1, 2, 7}),
          b + ": the point after the box");
    check(close(Interval::cart_prod(p, I).inf(), std::vector<double>{7, 0, 1}),
          b + ": the point before the box");
}

/// A batch is stacked member by member; mismatched batches are refused; checks off give the same.
void checks_and_their_switch() {
    const Interval A(column({0.0}), column({1.0})), B(column({2.0}), column({3.0}));
    const Interval two = Interval::stack({A, B}), three = Interval::stack({A, B, A});
    const Interval P = two.cart_prod(two);
    check(P.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(P.sup(), std::vector<double>{1, 1, 3, 3}),
          "cart_prod: a batch of two");
    check(test::raises([&] { two.cart_prod(three); }, Error::dimension_mismatch),
          "cart_prod: batches of 2 and 3");
    check(test::raises([&] { A.cart_prod(Tensor::from_data({1.0}, {1})); },
                       Error::dimension_mismatch),
          "cart_prod: a point that is no column");
    const Tensor on = A.cart_prod(B).sup();
    ChecksScope off(false);
    check(close(A.cart_prod(B).sup(), on, 0), "cart_prod: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        cart_prod_stacks_columns(b);
        cart_prod_joins_rows(b);
        cart_prod_with_empty_and_with_a_point(b);
    });
    checks_and_their_switch();
    return test::finish("interval cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
