// test_interval_horzcat - interval horzcat: intervals joined side by side
//
// CORA: contSet/@interval/horzcat.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void horzcat_joins_columns(const std::string &b) {
    const Interval I1(column({-0.5, 0.3}), column({0.5, 0.9})),
        I2(column({2.0, 3.0}), column({2.5, 4.0}));
    const Interval H = Interval::horzcat({I1, I2, I1});
    check(H.inf().shape() == std::vector<int64_t>({2, 3}) &&
              close(H.inf(), std::vector<double>{-0.5, 2.0, -0.5, 0.3, 3.0, 0.3}) &&
              close(H.sup(), std::vector<double>{0.5, 2.5, 0.5, 0.9, 4.0, 0.9}),
          b + ": three columns side by side");
    check(close(Interval::horzcat({I1}).sup(), I1.sup()), b + ": one interval is itself");
    const Interval M = Interval::horzcat({H, I2});
    check(M.inf().shape() == std::vector<int64_t>({2, 4}), b + ": a matrix and a column");
    check(close(Interval::horzcat({I1, I2}).inf(), Interval::cat(2, {I1, I2}).inf()),
          b + ": the same as cat along 2");
    const Interval S = Interval::horzcat({Interval::stack({I1, I2}), Interval::stack({I2, I1})});
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 2}), b + ": a batch member by member");
}

/// No interval, or other numbers of rows, are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0})), J(column({0.0}), column({1.0}));
    check(test::raises([] { Interval::horzcat({}); }, Error::wrong_value), "horzcat: no interval");
    check(test::raises([&] { Interval::horzcat({I, J}); }, Error::dimension_mismatch),
          "horzcat: columns of 2 and 1 rows");
    const Tensor on = Interval::horzcat({I, I}).sup();
    ChecksScope off(false);
    check(close(Interval::horzcat({I, I}).sup(), on, 0), "horzcat: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(horzcat_joins_columns);
    checks_and_their_switch();
    return test::finish("interval horzcat");
}
