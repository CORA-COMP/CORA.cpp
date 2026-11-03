// test_interval_transpose - interval transpose: the transpose of an interval matrix
//
// CORA: contSet/@interval/transpose.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void transpose_swaps_rows_and_columns(const std::string &b) {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    const Interval M(lo, lo + 1.0);
    const Interval T = M.transpose();
    check(T.inf().shape() == std::vector<int64_t>({3, 2}) &&
              close(T.inf(), std::vector<double>{1, 4, 2, 5, 3, 6}) &&
              close(T.sup(), std::vector<double>{2, 5, 3, 6, 4, 7}),
          b + ": the bounds transposed");
    check(close(T.transpose().inf(), M.inf()), b + ": twice is the identity");
    check(close(M.ctranspose().inf(), T.inf(), 0), b + ": the same as ctranspose");
    const Interval c(column({0.0, 1.0}), column({1.0, 2.0}));
    check(c.transpose().inf().shape() == std::vector<int64_t>({1, 2}),
          b + ": a column becomes a row");
    check(Interval::stack({M, M}).transpose().inf().shape() == std::vector<int64_t>({2, 3, 2}),
          b + ": a batch is transposed per member");
}

/// transpose has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval c(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor on = c.transpose().sup();
    ChecksScope off(false);
    check(close(c.transpose().sup(), on, 0), "transpose: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(transpose_swaps_rows_and_columns);
    checks_and_their_switch();
    return test::finish("interval transpose");
}
