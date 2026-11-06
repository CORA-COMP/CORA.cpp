// test_interval_ctranspose - interval ctranspose: the transpose of an interval matrix
//
// CORA: contSet/@interval/ctranspose.m
// CORA authors: Dmitry Grebenyuk
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

void ctranspose_swaps_rows_and_columns(const std::string &b) {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    const Interval M(lo, lo + 1.0);
    const Interval T = M.ctranspose();
    check(T.inf().shape() == std::vector<int64_t>({3, 2}) &&
              close(T.inf(), std::vector<double>{1, 4, 2, 5, 3, 6}) &&
              close(T.sup(), std::vector<double>{2, 5, 3, 6, 4, 7}),
          b + ": the bounds transposed");
    check(close(T.ctranspose().inf(), M.inf()), b + ": twice is the identity");
    const Interval c(column({0.0, 1.0}), column({1.0, 2.0}));
    check(c.ctranspose().inf().shape() == std::vector<int64_t>({1, 2}),
          b + ": a column becomes a row");
    const Interval S = Interval::stack({M, M}).ctranspose();
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 2}),
          b + ": a batch is transposed per member");
    check(close(S.sup().slice(0, 1, 2).reshape({3, 2}), T.sup()), b + ": a batch member");
}

/// ctranspose has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval c(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor on = c.ctranspose().sup();
    ChecksScope off(false);
    check(close(c.ctranspose().sup(), on, 0), "ctranspose: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(ctranspose_swaps_rows_and_columns);
    checks_and_their_switch();
    return test::finish("interval ctranspose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
