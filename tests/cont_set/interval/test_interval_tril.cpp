// test_interval_tril - interval tril: the lower triangular part of an interval matrix
//
// CORA: contSet/@interval/tril.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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

Interval aux_matrix() {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    return {lo, lo + 1.0};
}

void tril_keeps_the_lower_part(const std::string &b) {
    const Interval M = aux_matrix();
    check(close(M.tril().inf(), std::vector<double>{1, 0, 0, 4, 5, 0, 7, 8, 9}) &&
              close(M.tril().sup(), std::vector<double>{2, 0, 0, 5, 6, 0, 8, 9, 10}),
          b + ": K = 0 keeps the diagonal and below");
    check(close(M.tril(1).inf(), std::vector<double>{1, 2, 0, 4, 5, 6, 7, 8, 9}),
          b + ": K = 1 keeps the first superdiagonal");
    check(Interval::stack({M, M}).tril().inf().shape() == std::vector<int64_t>({2, 3, 3}),
          b + ": a batch keeps its shape");
}

/// K must not be negative; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval M = aux_matrix();
    check(test::raises([&] { M.tril(-1); }, Error::wrong_value), "tril: K = -1");
    const Tensor on = M.tril(1).sup();
    ChecksScope off(false);
    check(close(M.tril(1).sup(), on, 0), "tril: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(tril_keeps_the_lower_part);
    checks_and_their_switch();
    return test::finish("interval tril");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
