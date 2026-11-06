// test_interval_sum - interval sum: the sum of the entries of an interval matrix along a dimension
//
// CORA: contSet/@interval/sum.m
// PROPERTY: contSet/@interval/sum.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Interval aux_matrix() {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    return {lo, lo + 1.0};
}

void sum_adds_the_bounds(const std::string &b) {
    const Interval M = aux_matrix();
    const Interval down = M.sum(1), along = M.sum(2);
    check(down.inf().shape() == std::vector<int64_t>({1, 3}) &&
              close(down.inf(), std::vector<double>{5, 7, 9}) &&
              close(down.sup(), std::vector<double>{7, 9, 11}),
          b + ": dim 1 sums down the rows");
    check(along.inf().shape() == std::vector<int64_t>({2, 1}) &&
              close(along.inf(), std::vector<double>{6, 15}) &&
              close(along.sup(), std::vector<double>{9, 18}),
          b + ": dim 2 sums along the columns");
    const Interval v(column({1.0, 2.0, 3.0}), column({2.0, 3.0, 4.0}));
    check(close(v.sum().inf(), std::vector<double>{6}) &&
              close(v.sum().sup(), std::vector<double>{9}),
          b + ": a column sums to one entry by default");
    const Interval row = v.transpose();
    check(row.sum().inf().shape() == std::vector<int64_t>({1, 1}) &&
              close(row.sum().sup(), std::vector<double>{9}),
          b + ": a row sums along its columns by default");
    const Interval S = Interval::stack({M, M}).sum(1);
    check(S.inf().shape() == std::vector<int64_t>({2, 1, 3}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({1, 3}), down.sup()), b + ": a batch member");
}

/// The sum of every sampled point lies in the result.
void sum_encloses_the_sums(const std::string &b) {
    cora::Rng rng(51);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(rng, 4), R = I.sum();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).sum());
        check(inside, b + ": the sum of a sample is in the result");
    }
}

/// dim must be 1 or 2; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval M = aux_matrix();
    check(test::raises([&] { M.sum(3); }, Error::wrong_value), "sum: dim 3");
    const Tensor on = M.sum(1).sup();
    ChecksScope off(false);
    check(close(M.sum(1).sup(), on, 0), "sum: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sum_adds_the_bounds(b);
        sum_encloses_the_sums(b);
    });
    checks_and_their_switch();
    return test::finish("interval sum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
