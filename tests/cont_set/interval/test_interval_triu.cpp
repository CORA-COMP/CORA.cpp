// test_interval_triu - interval triu: the upper triangular part of an interval matrix
//
// CORA: contSet/@interval/triu.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

Interval aux_matrix() {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    return {lo, lo + 1.0};
}

void triu_keeps_the_upper_part(const std::string &b) {
    const Interval M = aux_matrix();
    check(close(M.triu().inf(), std::vector<double>{1, 2, 3, 0, 5, 6, 0, 0, 9}) &&
              close(M.triu().sup(), std::vector<double>{2, 3, 4, 0, 6, 7, 0, 0, 10}),
          b + ": K = 0 keeps the diagonal and above");
    check(close(M.triu(1).inf(), std::vector<double>{0, 2, 3, 0, 0, 6, 0, 0, 0}),
          b + ": K = 1 drops the diagonal");
    check(Interval::stack({M, M}).triu().inf().shape() == std::vector<int64_t>({2, 3, 3}),
          b + ": a batch keeps its shape");
}

/// K must not be negative; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval M = aux_matrix();
    check(test::raises([&] { M.triu(-1); }, Error::wrong_value), "triu: K = -1");
    const Tensor on = M.triu(1).sup();
    ChecksScope off(false);
    check(close(M.triu(1).sup(), on, 0), "triu: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(triu_keeps_the_upper_part);
    checks_and_their_switch();
    return test::finish("interval triu");
}
