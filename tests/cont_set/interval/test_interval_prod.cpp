// test_interval_prod - interval prod: the product of the rows or the columns of an interval matrix
//
// CORA: contSet/@interval/prod.m
// PROPERTY: contSet/@interval/prod.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// A vector is a column of factors; a matrix has a product per column (dim 1) or row (dim 2).
void prod_of_rows_and_columns(const std::string &b) {
    const Interval v(column({-3.0, 1.0, 0.5}), column({2.0, 3.0, 5.0}));
    const Interval P = v.prod();
    check(P.inf().shape() == std::vector<int64_t>({1, 1}) && close(P.inf(), std::vector<double>{-45}) &&
              close(P.sup(), std::vector<double>{30}),
          b + ": the product of a vector");
    // columns (1, 2, 3) and (-1, 4, 1) to (2, 3, 5), (1, 2, 2)
    const Interval M(Tensor::from_data({1, -1, 2, 4, 3, 1}, {3, 2}),
                     Tensor::from_data({2, 1, 3, 4, 5, 2}, {3, 2}));
    const Interval C = M.prod(1);
    check(C.inf().shape() == std::vector<int64_t>({1, 2}) &&
              close(C.inf(), std::vector<double>{6, -8}) && close(C.sup(), std::vector<double>{30, 8}),
          b + ": dim 1, a row of column products");
    const Interval R = M.prod(2);
    check(R.inf().shape() == std::vector<int64_t>({3, 1}) &&
              close(R.inf(), std::vector<double>{-2, 8, 3}) && close(R.sup(), std::vector<double>{2, 12, 10}),
          b + ": dim 2, a column of row products");
    const Interval row(Tensor::from_data({1, 2, 3}, {1, 3}), Tensor::from_data({1, 2, 4}, {1, 3}));
    check(close(row.prod().sup(), std::vector<double>{8}), b + ": a single row defaults to dim 2");
    const Interval S = Interval::stack({M, M}).prod(1);
    check(S.inf().shape() == std::vector<int64_t>({2, 1, 2}), b + ": a batch has a batch shape");
}

/// 0 * Inf counts as 0, as in CORA's times.
void prod_with_infinite_factors(const std::string &b) {
    const double kInf = std::numeric_limits<double>::infinity();
    const Interval v(column({0.0, 1.0}), column({0.0, kInf}));
    const Interval P = v.prod();
    check(P.inf().data()[0] == 0 && P.sup().data()[0] == 0, b + ": [0, 0] times [1, Inf] is 0");
}

/// The product of sampled points of the factors is in the result.
void prod_encloses_the_image(const std::string &b) {
    cora::Rng rng(61);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(4, rng), P = I.prod();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k)
            inside = inside && P.contains(x.slice(-1, k, k + 1).prod(-2, true));
        check(inside, b + ": the product of a sample is in the result");
    }
}

/// A dimension other than 1 and 2 is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval v(column({1.0, 2.0}), column({2.0, 3.0}));
    check(test::raises([&] { v.prod(3); }, Error::wrong_value), "prod: dimension 3");
    check(test::raises([&] { v.prod(0); }, Error::wrong_value), "prod: dimension 0");
    const Tensor on = v.prod().sup();
    ChecksScope off(false);
    check(close(v.prod().sup(), on, 0), "prod: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        prod_of_rows_and_columns(b);
        prod_with_infinite_factors(b);
        prod_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval prod");
}
