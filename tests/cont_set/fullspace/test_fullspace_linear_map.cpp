// test_fullspace_linear_map - fullspace linear_map: the image of the whole space under a matrix
//
// CORA: contSet/@fullspace/mtimes.m
// PROPERTY: contSet/@fullspace/mtimes.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Tensor matrix(const std::vector<double> &rows, int64_t m, int64_t n) {
    return Tensor::from_data(rows, {m, n});
}

void linear_map_cases_of_cora(const std::string &b) {
    const Fullspace F(2);
    const Set full_rank = F.linear_map(matrix({1, 2, 3, 4}, 2, 2));
    check(full_rank.is<Fullspace>() && full_rank->dim() == 2, b + ": a square matrix");
    const Set tall = F.linear_map(matrix({1, 2, 3, 4, 5, 6}, 3, 2));
    check(tall.is<Fullspace>() && tall->dim() == 3, b + ": more rows");
    const Set wide = Fullspace(3).linear_map(matrix({1, 2, 3}, 1, 3));
    check(wide.is<Fullspace>() && wide->dim() == 1, b + ": fewer rows");
    const Set zero_row = F.linear_map(matrix({1, 2, 0, 0}, 2, 2));
    check(zero_row.is<Interval>() &&
              zero_row.as<Interval>().inf().data() == std::vector<double>({-kInf, 0.0}) &&
              zero_row.as<Interval>().sup().data() == std::vector<double>({kInf, 0.0}),
          b + ": a zero row stays at 0, the other is unbounded");
    check((matrix({2, 0, 0, 3}, 2, 2) * F).is<Fullspace>(), b + ": the operator M * F");
    // a scalar, as a 1 x 1 matrix: 0 gives the origin, any other number the space
    check(F.linear_map(matrix({0.0}, 1, 1)).as<Interval>().sup().data() ==
              std::vector<double>(2, 0.0),
          b + ": the scalar 0 is the origin");
    check(F.linear_map(matrix({-2.0}, 1, 1)).is<Fullspace>(), b + ": a nonzero scalar");
    check(F.linear_map(matrix({1e-17}, 1, 1)).is<Interval>(), b + ": a scalar within eps of 0");
}

/// PROPERTY: for random matrices, with and without a zero row, every sampled image M x lies in the
/// box of the result (it is an enclosure).
void the_result_encloses_the_image(const std::string &b) {
    Rng rng(4);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> data(6);
        rng.normal(data.data(), data.size(), 1.0);
        if (trial % 3 == 0) data[2] = data[3] = 0.0; // a zero row
        const Tensor M = matrix(data, 3, 2);
        const Interval hull = Fullspace(2).linear_map(M)->interval();
        const std::vector<double> y = M.matmul(Fullspace(2).rand_point(5, rng)).data();
        const std::vector<double> lo = hull.inf().data(), hi = hull.sup().data();
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t j = 0; j < 5; ++j)
                check(lo[i] <= y[i * 5 + j] && y[i * 5 + j] <= hi[i],
                      b + ": a point of M F lies in the result");
    }
}

/// R^0, a wrong shape, a batch and an interval matrix are refused.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([] { Fullspace(0).linear_map(matrix({1.0}, 1, 1)); }, Error::not_supported),
          "linear_map: R^0");
    check(test::raises([&] { F.linear_map(matrix({1, 2, 3}, 1, 3)); }, Error::dimension_mismatch),
          "linear_map: three columns for R^2");
    check(test::raises([&] { F.linear_map(Tensor::zeros({2, 1, 2})); }, Error::dimension_mismatch),
          "linear_map: a batch of matrices");
    check(test::raises([&] { F.linear_map(Interval::unbounded(2)); }, Error::noops),
          "linear_map: an interval matrix");
    ChecksScope off(false);
    check(F.linear_map(matrix({1, 0, 0, 1}, 2, 2)).is<Fullspace>(),
          "linear_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(linear_map_cases_of_cora);
    test::for_each_backend(the_result_encloses_the_image);
    checks_and_their_switch();
    return test::finish("fullspace linear_map");
}
