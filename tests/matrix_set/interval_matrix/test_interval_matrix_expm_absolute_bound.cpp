// test_interval_matrix_expm_absolute_bound - IntervalMatrix expm_absolute_bound: CORA's matrix bound
//
// CORA: matrixSet/@intervalMatrix/expmAbsoluteBound.m
// PROPERTY: matrixSet/@intervalMatrix/expmAbsoluteBound.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The last block row of G^k [|C| t; S t; 0] for a large k, G CORA's block matrix: what the bound
/// is the limit of.
Tensor limit_by_powers(const IntervalMatrix &A, double t) {
    const Tensor C = A.center(), S = A.rad(), I = C.eye_like(A.rows()), O = C.zeros_like();
    const Tensor row1 = Tensor::concat({C.abs() * t / 2, O, O}, 1);
    const Tensor row2 = Tensor::concat({S * t / 2, (C.abs() + S) * t / 2, O}, 1);
    const Tensor row3 = Tensor::concat({O, I, I}, 1);
    const Tensor G = Tensor::concat({row1, row2, row3}, 0);
    Tensor power = G;
    for (int k = 1; k < 300; ++k) power = power.matmul(G);
    const Tensor aux = Tensor::concat({C.abs() * t, S * t, O}, 0);
    const int64_t n = A.rows();
    return power.matmul(aux).slice(0, 2 * n, 3 * n);
}

/// By hand for the 1 x 1 set 0.5 +- 0.5 at t = 1: B1 = 0.25, B2 = 0.5, the bound is
/// (0.25 * (0.5 / 0.75) + 0.5) / 0.5 = 4/3; and it is the limit of the powers of G.
void bound_of_small_sets(const std::string &b) {
    const IntervalMatrix A(test::matrix({0.5}, 1, 1), test::matrix({0.5}, 1, 1));
    check(close(A.expm_absolute_bound(1.0), std::vector<double>{4.0 / 3.0}, 1e-12), b + ": 1 x 1 by hand");
    const IntervalMatrix B(test::matrix({0.3, -0.2, 0.1, 0.4}, 2, 2),
                           test::matrix({0.1, 0.05, 0.02, 0.1}, 2, 2));
    check(close(B.expm_absolute_bound(1.0), limit_by_powers(B, 1.0), 1e-9), b + ": the limit of G^k, 2 x 2");
    check(close(B.expm_absolute_bound(0.5), limit_by_powers(B, 0.5), 1e-9), b + ": ... at t = 0.5");
    check(close(IntervalMatrix(test::matrix({0.3, 0.1, 0.2, 0.4}, 2, 2)).expm_absolute_bound(1.0),
                std::vector<double>(4, 0.0), 1e-12),
          b + ": a point matrix has no radius: 0");
}

/// Random sets of small norm: the limit of the powers; a batch is per member.
void bound_of_random_sets(const std::string &b) {
    Rng rng(34);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.2), T = test::random_matrix(rng, 3, 0.2);
    check(close(A.expm_absolute_bound(0.7), limit_by_powers(A, 0.7), 1e-8), b + ": random 3 x 3");
    const IntervalMatrix batch(Interval::stack({A.interval(), T.interval()}));
    const Tensor second = batch.expm_absolute_bound(0.7).slice(0, 1, 2).reshape({3, 3});
    check(close(second, T.expm_absolute_bound(0.7), 1e-12), b + ": a batch");
}

/// A matrix that is not square is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    check(test::raises([&] { IntervalMatrix(test::matrix({1, 2, 3}, 1, 3)).expm_absolute_bound(1.0); },
                       Error::dimension_mismatch),
          "expm_absolute_bound: not square");
    check(test::raises([&] { test::example().expm_absolute_bound(-1.0); }, Error::wrong_value),
          "expm_absolute_bound: a negative time");
    const IntervalMatrix A(test::matrix({0.5}, 1, 1), test::matrix({0.5}, 1, 1));
    const Tensor on = A.expm_absolute_bound(1.0);
    ChecksScope off(false);
    check(close(A.expm_absolute_bound(1.0), on, 0), "expm_absolute_bound: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        bound_of_small_sets(b);
        bound_of_random_sets(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_absolute_bound");
}
