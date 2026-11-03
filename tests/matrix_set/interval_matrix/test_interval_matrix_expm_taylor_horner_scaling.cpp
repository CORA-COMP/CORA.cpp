// test_interval_matrix_expm_taylor_horner_scaling - IntervalMatrix expm_taylor_horner_scaling
//
// CORA: matrixSet/@intervalMatrix/expmTaylorHornerScaling.m
// CORA: matrixSet/@intervalMatrix/private/priv_taylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_hornerTaylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_scalingSquaringHornerTaylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_intervalMatrixRemainder.m
// PROPERTY: matrixSet/@intervalMatrix/expmTaylorHornerScaling.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The point matrix -1 at order 2: the series is 1 - 1 + 1/2 = 1/2 and the remainder
/// (1/6) / (1 - 1/4) = 2/9, so [5/18, 13/18], which holds exp(-1) = 0.3679 (CORA's remainder for a
/// point matrix is the point 2/9, so its result is the point 13/18, which does not).
void the_point_matrix_minus_one(const std::string &b) {
    const IntervalMatrix A(test::matrix({-1}, 1, 1));
    for (const int style : {0, 1}) {
        const IntervalMatrix E = *A.expm_taylor_horner_scaling(2, style);
        check(close(E.interval().inf(), std::vector<double>{5.0 / 18}, 1e-12) &&
                  close(E.interval().sup(), std::vector<double>{13.0 / 18}, 1e-12),
              b + ": style " + std::to_string(style));
    }
    // scaling by 2: Horner of -1/2 is [0.601190, 0.648810] (by hand), squared as intervals
    const IntervalMatrix S = *A.expm_taylor_horner_scaling(2, 2);
    check(close(S.interval().inf(), std::vector<double>{std::pow(0.625 - 0.5 * 0.25 / 6 / 0.875, 2)}, 1e-12),
          b + ": scaled, lower bound");
    check(S.contains(test::matrix({std::exp(-1.0)}, 1, 1)), b + ": scaled, exp(-1) is inside");
}

/// The exponentials of members are inside, for all three styles and for sets and points.
void members_are_enclosed(const std::string &b) {
    Rng rng(38);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.3);
    const IntervalMatrix P(test::member(A, rng));
    for (const int style : {0, 1, 2}) {
        const std::optional<IntervalMatrix> E = A.expm_taylor_horner_scaling(8, style, 2);
        const std::optional<IntervalMatrix> F = P.expm_taylor_horner_scaling(8, style, 2);
        bool inside = E && F;
        for (int trial = 0; inside && trial < 20; ++trial)
            inside = E->contains(test::member(A, rng).expm(), "exact", 1e-12);
        inside = inside && F->contains(P.center().expm(), "exact", 1e-12);
        check(inside, b + ": style " + std::to_string(style));
    }
}

/// Nothing where the norm is not below the order + 2; a batch is per member.
void no_result_without_convergence(const std::string &b) {
    const IntervalMatrix big(test::matrix({5}, 1, 1));
    check(!big.expm_taylor_horner_scaling(2, 0) && !big.expm_taylor_horner_scaling(2, 1),
          b + ": norm 5, order 2");
    check(!big.expm_taylor_horner_scaling(2, 2, 0) && big.expm_taylor_horner_scaling(2, 2, 2),
          b + ": scaled by 4");
    const IntervalMatrix small(test::matrix({0.5}, 1, 1)), small2(test::matrix({-0.25}, 1, 1));
    const IntervalMatrix batch(Interval::stack({small.interval(), small2.interval()}));
    const std::optional<IntervalMatrix> E = batch.expm_taylor_horner_scaling(4, 0);
    check(E && close(E->interval().inf().slice(0, 1, 2).reshape({1}),
                     small2.expm_taylor_horner_scaling(4, 0)->interval().inf().reshape({1}), 0),
          b + ": a batch");
}

/// Another style, a style 4, a matrix that is not square, order 0 for Horner and a scaling
/// exponent out of range are refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    const IntervalMatrix A(test::matrix({0.5}, 1, 1));
    check(test::raises([&] { A.expm_taylor_horner_scaling(3, 3); }, Error::wrong_value), "style 3");
    check(test::raises([&] { A.expm_taylor_horner_scaling(3, 4); }, Error::wrong_value), "style 4");
    check(test::raises([&] { A.expm_taylor_horner_scaling(0, 1); }, Error::wrong_value), "order 0, Horner");
    check(test::raises([&] { A.expm_taylor_horner_scaling(3, 2, -1); }, Error::wrong_value), "exponent -1");
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { wide.expm_taylor_horner_scaling(3, 0); }, Error::dimension_mismatch),
          "not square");
    const Tensor on = A.expm_taylor_horner_scaling(3, 0)->interval().sup();
    ChecksScope off(false);
    check(close(A.expm_taylor_horner_scaling(3, 0)->interval().sup(), on, 0), "the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        the_point_matrix_minus_one(b);
        no_result_without_convergence(b);
    });
    test::for_each_backend(members_are_enclosed); // exp(M) is not on the lean backend
    checks_and_their_switch();
    return test::finish("interval_matrix expm_taylor_horner_scaling");
}
