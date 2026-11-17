// test_mat_exp_expm - Interval expm: the enclosure of the matrix exponential
//
// CORA: matrixSet/@intervalMatrix/expm.m
// PROPERTY: matrixSet/@intervalMatrix/expm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The exponential of every member is inside enclosure, for each form of the call; the zero matrix
/// gives I.
void expm_encloses_the_exponentials(const std::string &b) {
    Rng rng(17);
    const Interval A = test::random_matrix(rng, 3, 0.3);
    const Interval start = A.matrix_power(3);
    const std::vector<mat_exp::ExpmResult> results{
        mat_exp::expm(A), mat_exp::expm(A, 8), mat_exp::expm(A, 0.5, 8), mat_exp::expm(A, 2.0, 8)};
    for (std::size_t form = 0; form < results.size(); ++form) {
        bool inside = true;
        for (int trial = 0; trial < 30; ++trial) {
            const Tensor M = test::member(A, rng);
            inside = inside && results[form].enclosure.contains(M.expm(), 1e-12);
        }
        check(inside, b + ": exp(M) of every member is inside, form " + std::to_string(form));
    }
    // continued from [A]^3, the result is the series from the third order: exp(M) - I - M - M^2 / 2
    const mat_exp::ExpmResult tail = mat_exp::expm(A, 1.0, 8, 3, start);
    bool tail_inside = true;
    for (int trial = 0; trial < 30; ++trial) {
        const Tensor M = test::member(A, rng);
        const Tensor head = M.eye_like(3) + M + M.matmul(M) * 0.5;
        tail_inside = tail_inside && tail.enclosure.contains(M.expm() - head, 1e-12);
    }
    check(tail_inside, b + ": the five-argument form encloses the series from the third order");
    const mat_exp::ExpmResult zero =
        mat_exp::expm(test::interval_matrix(test::matrix({0, 0, 0, 0}, 2, 2)), 5);
    check(close(zero.enclosure.center(), std::vector<double>{1, 0, 0, 1}, 1e-14) &&
              close(zero.enclosure.rad(), std::vector<double>{0, 0, 0, 0}, 1e-14),
          b + ": exp(0) = I");
}

/// The series is summed to the order asked for: the powers listed, the remainder returned.
void expm_lists_its_powers(const std::string &b) {
    Rng rng(18);
    const Interval A = test::random_matrix(rng, 2, 0.3);
    const mat_exp::ExpmResult R = mat_exp::expm(A, 6);
    check(R.first_order == 2 && R.powers.size() == 5, b + ": the powers of order 2 to 6");
    check(close(R.powers[1].center(), A.matrix_power(3).center(), 0.2),
          b + ": the power of order 3");
    check(close(R.remainder.sup(), mat_exp::exponential_remainder(A, 6).sup(), 1e-14),
          b + ": the remainder of the order");
    const mat_exp::ExpmResult tighter = mat_exp::expm(A, 12);
    const double width = R.enclosure.rad().max().item();
    check(tighter.enclosure.rad().max().item() <= width + 1e-12,
          b + ": a higher order does not widen the result");
    const mat_exp::ExpmResult point = mat_exp::expm(test::interval_matrix(A.center()), 12);
    check(close(point.enclosure.center(), A.center().expm(), 1e-9),
          b + ": a point matrix is its exponential");
}

/// A matrix that is not square, a zero step and a negative order are refused; checks off: the same.
void checks_and_their_switch() {
    const Interval wide = test::interval_matrix(test::matrix({1, 2, 3}, 1, 3)), A = test::example();
    check(test::raises([&] { mat_exp::expm(wide, 4); }, Error::dimension_mismatch),
          "expm: a non-square matrix");
    check(test::raises([&] { mat_exp::expm(A, 0.0, 4); }, Error::wrong_value), "expm: the step 0");
    check(test::raises([&] { mat_exp::expm(A, -1); }, Error::wrong_value),
          "expm: a negative order");
    check(test::raises([&] { mat_exp::expm(A, 1.0, 4, 2, wide); }, Error::dimension_mismatch),
          "expm: an initial power of another size");
    const Tensor on = mat_exp::expm(A, 5).enclosure.center();
    ChecksScope off(false);
    check(close(mat_exp::expm(A, 5).enclosure.center(), on, 0),
          "expm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        expm_encloses_the_exponentials(b);
        expm_lists_its_powers(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
