// test_mat_exp_expm_independent_mixed - Interval expm_independent_mixed: two parts
//
// CORA: matrixSet/@intervalMatrix/expmIndMixed.m
// PROPERTY: matrixSet/@intervalMatrix/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
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

/// A = [[0, 1], [0, 0]] is nilpotent: the first part is I + A, the sum of both parts and the
/// remainder box encloses exp(A) = [[1, 1], [0, 1]]; the powers are those of powers().
void nilpotent_matrix(const std::string &b) {
    const Interval A = test::interval_matrix(test::matrix({0, 1, 0, 0}, 2, 2));
    const mat_exp::ExpmMixedResult R = mat_exp::expm_independent_mixed(A, 2, 5);
    check(close(R.enclosure.inf(), std::vector<double>{1, 1, 0, 1}) &&
              close(R.enclosure.sup(), std::vector<double>{1, 1, 0, 1}),
          b + ": I + A + A^2 / 2");
    check(R.powers.size() == 2 && R.powers2.size() == 3, b + ": the powers of both parts");
    const Interval total = test::interval_matrix(R.enclosure + R.enclosure_rest);
    check(total.contains(test::matrix({1, 1, 0, 1}, 2, 2), 1e-9), b + ": exp(A) is inside");
    const std::vector<Interval> P = mat_exp::powers(A, 2);
    check(close(R.powers[1].sup(), P[1].sup(), 0), b + ": the powers of powers()");
}

/// The exponentials of members are in the sum of both parts.
void members_are_enclosed(const std::string &b) {
    Rng rng(36);
    const Interval A = test::random_matrix(rng, 3, 0.2);
    const mat_exp::ExpmMixedResult R = mat_exp::expm_independent_mixed(A, 3, 9);
    const Interval total = test::interval_matrix(R.enclosure + R.enclosure_rest);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside = inside && total.contains(test::member(A, rng).expm(), 1e-12);
    check(inside, b + ": exp of members");
    check(R.powers.size() == 3 && R.powers2.size() == 6, b + ": sizes");
}

/// A matrix that is not square and an order below 1 are refused; with the checks off a valid call
/// gives the same.
void checks_and_their_switch() {
    check(test::raises(
              [&] {
                  mat_exp::expm_independent_mixed(
                      test::interval_matrix(test::matrix({1, 2, 3}, 1, 3)), 1, 3);
              },
              Error::dimension_mismatch),
          "expm_independent_mixed: not square");
    check(test::raises([&] { mat_exp::expm_independent_mixed(test::example(), 0, 3); },
                       Error::wrong_value),
          "expm_independent_mixed: intermediate order 0");
    const Tensor on = mat_exp::expm_independent_mixed(test::example(), 2, 4).enclosure.sup();
    ChecksScope off(false);
    check(close(mat_exp::expm_independent_mixed(test::example(), 2, 4).enclosure.sup(), on, 0),
          "expm_independent_mixed: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    // the remainder box uses exp, which the lean backend has not
    test::for_each_backend([](const std::string &b) {
        nilpotent_matrix(b);
        members_are_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_independent_mixed");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
