// test_interval_matrix_expm_independent_mixed - IntervalMatrix expm_independent_mixed: two parts
//
// CORA: matrixSet/@intervalMatrix/expmIndMixed.m
// PROPERTY: matrixSet/@intervalMatrix/expmIndMixed.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A = [[0, 1], [0, 0]] is nilpotent: the first part is I + A, the sum of both parts and the
/// remainder box encloses exp(A) = [[1, 1], [0, 1]]; the powers are those of powers().
void nilpotent_matrix(const std::string &b) {
    const IntervalMatrix A(test::matrix({0, 1, 0, 0}, 2, 2));
    const ExpmMixedResult R = A.expm_independent_mixed(2, 5);
    check(close(R.enclosure.interval().inf(), std::vector<double>{1, 1, 0, 1}) &&
              close(R.enclosure.interval().sup(), std::vector<double>{1, 1, 0, 1}),
          b + ": I + A + A^2 / 2");
    check(R.powers.size() == 2 && R.powers2.size() == 3, b + ": the powers of both parts");
    const IntervalMatrix total(R.enclosure.interval() + R.enclosure_rest.interval());
    check(total.contains(test::matrix({1, 1, 0, 1}, 2, 2), "exact", 1e-9), b + ": exp(A) is inside");
    const std::vector<IntervalMatrix> P = A.powers(2);
    check(close(R.powers[1].interval().sup(), P[1].interval().sup(), 0), b + ": the powers of powers()");
}

/// The exponentials of members are in the sum of both parts.
void members_are_enclosed(const std::string &b) {
    Rng rng(36);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.2);
    const ExpmMixedResult R = A.expm_independent_mixed(3, 9);
    const IntervalMatrix total(R.enclosure.interval() + R.enclosure_rest.interval());
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside = inside && total.contains(test::member(A, rng).expm(), "exact", 1e-12);
    check(inside, b + ": exp of members");
    check(R.powers.size() == 3 && R.powers2.size() == 6, b + ": sizes");
}

/// A matrix that is not square and an order below 1 are refused; with the checks off a valid call
/// gives the same.
void checks_and_their_switch() {
    check(test::raises([&] { IntervalMatrix(test::matrix({1, 2, 3}, 1, 3)).expm_independent_mixed(1, 3); },
                       Error::dimension_mismatch),
          "expm_independent_mixed: not square");
    check(test::raises([&] { test::example().expm_independent_mixed(0, 3); }, Error::wrong_value),
          "expm_independent_mixed: intermediate order 0");
    const Tensor on = test::example().expm_independent_mixed(2, 4).enclosure.interval().sup();
    ChecksScope off(false);
    check(close(test::example().expm_independent_mixed(2, 4).enclosure.interval().sup(), on, 0),
          "expm_independent_mixed: the same with the checks off");
}

} // namespace

int main() {
    // the remainder box uses exp, which the lean backend has not
    test::for_each_backend([](const std::string &b) {
        nilpotent_matrix(b);
        members_are_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_independent_mixed");
}
