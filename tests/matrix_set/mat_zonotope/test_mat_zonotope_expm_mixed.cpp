// test_mat_zonotope_expm_mixed - MatZonotope expm_mixed: the exponential in two parts
//
// CORA: matrixSet/@matZonotope/expmMixed.m
// PROPERTY: matrixSet/@matZonotope/expmMixed.m

#include "mat_zonotope_testing.h"

#include <tuple>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The exponential of every member is in the sum of the two parts' boxes, for several orders and
/// steps; the lists have the powers of orders 1 to the intermediate order and of the rest.
void expm_mixed_encloses_the_exponentials(const std::string &b) {
    Rng rng(81);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.3);
    const std::vector<std::tuple<double, int, int>> cases{{1.0, 2, 8}, {0.5, 3, 9}, {2.0, 4, 10}};
    for (const auto &[r, intermediate, maximum] : cases) {
        const auto R = A.expm_mixed(r, intermediate, maximum);
        const Interval total =
            R.enclosure.interval_matrix().interval() + R.enclosure_rest.interval();
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial)
            inside = inside && IntervalMatrix(total).contains(
                                   test::exponential(test::member(A, test::coefficients(rng, 2))),
                                   "exact", 1e-12);
        check(inside, b + ": exp(M) of every member is inside, r = " + std::to_string(r));
        check(int(R.powers.size()) == intermediate &&
                  int(R.powers2.size()) == maximum - intermediate,
              b + ": the lists of powers");
    }
    const auto R = A.expm_mixed(1.0, 2, 6);
    check(test::same_set(R.powers[1], A.dependent_terms(1.0).first, 1e-12) &&
              test::same_set(R.powers[0], A, 0),
          b + ": the powers start with A and the square");
    check(test::same_set(R.enclosure, A.dependent_terms(1.0).second, 1e-12),
          b + ": the first part of the order 2 is the dependent terms");
    const auto batch = test::batch_of(A, A).expm_mixed(1.0, 3, 6);
    check(test::same_set(test::batch_member(batch.enclosure, 1), A.expm_mixed(1.0, 3, 6).enclosure,
                         1e-12),
          b + ": a batch");
}

/// A non-square set, a step of 0 or less, an intermediate order below 2 and a negative maximum are
/// refused; with the checks off the same result.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).expm_mixed(1.0, 2, 5); },
                       Error::dimension_mismatch),
          "expm_mixed: a non-square set");
    check(test::raises([&] { A.expm_mixed(0.0, 2, 5); }, Error::wrong_value),
          "expm_mixed: the step 0");
    check(test::raises([&] { A.expm_mixed(1.0, 1, 5); }, Error::wrong_value),
          "expm_mixed: the order 1");
    check(test::raises([&] { A.expm_mixed(1.0, 2, -1); }, Error::wrong_value),
          "expm_mixed: a negative maximum");
    const Tensor on = A.expm_mixed(1.0, 2, 4).enclosure.center();
    ChecksScope off(false);
    check(close(A.expm_mixed(1.0, 2, 4).enclosure.center(), on, 0),
          "expm_mixed: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(expm_mixed_encloses_the_exponentials);
    checks_and_their_switch();
    return test::finish("mat_zonotope expm_mixed");
}
