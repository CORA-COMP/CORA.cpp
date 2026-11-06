// test_mat_zonotope_expm_ind_mixed - MatZonotope::expm_ind_mixed
//
// CORA: matrixSet/@matZonotope/expmIndMixed.m
// PROPERTY: matrixSet/@matZonotope/expmIndMixed.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The exponential of every member is in the sum of the two parts' boxes, for several orders.
void expm_ind_mixed_encloses(const std::string &b) {
    Rng rng(83);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.3);
    for (const auto &[intermediate, maximum] :
         std::vector<std::pair<int, int>>{{1, 6}, {2, 8}, {3, 10}}) {
        const auto R = A.expm_ind_mixed(intermediate, maximum);
        const Interval total =
            R.enclosure.interval_matrix().interval() + R.enclosure_rest.interval();
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial)
            inside = inside && IntervalMatrix(total).contains(
                                   test::exponential(test::member(A, test::coefficients(rng, 2))),
                                   "exact", 1e-12);
        check(inside, b + ": exp(M) of every member is inside, intermediate order " +
                          std::to_string(intermediate));
        check(int(R.powers.size()) == intermediate &&
                  int(R.powers2.size()) == maximum - intermediate,
              b + ": the lists of powers");
    }
    // a point set: I + C + C^2 / 2 exactly in the first part
    const Tensor C = test::matrix({0.5, 0.1, -0.2, 0.3}, 2, 2);
    const auto R = MatZonotope(C).expm_ind_mixed(2, 5);
    check(close(R.enclosure.center(), C.eye_like(2) + C + C.matmul(C) * 0.5, 1e-14),
          b + ": the first part of a point is its Taylor polynomial");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(expm_ind_mixed_encloses);
    check(test::raises([] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).expm_ind_mixed(2, 5); },
                       Error::dimension_mismatch),
          "expm_ind_mixed: a non-square set");
    check(test::raises([] { test::example().expm_ind_mixed(0, 5); }, Error::wrong_value),
          "expm_ind_mixed: order 0");
    check(test::raises([] { test::example().expm_ind_mixed(2, -1); }, Error::wrong_value),
          "expm_ind_mixed: a negative maximum");
    ChecksScope off(false);
    check(test::example().expm_ind_mixed(2, 4).powers.size() == 2,
          "expm_ind_mixed: the same with the checks off");
    return test::finish("mat_zonotope expm_ind_mixed");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
