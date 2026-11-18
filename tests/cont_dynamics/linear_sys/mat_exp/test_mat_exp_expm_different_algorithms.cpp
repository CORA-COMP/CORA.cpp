// test_mat_exp_expm_different_algorithms - Interval expm_different_algorithms: by
//
// style
//
// CORA: matrixSet/@intervalMatrix/expmDifferentAlgorithms.m
// PROPERTY: matrixSet/@intervalMatrix/expmDifferentAlgorithms.m
// CORA authors: Ivan Brkan
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

bool same(const Interval &a, const Interval &b) {
    return close(a.inf(), b.inf(), 0) && close(a.sup(), b.sup(), 0);
}

/// Styles 0 to 2 are expm_taylor_horner_scaling's, style 4 is the enclosure of expm at step 1.
void styles_pick_the_algorithm(const std::string &b) {
    Rng rng(40);
    const Interval A = test::random_matrix(rng, 3, 0.2);
    for (const int style : {0, 1, 2}) {
        const Interval E = *mat_exp::expm_different_algorithms(A, 6, style, 2);
        const Interval F = *mat_exp::expm_taylor_horner_scaling(A, 6, style, 2);
        check(same(E, F), b + ": style " + std::to_string(style));
    }
    const Interval E = *mat_exp::expm_different_algorithms(A, 6, 4),
                   F = mat_exp::expm(A, 1.0, 6).enclosure;
    check(same(E, F), b + ": style 4 is expm");
}

/// Every style's result holds the exponentials of the members, and style 0 of -1 is [5/18, 13/18].
void results_enclose(const std::string &b) {
    Rng rng(41);
    const Interval A = test::random_matrix(rng, 3, 0.2);
    bool inside = true;
    for (const int style : {0, 1, 2, 4}) {
        const std::optional<Interval> E = mat_exp::expm_different_algorithms(A, 7, style);
        for (int trial = 0; trial < 10; ++trial)
            inside = inside && E && E->contains(test::member(A, rng).expm(), 1e-12);
    }
    check(inside, b + ": exp of members");
    const Interval E =
        *mat_exp::expm_different_algorithms(test::interval_matrix(test::matrix({-1}, 1, 1)), 2, 0);
    check(close(E.inf(), std::vector<double>{5.0 / 18}, 1e-12), b + ": the point -1");
}

/// A style that CORA has no case for is refused, as are a matrix that is not square; with the
/// checks off a valid call gives the same.
void checks_and_their_switch() {
    const Interval A = test::interval_matrix(test::matrix({0.5}, 1, 1));
    check(test::raises([&] { mat_exp::expm_different_algorithms(A, 3, 3); }, Error::wrong_value),
          "style 3");
    check(test::raises(
              [&] {
                  mat_exp::expm_different_algorithms(
                      test::interval_matrix(test::matrix({1, 2, 3}, 1, 3)), 3, 4);
              },
              Error::dimension_mismatch),
          "not square, style 4");
    const Tensor on = mat_exp::expm_different_algorithms(A, 3, 4)->interval().sup();
    ChecksScope off(false);
    check(close(mat_exp::expm_different_algorithms(A, 3, 4)->interval().sup(), on, 0),
          "the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    // style 4 is expm, whose remainder box uses exp, which the lean backend has not
    test::for_each_backend([](const std::string &b) {
        styles_pick_the_algorithm(b);
        results_enclose(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_different_algorithms");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
