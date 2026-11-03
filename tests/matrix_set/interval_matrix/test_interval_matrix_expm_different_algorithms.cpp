// test_interval_matrix_expm_different_algorithms - IntervalMatrix expm_different_algorithms: by
// style
//
// CORA: matrixSet/@intervalMatrix/expmDifferentAlgorithms.m
// PROPERTY: matrixSet/@intervalMatrix/expmDifferentAlgorithms.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

bool same(const IntervalMatrix &a, const IntervalMatrix &b) {
    return close(a.interval().inf(), b.interval().inf(), 0) &&
           close(a.interval().sup(), b.interval().sup(), 0);
}

/// Styles 0 to 2 are expm_taylor_horner_scaling's, style 4 is the enclosure of expm at step 1.
void styles_pick_the_algorithm(const std::string &b) {
    Rng rng(40);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.2);
    for (const int style : {0, 1, 2}) {
        const IntervalMatrix E = *A.expm_different_algorithms(6, style, 2);
        const IntervalMatrix F = *A.expm_taylor_horner_scaling(6, style, 2);
        check(same(E, F), b + ": style " + std::to_string(style));
    }
    const IntervalMatrix E = *A.expm_different_algorithms(6, 4), F = A.expm(1.0, 6).enclosure;
    check(same(E, F), b + ": style 4 is expm");
}

/// Every style's result holds the exponentials of the members, and style 0 of -1 is [5/18, 13/18].
void results_enclose(const std::string &b) {
    Rng rng(41);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.2);
    bool inside = true;
    for (const int style : {0, 1, 2, 4}) {
        const std::optional<IntervalMatrix> E = A.expm_different_algorithms(7, style);
        for (int trial = 0; trial < 10; ++trial)
            inside = inside && E && E->contains(test::member(A, rng).expm(), "exact", 1e-12);
    }
    check(inside, b + ": exp of members");
    const IntervalMatrix E =
        *IntervalMatrix(test::matrix({-1}, 1, 1)).expm_different_algorithms(2, 0);
    check(close(E.interval().inf(), std::vector<double>{5.0 / 18}, 1e-12), b + ": the point -1");
}

/// A style that CORA has no case for is refused, as are a matrix that is not square; with the
/// checks off a valid call gives the same.
void checks_and_their_switch() {
    const IntervalMatrix A(test::matrix({0.5}, 1, 1));
    check(test::raises([&] { A.expm_different_algorithms(3, 3); }, Error::wrong_value), "style 3");
    check(
        test::raises(
            [&] { IntervalMatrix(test::matrix({1, 2, 3}, 1, 3)).expm_different_algorithms(3, 4); },
            Error::dimension_mismatch),
        "not square, style 4");
    const Tensor on = A.expm_different_algorithms(3, 4)->interval().sup();
    ChecksScope off(false);
    check(close(A.expm_different_algorithms(3, 4)->interval().sup(), on, 0),
          "the same with the checks off");
}

} // namespace

int main() {
    // style 4 is expm, whose remainder box uses exp, which the lean backend has not
    test::for_each_backend([](const std::string &b) {
        styles_pick_the_algorithm(b);
        results_enclose(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_different_algorithms");
}
