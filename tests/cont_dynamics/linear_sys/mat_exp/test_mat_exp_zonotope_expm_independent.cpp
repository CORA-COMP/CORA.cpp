// test_mat_exp_zonotope_expm_independent - the exponential of a matrix set, independent powers
//
// CORA: matrixSet/@matZonotope/expmInd.m
// PROPERTY: matrixSet/@matZonotope/expmInd.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A point matrix is its exponential: the nilpotent [[0, 1], [0, 0]] gives [[1, 1], [0, 1]] and the
/// zero matrix the identity, with the remainder box of width about zero.
void points(const std::string &b) {
    const Zonotope nilpotent = test::point_set(test::matrix({0, 1, 0, 0}, 2, 2)),
                   zero = test::point_set(test::matrix({0, 0, 0, 0}, 2, 2));
    const Interval E = mat_exp::expm_independent(nilpotent, 5).interval();
    check(close(E.center(), {1, 1, 0, 1}, 1e-12) && E.interval().half_width().max().item() < 1e-12,
          b + ": exp of a nilpotent matrix");
    const Interval I = mat_exp::expm_independent(zero, 3).interval();
    check(close(I.center(), {1, 0, 0, 1}, 1e-12) && I.interval().half_width().max().item() < 1e-12,
          b + ": exp(0) = I");
    const Zonotope low = mat_exp::expm_independent(zero, 0);
    check(close(low.center(), {1, 0, 0, 1}), b + ": the order 0 is the identity");
}

/// The exponential of every member is in the box of the result, which has a generator per power
/// term and per entry of the remainder; a higher order does not widen it much.
void expm_encloses_the_exponentials(const std::string &b) {
    Rng rng(61);
    const Zonotope A = test::random_set(rng, 3, 2, 0.3);
    for (const int64_t order : {4, 8}) {
        const Interval box = mat_exp::expm_independent(A, order).interval();
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial)
            inside =
                inside &&
                box.contains(test::exponential(test::member(A, test::coefficients(rng, 2))), 1e-12);
        check(inside, b + ": exp(M) of every member is inside, order " + std::to_string(order));
    }
    const Zonotope order2 = mat_exp::expm_independent(A, 2);
    check(order2.nr_generators() >= 9, b + ": the remainder adds a generator per entry");
    const Zonotope batch = mat_exp::expm_independent(test::batch_of(A, A), 4);
    check(test::same_set(test::batch_member(batch, 1), mat_exp::expm_independent(A, 4), 1e-12),
          b + ": a batch");
}

/// A non-square set and a negative order are refused; with the checks off the same result.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    check(test::raises(
              [&] { mat_exp::expm_independent(test::point_set(test::matrix({1, 2, 3}, 1, 3)), 3); },
              Error::dimension_mismatch),
          "expm_independent: a non-square set");
    check(test::raises([&] { mat_exp::expm_independent(A, -1); }, Error::wrong_value),
          "expm_independent: a negative order");
    const Tensor on = mat_exp::expm_independent(A, 3).center();
    ChecksScope off(false);
    check(close(mat_exp::expm_independent(A, 3).center(), on, 0),
          "expm_independent: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        expm_encloses_the_exponentials(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope expm_independent");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
