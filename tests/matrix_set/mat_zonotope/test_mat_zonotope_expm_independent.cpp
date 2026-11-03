// test_mat_zonotope_expm_independent - MatZonotope expm_independent: the exponential with independent powers
//
// CORA: matrixSet/@matZonotope/expmInd.m
// PROPERTY: matrixSet/@matZonotope/expmInd.m

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A point matrix is its exponential: the nilpotent [[0, 1], [0, 0]] gives [[1, 1], [0, 1]] and the
/// zero matrix the identity, with the remainder box of width about zero.
void points(const std::string &b) {
    const MatZonotope nilpotent(test::matrix({0, 1, 0, 0}, 2, 2)), zero(test::matrix({0, 0, 0, 0}, 2, 2));
    const IntervalMatrix E = nilpotent.expm_independent(5).interval_matrix();
    check(close(E.center(), {1, 1, 0, 1}, 1e-12) && E.interval().rad().max().item() < 1e-12,
          b + ": exp of a nilpotent matrix");
    const IntervalMatrix I = zero.expm_independent(3).interval_matrix();
    check(close(I.center(), {1, 0, 0, 1}, 1e-12) && I.interval().rad().max().item() < 1e-12,
          b + ": exp(0) = I");
    const MatZonotope low = zero.expm_independent(0);
    check(close(low.center(), {1, 0, 0, 1}), b + ": the order 0 is the identity");
}

/// The exponential of every member is in the box of the result, which has a generator per power
/// term and per entry of the remainder; a higher order does not widen it much.
void expm_encloses_the_exponentials(const std::string &b) {
    Rng rng(61);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.3);
    for (const int64_t order : {4, 8}) {
        const IntervalMatrix box = A.expm_independent(order).interval_matrix();
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial)
            inside = inside && box.contains(
                                   test::exponential(test::member(A, test::coefficients(rng, 2))),
                                   "exact", 1e-12);
        check(inside, b + ": exp(M) of every member is inside, order " + std::to_string(order));
    }
    const MatZonotope order2 = A.expm_independent(2);
    check(order2.num_generators() >= 9, b + ": the remainder adds a generator per entry");
    const MatZonotope batch = test::batch_of(A, A).expm_independent(4);
    check(test::same_set(test::batch_member(batch, 1), A.expm_independent(4), 1e-12), b + ": a batch");
}

/// A non-square set and a negative order are refused; with the checks off the same result.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).expm_independent(3); },
                       Error::dimension_mismatch),
          "expm_independent: a non-square set");
    check(test::raises([&] { A.expm_independent(-1); }, Error::wrong_value),
          "expm_independent: a negative order");
    const Tensor on = A.expm_independent(3).center();
    ChecksScope off(false);
    check(close(A.expm_independent(3).center(), on, 0), "expm_independent: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        expm_encloses_the_exponentials(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope expm_independent");
}
