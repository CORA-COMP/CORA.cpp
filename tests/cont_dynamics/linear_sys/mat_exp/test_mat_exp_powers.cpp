// test_mat_exp_powers - Interval powers: the powers from a given order, from the
//
// right
//
// CORA: matrixSet/@intervalMatrix/powers.m
// PROPERTY: matrixSet/@intervalMatrix/powers.m
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

bool same(const Interval &a, const Interval &b) {
    return close(a.inf(), b.inf(), 0) && close(a.sup(), b.sup(), 0);
}

/// The default start is [A]^1 = A, each power the previous times A; by hand A^2 of the example is
/// [-2, 7], [-4, 4], [0, 12], [1, 7].
void powers_of_the_example(const std::string &b) {
    const Interval A = test::example();
    const std::vector<Interval> P = mat_exp::powers(A, 3);
    check(P.size() == 3 && same(P[0], A), b + ": the first power is A");
    check(close(P[1].inf(), std::vector<double>{-2, -4, 0, 1}) &&
              close(P[1].sup(), std::vector<double>{7, 4, 12, 7}),
          b + ": A^2");
    check(same(P[2], P[1].matmul(A)), b + ": A^3 is A^2 times A, from the right");
    check(mat_exp::powers(A, 1).size() == 1 && mat_exp::powers(A, 0).size() == 1,
          b + ": at least the initial power");
}

/// From the order of a given power, and the powers of members are in the powers of the set.
void powers_from_an_initial_power(const std::string &b) {
    Rng rng(14);
    const Interval A = test::random_matrix(rng, 3, 0.2), X = test::random_matrix(rng, 3, 0.2);
    const std::vector<Interval> P = mat_exp::powers(A, 5, 3, X);
    check(P.size() == 3 && same(P[0], X) && same(P[1], X.matmul(A)) && same(P[2], P[1].matmul(A)),
          b + ": X, X A, X A A");
    const std::vector<Interval> Q = mat_exp::powers(A, 4);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial) {
        const Tensor M = test::member(A, rng);
        Tensor power = M;
        for (const Interval &q : Q) {
            inside = inside && q.contains(power, 1e-12);
            power = power.matmul(M);
        }
    }
    check(inside, b + ": members' powers are inside");
}

/// An initial order below 1 and factors that do not multiply are refused; with the checks off a
/// valid call gives the same.
void checks_and_their_switch() {
    const Interval A = test::example();
    check(test::raises([&] { mat_exp::powers(A, 3, 0, A); }, Error::wrong_value),
          "powers: initial order 0");
    const Interval wide = test::interval_matrix(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { mat_exp::powers(wide, 2); }, Error::dimension_mismatch),
          "powers: not square");
    const Tensor on = mat_exp::powers(A, 3)[2].infimum();
    ChecksScope off(false);
    check(close(mat_exp::powers(A, 3)[2].infimum(), on, 0), "powers: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        powers_of_the_example(b);
        powers_from_an_initial_power(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix powers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
