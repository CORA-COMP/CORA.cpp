// test_mat_exp_zonotope_powers - the powers of a matrix zonotope from an initial one
//
// CORA: matrixSet/@matZonotope/powers.m
// PROPERTY: matrixSet/@matZonotope/powers.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// From A the entries are A, A A, (A A) A: each is the previous one times A from the right.
void powers_from_the_set(const std::string &b) {
    const Zonotope A = test::example();
    const std::vector<Zonotope> P = mat_exp::powers(A, 3);
    check(P.size() == 3, b + ": the orders 1 to 3");
    // each power is the previous one times A
    check(test::same_set(P[0], A, 0), b + ": the first is A");
    check(test::same_set(P[1], A.matmul(A), 0), b + ": the second is A A");
    check(test::same_set(P[2], A.matmul(A).matmul(A), 0), b + ": the third is (A A) A");
}

/// From an initial power the entry k has the order initial_order + k and is that power times A^k
/// from the right; below the initial order only the initial power remains, as CORA's cell.
void powers_from_an_initial_power(const std::string &b) {
    const Zonotope A = test::example();
    const Zonotope square = A.matmul(A);
    const std::vector<Zonotope> P = mat_exp::powers(A, 4, 2, square);
    check(P.size() == 3 && test::same_set(P[0], square, 0), b + ": the orders 2 to 4 from A A");
    check(test::same_set(P[1], square.matmul(A), 0), b + ": the order 3");
    check(test::same_set(P[2], square.matmul(A).matmul(A), 0), b + ": the order 4");
    check(mat_exp::powers(A, 1, 2, square).size() == 1,
          b + ": the maximum below the initial order");
    // the powers enclose the powers of the members
    Rng rng(41);
    const Zonotope Q = test::random_set(rng, 3, 2, 0.4);
    const std::vector<Zonotope> R = mat_exp::powers(Q, 4);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial) {
        const Tensor M = test::member(Q, test::coefficients(rng, 2));
        Tensor power = M;
        for (std::size_t k = 0; k < R.size(); ++k, power = power.matmul(M))
            inside = inside && R[k].interval().contains(power, 1e-12);
    }
    check(inside, b + ": the powers of every member are inside");
}

/// A non-square set, an initial power of another size and an initial order of 0 are refused; with
/// the checks off the same list.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    const Zonotope wide = test::point_set(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { mat_exp::powers(wide, 2); }, Error::dimension_mismatch),
          "powers: a non-square set");
    check(test::raises([&] { mat_exp::powers(A, 2, 1, wide); }, Error::dimension_mismatch),
          "powers: an initial power of another size");
    check(test::raises([&] { mat_exp::powers(A, 2, 0, A); }, Error::wrong_value),
          "powers: the initial order 0");
    const std::size_t on = mat_exp::powers(A, 3).size();
    ChecksScope off(false);
    check(mat_exp::powers(A, 3).size() == on, "powers: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        powers_from_the_set(b);
        powers_from_an_initial_power(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope powers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
