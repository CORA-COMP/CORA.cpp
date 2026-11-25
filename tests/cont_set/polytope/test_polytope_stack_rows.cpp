// test_polytope_stack_rows - Polytope::stack_rows: the rows of two polytopes, all kept
//
// The inequalities and the equalities of both sets in order, a row that holds everywhere kept
// (intersect drops such a set); a batch stacks member by member, and the empty flag of either
// set stays. The stated property: a point is in the stacked set exactly if it is in both.
//
// CORA.cpp only: tests CORALean's HPolytope.stackRows
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of the square with equalities, and with the trivial row 0 = 0.
void rows(const std::string &b) {
    const Polytope E = Polytope::equalities(Tensor({{1.0, -1.0}}), vec({0.5}));
    const Polytope P = test::square_h().stack_rows(E);
    check(close(P.b(), {1, 1, 0, 0}, 0.0) && close(P.Ae(), {1, -1}, 0.0) &&
              close(P.be(), {0.5}, 0.0),
          b + ": inequalities, then equalities");
    // 0 = 0 holds everywhere and stays a row
    const Polytope zero = Polytope::equalities(Tensor({{0.0, 0.0}}), vec({0.0}));
    check(E.stack_rows(zero).be().numel() == 2, b + ": the trivial row is kept");
    check(test::square_h().stack_rows(Polytope::empty(2)).empty_mask().all(),
          b + ": the empty flag of either set");
    // a batch stacks member by member
    const Polytope B(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1}));
    check(B.stack_rows(B).A().shape() == Shape{2, 2, 2}, b + ": a batch");
}

/// x in P.stack_rows(S) exactly if x in P and x in S.
void properties(const std::string &b) {
    Rng rng(9);
    const Polytope P = test::random_polytope(rng, 2, 3), S = test::random_polytope(rng, 2, 3);
    const Polytope both = P.stack_rows(S);
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const Tensor x = vec(test::random_direction(rng, 2)) * 1.5;
        same = same && both.contains(x) == (P.contains(x) && S.contains(x));
    }
    check(same, b + ": the intersection");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().stack_rows(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "stack_rows: the sets differ in dimension");
    check(test::raises(
              [] {
                  const Polytope B(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1}));
                  test::square_h().stack_rows(B);
              },
              Error::dimension_mismatch),
          "stack_rows: the sets differ in batch shape");
    ChecksScope off(false);
    check(test::square_h().stack_rows(test::square_h()).b().numel() == 8,
          "stack_rows: the same with the checks off");
    return test::finish("polytope stack_rows");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
