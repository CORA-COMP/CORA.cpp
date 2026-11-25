// test_polytope_select_rows - Polytope::select_rows: some of the rows of a polytope
//
// Rows of the square kept in a new order, none kept (R^n), equality rows of a batch, and the
// empty flag, which stays only if every row is kept. The stated property: the kept rows hold
// every point of P.
//
// CORA.cpp only: tests CORALean's HPolytope.restrict
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

/// Rows of the square, of equalities and of a batch, and the empty flag.
void rows(const std::string &b) {
    const Polytope P = test::square_h().select_rows({3, 0}, {});
    check(close(P.A(), {0, -1, 1, 0}, 0.0) && close(P.b(), {0, 1}, 0.0), b + ": two rows");
    check(test::square_h().select_rows({}, {}).b().numel() == 0, b + ": no row");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 0.0}, {0.0, 1.0}}), vec({1.0, 2.0}));
    check(close(E.select_rows({}, {1}).Ae(), {0, 1}, 0.0) &&
              close(E.select_rows({}, {1}).be(), {2}, 0.0),
          b + ": an equality row");
    const Polytope B(Tensor::ones({2, 3, 2}), Tensor::ones({2, 3}));
    check(B.select_rows({0, 2}, {}).A().shape() == Shape{2, 2, 2}, b + ": a batch");
    const Polytope empty = Polytope::empty(2);
    check(empty.select_rows({0}, {}).empty_mask().all() &&
              !empty.select_rows({}, {}).empty_mask().any(),
          b + ": the empty flag stays only with every row");
}

/// Every point of P satisfies the kept rows.
void properties(const std::string &b) {
    Rng rng(11);
    const Polytope P = test::random_polytope(rng, 2, 4), S = P.select_rows({0, 2, 5}, {});
    bool contained = true;
    for (int k = 0; k < 30; ++k) {
        const Tensor x = vec(test::random_direction(rng, 2)) * 2.0;
        contained = contained && (!P.contains(x) || S.contains(x));
    }
    check(contained, b + ": the kept rows contain P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().select_rows({4}, {}); }, Error::wrong_value),
          "select_rows: a row beyond the inequalities");
    check(test::raises([] { test::square_h().select_rows({}, {0}); }, Error::wrong_value),
          "select_rows: an equality row the set does not have");
    ChecksScope off(false);
    check(test::square_h().select_rows({1}, {}).b().numel() == 1,
          "select_rows: the same with the checks off");
    return test::finish("polytope select_rows");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
