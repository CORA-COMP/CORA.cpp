// test_polytope_select_cols - Polytope::select_cols: the rows in some of the coordinates
//
// The columns of the square reordered and dropped, of equalities, none at all, and of a batch.
// The stated property: y is in P.select_cols(cols) exactly if the point x with x_cols[l] = y_l
// and zeros elsewhere is in P.
//
// CORA.cpp only: tests CORALean's HPolytope.selectCols
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

/// The columns of the square swapped and one dropped, of equalities, and of a batch.
void rows(const std::string &b) {
    const Polytope swapped = test::square_h().select_cols({1, 0});
    check(close(swapped.A(), {0, 1, 1, 0, 0, -1, -1, 0}, 0.0) &&
              close(swapped.b(), {1, 1, 0, 0}, 0.0),
          b + ": swapped columns");
    check(close(test::square_h().select_cols({1}).A(), {0, 1, 0, -1}, 0.0),
          b + ": a dropped column");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 2.0, 3.0}}), vec({1.0}));
    check(close(E.select_cols({2, 0}).Ae(), {3, 1}, 0.0) && E.select_cols({}).dim() == 0,
          b + ": equalities, and no column");
    const Polytope B(Tensor::ones({2, 1, 3}), Tensor::ones({2, 1}));
    check(B.select_cols({0, 2}).A().shape() == Shape{2, 1, 2}, b + ": a batch");
}

/// y in P.select_cols(cols) exactly if x (x_cols[l] = y_l, zeros elsewhere) in P.
void properties(const std::string &b) {
    Rng rng(10);
    const Polytope P = test::random_polytope(rng, 3, 4), S = P.select_cols({2, 0});
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const std::vector<double> y = test::random_direction(rng, 2);
        same = same && S.contains(vec(y)) == P.contains(vec({y[1], 0.0, y[0]}));
    }
    check(same, b + ": the section at 0");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().select_cols({2}); }, Error::wrong_value),
          "select_cols: a position beyond the dimension");
    ChecksScope off(false);
    check(test::square_h().select_cols({0}).dim() == 1,
          "select_cols: the same with the checks off");
    return test::finish("polytope select_cols");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
