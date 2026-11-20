// test_zonotope_layout - the batch-first layout: generator rows, matrix zonotopes, broadcasting
//
// CORA.cpp only: CORA stores the generators as columns; CORA.cpp stores them batch first (#163).
// Authors:       Claude Opus 5.5
// Written:       18-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators are rows: G (h, n); the center is a vector (n), or a column that becomes one.
void rows_and_center(const std::string &b) {
    const Zonotope Z(test::vec({1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}));
    check(Z.dim() == 2 && Z.nr_generators() == 3 && Z.elem_rank() == 1,
          b + ": two dimensions, three generators");
    check(Z.center().shape() == Shape{2} && Z.generators().shape() == Shape{3, 2},
          b + ": center (2), generators (3, 2)");
    const Zonotope C(Tensor({1.0, 0.0}), Z.generators());
    check(close(C.center(), Z.center(), 0), b + ": a column center is the vector");
    // the box: the radius sums the rows
    check(close(Z.interval().sup(), test::vec({3.0, 2.0}), 0), b + ": the box sums |G| over h");
}

/// A matrix zonotope: C (r, c), G (h, r, c); M @ X, X @ M, plus, reduce and the box.
void matrix_zonotope(const std::string &b) {
    const Zonotope A =
        Zonotope::matrix(Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                         Tensor::from_data({0.1, 0, 0, 0, 0, 0.2, 0.2, 0}, {2, 2, 2}));
    check(A.elem_rank() == 2 && A.elem_shape() == Shape{2, 2} && A.nr_generators() == 2 &&
              A.dim() == 4,
          b + ": elements (2, 2), two generators");
    // M @ X and X @ M multiply every generator matrix
    const Tensor M({{2.0, 0.0}, {1.0, 1.0}});
    const Zonotope left = matmul(M, A), right = matmul(A, M);
    check(close(left.center(), M, 0) && close(right.center(), M, 0), b + ": M I = I M = M");
    check(close(left.generators(), M.matmul(A.generators()), 0) &&
              close(right.generators(), A.generators().matmul(M), 0),
          b + ": every generator multiplied");
    // the sum, the box, the reduction and the scaling keep the matrix shape
    const Zonotope S = A + A;
    check(S.elem_rank() == 2 && S.nr_generators() == 4, b + ": the sum joins the generators");
    const Interval I = A.interval();
    check(close(I.sup(), Tensor({{1.1, 0.2}, {0.2, 1.0}}), 1e-15), b + ": the box of the matrices");
    const Zonotope R = (A + A + A).reduce(1);
    check(R.elem_rank() == 2 && R.nr_generators() <= 4 && R.generators().shape()[1] == 2,
          b + ": reduced to order 1, the shape kept");
    check((2.0 * A).elem_rank() == 2 && close((2.0 * A).generators(), A.generators() * 2.0, 0),
          b + ": scaling");
}

/// x @ M for a vector zonotope is M' x; batch dims of the map broadcast.
void maps_and_broadcasting(const std::string &b) {
    const Zonotope Z(test::vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.5, 0.5}}));
    const Tensor M({{1.0, 2.0}, {3.0, 4.0}});
    check(close(matmul(Z, M).center(),
                M.matrix_transpose().matmul(test::column({1.0, 2.0})).squeeze(-1), 0),
          b + ": x @ M is M' x");
    const Tensor Ms = Tensor::stack({M, M * 2.0}); // (2, 2, 2): a batch of maps
    const Zonotope B = matmul(Ms, Z);
    check(B.batch_shape() == Shape{2} && B.generators().shape() == Shape{2, 2, 2},
          b + ": a batch of maps gives a batch of sets");
    const Zonotope batch = Zonotope::stack({Z, Z});
    const Zonotope sum = batch + Z; // the unbatched one broadcasts
    check(sum.batch_shape() == Shape{2} && sum.nr_generators() == 4, b + ": batch (2) + one set");
}

/// Shapes that do not fit are refused, naming both.
void refusals() {
    check(test::raises([] { Zonotope(test::vec({1.0, 0.0}), Tensor({{1.0, 0.0, 0.0}})); },
                       Error::wrong_input_in_constructor),
          "the generator rows must have n entries");
    check(test::raises([] { Zonotope(test::column({1.0, 0.0}), Tensor({{1.0, 0.0}})); },
                       Error::wrong_input_in_constructor),
          "CORA's column (n, 1) is no center (#163)");
    check(test::raises([] { Zonotope::matrix(Tensor({{1.0, 0.0}}), Tensor({{1.0, 0.0}})); },
                       Error::wrong_input_in_constructor),
          "a matrix zonotope needs (h, r, c) generators");
    const Zonotope A = Zonotope::matrix(Tensor({{1.0}}), Tensor::from_data({1.0}, {1, 1, 1}));
    check(test::raises([&] { A.box(); }, Error::not_supported), "box: not for a matrix zonotope");
    ChecksScope off(false);
    check(Zonotope(test::vec({1.0}), Tensor({{2.0}})).nr_generators() == 1,
          "the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows_and_center(b);
        matrix_zonotope(b);
        maps_and_broadcasting(b);
    });
    refusals();
    return test::finish("zonotope layout");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
