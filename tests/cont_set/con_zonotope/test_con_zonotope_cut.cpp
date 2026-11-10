// test_con_zonotope_cut - ConZonotope::cut: the rows of a polytope kept on the factors
//
// A square cut by a halfspace, by a hyperplane with a halfspace, and by a polytope beyond it, by
// hand. On random sets a point of the set that lies in the polytope is in the box of the cut, and
// the cut of an image is the image of the cut (the stated properties).
//
// CORA.cpp only: tests cut, CORALean's float replacement of the slack factors of polytope/and_
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/polytope/polytope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// [-1, 1]^2 cut by x1 <= 0.5, by x2 = 0.5 with x1 <= 0, and by x1 <= -3.
void by_hand(const std::string &b) {
    const ConZonotope square(Zonotope(column({0.0, 0.0}), Tensor::eye(2)));
    const Interval half =
        square.cut(Polytope(Tensor({{1.0, 0.0}}), column({0.5}))).interval_tight(3);
    check(close(half.inf(), {-1, -1}, 0) && close(half.sup(), {0.5, 1}, 0), b + ": a halfspace");
    const Polytope line(Tensor({{1.0, 0.0}}), column({0.0}), Tensor({{0.0, 1.0}}), column({0.5}));
    const Interval J = square.cut(line).interval_tight(3);
    check(close(J.inf(), {-1, 0.5}, 0) && close(J.sup(), {0, 0.5}, 0),
          b + ": a hyperplane with a halfspace");
    const Polytope far(Tensor({{1.0, 0.0}}), column({-3.0}));
    check(square.cut(far).interval_tight(3).is_empty(), b + ": beyond the set");
    // the rows are kept as rows: no factor is added
    check(square.cut(line).G.shape()[1] == 2 && square.cut(line).C.shape()[0] == 3,
          b + ": the rows stay rows (an equality is two)");
}

/// Stated properties: a point of the set in the polytope is in the box; the image commutes.
void properties(const std::string &b) {
    cora::Rng rng(17);
    for (int trial = 0; trial < 4; ++trial) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1);
        const Polytope P(Tensor({{1.0, 0.5}, {-0.3, 1.0}}), A.c() + column({0.2, 0.1}));
        const ConZonotope::Cut K = A.cut(P);
        const Interval I = K.interval_tight(5);
        const Tensor points = test::sample_points(A, rng, 80);
        for (int64_t j = 0; j < points.shape()[1]; ++j) {
            const Tensor p = points.slice(1, j, j + 1);
            if (P.contains(p, "exact", 0.0))
                check(!I.is_empty() && test::inside_box(p, I), b + ": a point of both is boxed");
        }
        // the image under M: its box holds M p
        const Tensor M = Tensor({{0.5, 1.0}, {-1.0, 0.2}});
        const Interval IM = K.linear_map(M).interval_tight(5);
        for (int64_t j = 0; j < points.shape()[1]; ++j) {
            const Tensor p = points.slice(1, j, j + 1);
            if (P.contains(p, "exact", 0.0))
                check(!IM.is_empty() && test::inside_box(M.matmul(p), IM),
                      b + ": the image of a point of both is boxed");
        }
    }
}

/// Another dimension and a wrong matrix are refused.
void refusals() {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    check(test::raises([&] { A.cut(Polytope(Tensor({{1.0, 0.0}}), column({0.0}))); },
                       Error::dimension_mismatch),
          "cut: different dimensions");
    const ConZonotope::Cut K = A.cut(Polytope(Tensor({{1.0}}), column({0.0})));
    check(test::raises([&] { K.linear_map(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "cut: the matrix needs the columns of the dimension");
    const Tensor on = K.interval_tight(2).sup();
    ChecksScope off(false);
    check(close(A.cut(Polytope(Tensor({{1.0}}), column({0.0}))).interval_tight(2).sup(), on, 0),
          "cut: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        by_hand(b);
        properties(b);
    });
    refusals();
    return test::finish("con_zonotope cut");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
