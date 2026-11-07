// test_zonotope_mat_zonotope - Zonotope::mat_zonotope: a zonotope as a matrix zonotope
//
// CORA: contSet/@zonotope/matZonotope.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 3.0}, {0.0, 2.0, 4.0}}));
        const MatZonotope M = Z.mat_zonotope();
        check(M.rows() == 2 && M.cols() == 1 && M.num_generators() == 3,
              b + ": n x 1 matrices, one generator each");
        check(close(M.center(), {1, 2}, 0) && close(M.generators(), {1, 0, 0, 2, 3, 4}, 0),
              b + ": the center and the generators");
        const Zonotope back = M.zonotope();
        check(close(back.center(), Z.center(), 0) && close(back.generators(), Z.generators(), 0),
              b + ": the stacked zonotope is the zonotope");
    });
    ChecksScope off(false);
    check(Zonotope(column({0.0}), Tensor({{1.0}})).mat_zonotope().rows() == 1,
          "mat_zonotope: the same with the checks off");
    return test::finish("zonotope mat_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
