// test_zonotope_mat_zonotope - Zonotope::mat_zonotope: a zonotope as a matrix zonotope
//
// CORA: contSet/@zonotope/matZonotope.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Zonotope Z(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}, {3.0, 4.0}}));
        const Zonotope M = Z.mat_zonotope();
        check(M.center().size(-2) == 2 && M.center().size(-1) == 1 && M.nr_generators() == 3,
              b + ": n x 1 matrices, one generator each");
        check(close(M.center(), {1, 2}, 0) && close(M.generators(), {1, 0, 0, 2, 3, 4}, 0),
              b + ": the center and the generators");
        const Zonotope back = M.stacked();
        check(close(back.center(), Z.center(), 0) && close(back.generators(), Z.generators(), 0),
              b + ": the stacked zonotope is the zonotope");
    });
    ChecksScope off(false);
    check(Zonotope(vec({0.0}), Tensor({{1.0}})).mat_zonotope().center().size(-2) == 1,
          "mat_zonotope: the same with the checks off");
    return test::finish("zonotope mat_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
