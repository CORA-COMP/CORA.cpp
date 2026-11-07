// test_mat_zonotope_vertices - MatZonotope::vertices: the corner matrices of a matrix zonotope
//
// CORA: matrixSet/@matZonotope/vertices.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const MatZonotope M(test::matrix({1, 0, 0, 1}, 2, 2),
                            Tensor::from_data({1, 0, 0, 0, 0, 0, 0, 2}, {2, 2, 2}));
        const std::vector<Tensor> V = M.vertices();
        check(V.size() == 1 && V[0].shape() == Shape{4, 2, 2}, b + ": four matrices");
        // C +- G_1 +- G_2, ascending in the stacked columns
        check(close(V[0], {0, 0, 0, -1, 0, 0, 0, 3, 2, 0, 0, -1, 2, 0, 0, 3}, 0),
              b + ": the corners in order");
        const MatZonotope padded(M.center(),
                                 Tensor::concat({M.generators(), Tensor::zeros({1, 2, 2})}, 0));
        check(padded.vertices()[0].shape() == Shape{4, 2, 2},
              b + ": a zero generator adds no corner");
        const MatZonotope point(M.center());
        check(point.vertices()[0].shape() == Shape{1, 2, 2}, b + ": a point has itself");
    });
    check(test::raises(
              [] { MatZonotope(test::matrix({0}, 1, 1), Tensor::zeros({21, 1, 1})).vertices(); },
              Error::not_supported),
          "vertices: too many generators");
    ChecksScope off(false);
    check(test::example().vertices()[0].shape() == Shape{4, 2, 2},
          "vertices: the same with the checks off");
    return test::finish("mat_zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
