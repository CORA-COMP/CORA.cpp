// test_zonotope_vertex_matrices - the corner matrices of a matrix zonotope
//
// CORA: matrixSet/@matZonotope/vertices.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Zonotope M = Zonotope::matrix(test::matrix({1, 0, 0, 1}, 2, 2),
                                            Tensor::from_data({1, 0, 0, 0, 0, 0, 0, 2}, {2, 2, 2}));
        // two generators: four corner matrices
        const std::vector<Tensor> V = M.vertex_matrices();
        check(V.size() == 1 && V[0].shape() == Shape{4, 2, 2}, b + ": four matrices");
        // C +- G_1 +- G_2, ascending in the stacked columns
        check(close(V[0], {0, 0, 0, -1, 0, 0, 0, 3, 2, 0, 0, -1, 2, 0, 0, 3}, 0),
              b + ": the corners in order");
        const Zonotope padded = Zonotope::matrix(
            M.center(), Tensor::concat({M.generators(), Tensor::zeros({1, 2, 2})}, 0));
        check(padded.vertex_matrices()[0].shape() == Shape{4, 2, 2},
              b + ": a zero generator adds no corner");
        const Zonotope point = test::point_set(M.center());
        check(point.vertex_matrices()[0].shape() == Shape{1, 2, 2}, b + ": a point has itself");
    });
    // more than 20 generators are refused
    check(test::raises(
              [] {
                  Zonotope::matrix(test::matrix({0}, 1, 1), Tensor::zeros({21, 1, 1}))
                      .vertex_matrices();
              },
              Error::not_supported),
          "vertices: too many generators");
    ChecksScope off(false);
    check(test::example().vertex_matrices()[0].shape() == Shape{4, 2, 2},
          "vertices: the same with the checks off");
    return test::finish("mat_zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
