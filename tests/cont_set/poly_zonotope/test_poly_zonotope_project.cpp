// test_poly_zonotope_project - PolyZonotope project: the rows of the representation
//
// CORA: contSet/@polyZonotope/project.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {vec({1.0, 0.0}),
            Tensor::from_data({1, 1, 0, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {1, 2}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const Set S = P.project({1});
        const PolyZonotope &Q = S.as<PolyZonotope>();
        check(Q.dim() == 1 && close(Q.center(), {0.0}, 0) && Q.num_dependent() == 2,
              b + ": the second row, the factors stay");
    });
    check(test::raises([] { known().project({2}); }, Error::wrong_value),
          "project: a dimension out of range");
    ChecksScope off(false);
    check(known().project({0})->dim() == 1, "project: valid with the checks off");
    return test::finish("poly_zonotope project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
