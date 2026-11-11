// test_poly_zonotope_vertices - PolyZonotope vertices: not available for a polynomial zonotope
//
// CORA: contSet/@polyZonotope/polygon.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { check(known().dim() == 2, b + ": a set"); });
    check(test::raises([] { known().vertices(); }, Error::not_supported), "vertices: none");
    check(test::raises([] { known().vertices_3d(); }, Error::not_supported), "vertices_3d: none");
    ChecksScope off(false);
    check(known().dim() == 2, "vertices: valid with the checks off");
    return test::finish("poly_zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
