// test_poly_zonotope_get_subset - PolyZonotope get_subset: a factor fixed or restricted
//
// A value v for a factor: the points of the result are those of the set with that factor at v.
// A box [c - r, c + r]: the factor is c + r a for the new a. [-1, 1] changes nothing. Equal
// exponents are merged, constants join the center, unused factors go.
//
// CORA: contSet/@polyZonotope/getSubset.m
// PROPERTY: contSet/@polyZonotope/getSubset.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(13);
        for (int k = 0; k < 5; ++k) {
            const PolyZonotope P = test::random_poly(rng, 2);
            const PolyZonotope S = P.get_subset({1}, vec({0.5}));
            check(close(Tensor::from_data(test::point(S, {{2, -0.3}}, {0.4, 0.1}), {2}),
                        Tensor::from_data(test::point(P, {{1, 0.5}, {2, -0.3}}, {0.4, 0.1}), {2}),
                        1e-12),
                  b + ": a value for a factor");
            const Interval I(vec({-0.5}), vec({1.0})); // 0.25 + 0.75 a
            const PolyZonotope T = P.get_subset({1}, I);
            check(close(Tensor::from_data(test::point(T, {{1, 0.8}, {2, -0.3}}, {0.4, 0.1}), {2}),
                        Tensor::from_data(
                            test::point(P, {{1, 0.25 + 0.75 * 0.8}, {2, -0.3}}, {0.4, 0.1}), {2}),
                        1e-11),
                  b + ": a box for a factor");
            const PolyZonotope U =
                P.get_subset({1, 2}, Interval(vec({-1.0, -0.2}), vec({1.0, 0.6})));
            check(close(Tensor::from_data(test::point(U, {{1, 0.8}, {2, 0.5}}, {0.4, 0.1}), {2}),
                        Tensor::from_data(
                            test::point(P, {{1, 0.8}, {2, 0.2 + 0.4 * 0.5}}, {0.4, 0.1}), {2}),
                        1e-11),
                  b + ": the box [-1, 1] stays, another follows");
        }
        // c = 1, a + 2 a^2: at a = [0, 1] (a = 0.5 + 0.5 s): 1.5 + 1.5... by hand at s = 1: a = 1
        const PolyZonotope Q(vec({1.0}), Tensor::from_data({1, 2}, {2, 1}), Tensor::zeros({0, 1}),
                             {{1}, {2}}, {7});
        const PolyZonotope S = Q.get_subset({7}, Interval(vec({0.0}), vec({1.0})));
        check(close(Tensor::from_data(test::point(S, {{7, 1.0}}, {}), {1}), {4.0}),
              b + ": the box [0, 1] at its end is a = 1");
        // all factors fixed: a point, no factor left
        const PolyZonotope V = Q.get_subset({7}, vec({2.0}));
        check(V.num_dependent() == 0 && V.num_factors() == 0 && close(V.center(), {11.0}),
              b + ": a point is a set without generators");
        check(Q.get_subset({7}, Interval(vec({-1.0}), vec({1.0}))).isequal(Q),
              b + ": the box [-1, 1] changes nothing");
        // the power is multiplied up as CORALean's powDown, (v v) v, which std::pow(v, 3) is not
        const double v = 0.3, cube = (v * v) * v;
        const PolyZonotope C(vec({0.0}), Tensor::from_data({1.0}, {1, 1}), Tensor::zeros({0, 1}),
                             {{3}}, {7});
        check(C.get_subset({7}, vec({v})).center().data()[0] == cube,
              b + ": the cube of the value is (v v) v");
    });
    const PolyZonotope Q = test::sample_poly();
    check(test::raises([&] { Q.get_subset({9}, vec({0.1})); }, Error::wrong_value),
          "get_subset: an unknown identifier");
    check(test::raises([&] { Q.get_subset({1, 2}, vec({0.1})); }, Error::wrong_value),
          "get_subset: too few values");
    check(test::raises([&] { Q.get_subset({9}, Interval(vec({0.0}), vec({0.5}))); },
                       Error::wrong_value),
          "get_subset: an unknown identifier in a box");
    ChecksScope off(false);
    check(Q.get_subset({1}, vec({0.5})).dim() == 2, "get_subset: valid with the checks off");
    return test::finish("poly_zonotope get_subset");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
