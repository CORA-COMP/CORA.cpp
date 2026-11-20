// test_poly_zonotope_quad_map - PolyZonotope quad_map: the image under z = x' Q x
//
// CORA: contSet/@polyZonotope/quadMap.m
// PROPERTY: contSet/@polyZonotope/quadMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 1, 0, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {1, 2}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

/// Stated property: x' Q x of the points of the set lies in the zonotope of the image.
void quad_property(const std::string &b) {
    Rng rng(31);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope P = test::random_poly(rng, 3);
        const std::vector<Tensor> Q = {test::normal(rng, 3, 3), test::normal(rng, 3, 3)};
        const Tensor x = test::sample(P, rng, 25);
        // x' Q x of every sampled point, one row per matrix
        std::vector<double> img;
        const std::vector<double> d = x.matrix_transpose().data(); // columns (3, 25)
        for (const Tensor &Qi : Q) {
            const std::vector<double> q = Qi.data();
            for (int j = 0; j < 25; ++j) {
                double s = 0;
                for (int a = 0; a < 3; ++a)
                    for (int c = 0; c < 3; ++c) s += d[a * 25 + j] * q[a * 3 + c] * d[c * 25 + j];
                img.push_back(s);
            }
        }
        check(test::inside(P.quad_map(Q).zonotope(),
                           Tensor::from_data(img, {2, 25}).matrix_transpose()),
              b + ": the images are inside");
    }
}

/// No dependent generators: the image is the quadratic map of the zonotope (c, GI).
void independent_only() {
    const PolyZonotope P(column({1.0, 0.0}), Tensor::zeros({0, 2}),
                         Tensor::from_data({0.5, 1.0}, {1, 2}), {}, {});
    const std::vector<Tensor> Q = {Tensor::from_data({0, 1, 0, 0}, {2, 2}), Tensor::eye(2)};
    const Zonotope image = P.quad_map(Q).zonotope(), expected = P.zonotope().quad_map(Q);
    check(P.quad_map(Q).num_dependent() == 0 && test::close(image.center(), expected.center(), 0) &&
              test::close(image.generators(), expected.generators(), 0),
          "quad_map: no dependent generators");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // x = 1, y = a: x y = a and y y = a^2 are the generators of the image
        const PolyZonotope P(column({1.0, 0.0}), Tensor::from_data({0.0, 1.0}, {1, 2}),
                             Tensor::zeros({0, 2}), {{1}}, {1});
        const PolyZonotope Z = P.quad_map(
            {Tensor::from_data({0, 1, 0, 0}, {2, 2}), Tensor::from_data({0, 0, 0, 1}, {2, 2})});
        check(Z.dim() == 2 && close(Z.center(), {0.0, 0.0}, 0), b + ": the image has the center 0");
        check(Z.num_dependent() == 2 && Z.exponents()[0] == std::vector<int>({1}) &&
                  Z.exponents()[1] == std::vector<int>({2}),
              b + ": a and a^2 are generators");
        quad_property(b);
    });
    check(test::raises([] { known().quad_map({}); }, Error::wrong_value), "quad_map: no matrix");
    check(
        test::raises([] { known().quad_map({Tensor::zeros({3, 3})}); }, Error::dimension_mismatch),
        "quad_map: size");
    independent_only();
    ChecksScope off(false);
    check(known().quad_map({Tensor::eye(2)}).dim() == 1, "quad_map: valid with the checks off");
    return test::finish("poly_zonotope quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
