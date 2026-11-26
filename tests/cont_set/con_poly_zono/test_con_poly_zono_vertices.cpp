// test_con_poly_zono_vertices - ConPolyZono's support function and plot hold the set's points
//
// Random sets (fixed seed) with a factor point a0 that meets the constraint by construction
// (b = A mu(a0)): the support function bounds d' x0 in random directions, and the polygons of the
// plot reach x0 (it lies in their bounding box).
//
// CORA: contSet/@conPolyZono/supportFunc_.m
// CORA: contSet/@conPolyZono/vertices_.m
// PROPERTY: contSet/@conPolyZono/supportFunc_.m
// PROPERTY: contSet/@conPolyZono/vertices_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"
#include "cora/cont_set/con_poly_zono/con_poly_zono.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// n normal draws of the scale 1.
std::vector<double> aux_draw(Rng &rng, std::size_t n) {
    std::vector<double> v(n);
    rng.normal(v.data(), n, 1.0);
    return v;
}

/// The monomial prod_i a_i^e_i.
double aux_mono(const std::vector<int> &e, const std::vector<double> &a) {
    double m = 1;
    for (std::size_t i = 0; i < a.size(); ++i) m *= std::pow(a[i], e[i]);
    return m;
}

/// A random set of 2 dimensions on 2 factors, its row met at a0, and its point x0 at a0.
std::pair<ConPolyZono, std::vector<double>> aux_set(Rng &rng, const std::vector<double> &a0) {
    const Exponents E = {{1, 0}, {0, 1}, {1, 1}}, EC = {{1, 0}, {0, 2}};
    const std::vector<double> c = aux_draw(rng, 2), g = aux_draw(rng, 6), A = aux_draw(rng, 2);
    const double b = A[0] * aux_mono(EC[0], a0) + A[1] * aux_mono(EC[1], a0);
    std::vector<double> x0 = c;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t r = 0; r < 2; ++r) x0[r] += aux_mono(E[j], a0) * g[j * 2 + r];
    return {ConPolyZono(Tensor::from_data(c, {2}), Tensor::from_data(g, {3, 2}), E,
                        Tensor::from_data(A, {1, 2}), Tensor::from_data({b}, {1}), EC,
                        Tensor::zeros({0, 2}), {1, 2}),
            x0};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    Rng rng(4);
    for (int trial = 0; trial < 5; ++trial) {
        std::vector<double> a0(2);
        rng.uniform(a0.data(), 2, -1.0, 1.0);
        const auto [S, x0] = aux_set(rng, a0);
        // the support function in random directions
        for (int k = 0; k < 4; ++k) {
            const std::vector<double> d = aux_draw(rng, 2);
            const double value = S.support_function(Tensor::from_data(d, {2})).item();
            check(value >= d[0] * x0[0] + d[1] * x0[1] - 1e-9, "support_function bounds x0");
        }
        // the plot's polygons reach x0
        double lo0 = 1e300, hi0 = -1e300, lo1 = 1e300, hi1 = -1e300;
        for (const Outline &P : S.vertices())
            for (const Point &v : P) {
                lo0 = std::min(lo0, v[0]);
                hi0 = std::max(hi0, v[0]);
                lo1 = std::min(lo1, v[1]);
                hi1 = std::max(hi1, v[1]);
            }
        check(lo0 <= x0[0] + 1e-9 && x0[0] <= hi0 + 1e-9 && lo1 <= x0[1] + 1e-9 &&
                  x0[1] <= hi1 + 1e-9,
              "vertices reach x0");
    }
    {
        ChecksScope off(false);
        check(!test::cpz::known().vertices().empty(), "vertices: with the checks off");
    }
    return test::finish("con_poly_zono vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
