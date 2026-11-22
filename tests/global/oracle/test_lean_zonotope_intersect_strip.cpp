// test_lean_zonotope_intersect_strip - Zonotope::intersect_strip against the CORALean oracle, bit
// for bit
//
// For a given gain Lambda the oracle's `zonotope.intersectStripCora` has CORA.cpp's center
// c + Lambda (y - C c) and the generators [(I - Lambda C) G, Lambda diag(phi)]. The rounding of the
// interval matrix adds generators, all zero at nearest, which the test drops. (`zonotope.
// intersectStrip` forms the center as (I - Lambda C) c + Lambda y and is only compared by its box in
// test_lean_zonotope_measures_bound.) The choice of the gain is not compared.
//
// ORACLE: contSet/@zonotope/intersectStrip.m
// Authors:       Claude Sonnet 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <algorithm>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators (rows) of Z without the zero ones.
Tensor aux_nonzero_rows(const Tensor &G) {
    const int64_t n = G.shape()[1];
    const std::vector<double> data = G.data();
    std::vector<double> kept;
    for (std::size_t row = 0; row * n < data.size(); ++row) {
        const auto first = data.begin() + row * n;
        if (std::any_of(first, first + n, [](double x) { return x != 0.0; }))
            kept.insert(kept.end(), first, first + n);
    }
    return Tensor::from_data(kept, {int64_t(kept.size()) / n, n});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.intersectStripCora", [](oracle::Random &rnd) {
        for (const auto &[n, m, k] : {std::tuple{2, 3, 1}, {3, 5, 2}, {5, 12, 3}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor C = rnd.matrix(k, n), Lambda = rnd.matrix(n, k), y = rnd.vector(k);
            const Tensor phi = Tensor::from_data({0.5, 1.5, 1.0}, {3, 1}).slice(-2, 0, k);
            const Zonotope mine = Z.intersect_strip(C, phi.squeeze(-1), y.squeeze(-1), Lambda);
            const auto [theirs, error] =
                lean::Zonotope(Z).intersect_strip_cora(C, phi, y, Lambda).gather();
            const std::vector<double> lo = error.inf().data(), hi = error.sup().data();
            test::check(std::all_of(lo.begin(), lo.end(), [](double x) { return x == 0.0; }) &&
                            std::all_of(hi.begin(), hi.end(), [](double x) { return x == 0.0; }),
                        "intersectStripCora: no error box at nearest");
            oracle::check("intersectStripCora.c", mine.center(), theirs.center());
            oracle::check("intersectStripCora.G", aux_nonzero_rows(mine.generators()),
                          aux_nonzero_rows(theirs.generators()));
        }
    });
    return test::finish("lean zonotope intersect_strip");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
