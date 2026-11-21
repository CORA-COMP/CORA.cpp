// test_lean_zonotope_rotate - Zonotope::rotate against the CORALean oracle
//
// The oracle rotates by the interval matrix of cos and sin, which adds one generator per
// dimension for the radius of that matrix. At dtype nearest the radius is zero, and the
// oracle's cos and sin are the platform's: the center and the generators that remain agree
// bit for bit with CORA.cpp's. The test asserts that the added generators are zero.
//
// ORACLE: contSet/@zonotope/rotate.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <algorithm>

using namespace cora;

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
    if (!test::oracle::available()) return 0;
    test::oracle::run("zonotope.rotate", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 50}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const double angle = rnd.uniform(-3, 3);
            const Zonotope mine = Z.rotate({0, int64_t(n - 1)}, angle);
            const auto [theirs, error] = lean::Zonotope(Z).rotate(20, 0, n - 1, angle).gather();
            const std::vector<double> lo = error.inf().data(), hi = error.sup().data();
            test::check(std::all_of(lo.begin(), lo.end(), [](double x) { return x == 0.0; }) &&
                            std::all_of(hi.begin(), hi.end(), [](double x) { return x == 0.0; }),
                        "zonotope.rotate: no error box at nearest");
            test::oracle::check("zonotope.rotate.c", mine.center(), theirs.center());
            test::oracle::check("zonotope.rotate.G", aux_nonzero_rows(mine.generators()),
                                aux_nonzero_rows(theirs.generators()));
        }
    });
    return test::finish("lean zonotope rotate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
