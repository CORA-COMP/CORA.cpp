// test_lean_poly_zonotope_compact - PolyZonotope::compact against the CORALean oracle, bit for bit
//
// The oracle's `polyZonotope.compact` is CORA.cpp's: the generators of no factor added to the
// center (each sum rounded), the factors no column reads dropped with their identifiers, the
// constructor again. The sets have constant columns, repeated columns, a factor that only the
// independent generators use and one that nothing uses.
//
// ORACLE: contSet/@polyZonotope/compact_.m
// Authors:       Claude Sonnet 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/global/oracle/poly_zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A set of n dimensions over p + 1 factors, the last of which no column reads: h columns, a
/// constant one among them, and (with repeats) two columns that share their exponents.
std::pair<PolyZonotope, lean::PolyZonotope> aux_sets(oracle::Random &rnd, int64_t n, int64_t h,
                                                     int64_t p, int64_t q, bool repeats) {
    Exponents E;
    for (int64_t l = 0; l < h; ++l) {
        std::vector<int> col(std::size_t(p + 1), 0);
        col[std::size_t(l % p)] = 1 + int(l / p);
        E.push_back(col);
    }
    // a constant column, and one that repeats another
    E[std::size_t(h / 2)] = std::vector<int>(std::size_t(p + 1), 0);
    if (repeats && h > 3) E[std::size_t(h - 1)] = E[0];
    std::vector<int64_t> id(std::size_t(p + 1), 0);
    for (int64_t i = 0; i <= p; ++i) id[std::size_t(i)] = 2 * i + 3;
    const Tensor c = rnd.vector(n), G = rnd.matrix(n, h), GI = rnd.matrix(n, q);
    return {PolyZonotope(c.squeeze(-1), G.matrix_transpose(), GI.matrix_transpose(), E, id),
            lean::PolyZonotope::construct(c, G, GI, E, id)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polyZonotope.compact", [](oracle::Random &rnd) {
        for (const auto &[n, h, p, q] : {std::tuple{2, 4, 2, 0}, {3, 6, 3, 2}, {4, 9, 4, 3}})
            for (const bool repeats : {false, true}) {
                const auto [P, theirs] = aux_sets(rnd, n, h, p, q, repeats);
                oracle::check(P.compact(), theirs.compact());
            }
    });
    return test::finish("lean poly_zonotope compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
