// test_lean_poly_zonotope_project - PolyZonotope::project against the CORALean oracle, bit for bit
//
// Selecting coordinates does no arithmetic: the center and both blocks of generators at the chosen
// coordinates, the exponents and the identifiers kept. The sets are random, with and without
// independent generators; dims repeat and reorder coordinates.
//
// ORACLE: contSet/@polyZonotope/project.m
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

/// A random set of n dimensions: h dependent generators over p factors (distinct exponent
/// columns, none constant), q independent ones, as the oracle's constructor takes them.
std::pair<PolyZonotope, lean::PolyZonotope> aux_sets(oracle::Random &rnd, int64_t n, int64_t h,
                                                     int64_t p, int64_t q) {
    Exponents E;
    for (int64_t l = 0; l < h; ++l) {
        std::vector<int> col(static_cast<std::size_t>(p), 0);
        col[std::size_t(l % p)] = 1 + int(l / p);
        E.push_back(col);
    }
    std::vector<int64_t> id(std::size_t(p), 0);
    for (int64_t i = 0; i < p; ++i) id[std::size_t(i)] = 2 * i + 3;
    const Tensor c = rnd.vector(n), G = rnd.matrix(n, h), GI = rnd.matrix(n, q);
    return {PolyZonotope(c.squeeze(-1), G.matrix_transpose(), GI.matrix_transpose(), E, id),
            lean::PolyZonotope::construct(c, G, GI, E, id)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polyZonotope.project", [](oracle::Random &rnd) {
        for (const auto &[n, h, p, q] : {std::tuple{3, 4, 2, 0}, {4, 6, 3, 2}, {5, 9, 4, 3}}) {
            const auto [P, theirs] = aux_sets(rnd, n, h, p, q);
            for (const std::vector<int64_t> &dims :
                 {std::vector<int64_t>{1}, {0, n - 1}, {n - 1, 0, 1}, {2, 2}})
                oracle::check(P.project(dims).as<PolyZonotope>(), theirs.project(dims));
        }
    });
    return test::finish("lean poly_zonotope project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
