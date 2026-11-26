// vertices - polygons that enclose a constrained polynomial zonotope, for plotting
//
// The set is split along its longest generators (split_longest_gen) four times; a piece whose
// constraint cannot hold (the box of the rows' polynomial, PolyLevelSet::lhs, misses 0) is
// dropped, and each other piece is drawn as its zonotope (zonotope()). CORA plots the union of
// the constrained zonotopes of the split pieces; this outer enclosure plots the same way.
//
// Syntax:   V = cPZ.vertices();
// Inputs:   -
// Outputs:  V - one polygon per piece
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/vertices_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the rows of the set can hold: 0 lies in the box of every row's polynomial.
bool aux_feasible(const ConPolyZono &S) {
    if (S.constraints().num_rows() == 0) return true;
    const Interval box = S.constraints().lhs(S.ids()).interval();
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (lo[i] > 0 || hi[i] < 0) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Outline> ConPolyZono::vertices_impl() const {
    std::vector<ConPolyZono> pieces = {*this};
    // four rounds of halving, the pieces that cannot hold dropped
    for (int round = 0; round < 4; ++round) {
        std::vector<ConPolyZono> next;
        for (const ConPolyZono &S : pieces) {
            if (S.ids().empty()) {
                next.push_back(S);
                continue;
            }
            // the halves, those whose rows can hold
            auto [a, b] = S.split_longest_gen();
            if (aux_feasible(a)) next.push_back(a.compact());
            if (aux_feasible(b)) next.push_back(b.compact());
        }
        pieces = std::move(next);
    }
    // each piece drawn as its zonotope
    std::vector<Outline> out;
    for (const ConPolyZono &S : pieces)
        for (Outline &P : S.zonotope().vertices()) out.push_back(std::move(P));
    return out;
}

std::vector<std::vector<Point3>> ConPolyZono::vertices_3d_impl() const {
    CORA_FAIL(
        Error::not_supported,
        "vertices_3d: a constrained polynomial zonotope has no 3-D plot (CORA's plot3D splits "
        "it into constrained zonotopes); plot zonotope() instead");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
