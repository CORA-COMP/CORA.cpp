// polygon - the constructors of a polygon, checked, as CORA's polygon
//
// From convex pieces (CORALean's Polygon.ofPieces), or from a boundary as CORA's polygon(V): the
// boundary is split into convex pieces (priv_convex_split), one region per run of rows between
// NaN rows. CORA reads a region inside another as a hole; CORA.cpp refuses nested boundaries
// (not ported). A halfspace piece is held by its vertices (its outline).
//
// Syntax:   Polygon pgon({P1, P2});   Polygon pgon(V);
// Inputs:   pieces - polytopes of two dimensions, bounded, not a batch
//           V - the boundary (p, 2), one vertex per row, NaN rows between regions
// Outputs:  pgon - the polygon
// See also: empty, from, priv_convex_split
//
// CORA: contSet/@polygon/polygon.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"
#include "cont_set/polygon/private/priv.h"

#include <cmath>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The (0, 2) tensor like the first piece, or like V.
Tensor aux_like(const Tensor &t) { return t.zeros_like({0, 2}); }

/// The piece as a vertex polytope of two dimensions (a halfspace one by its outline).
Polytope aux_vertex_piece(const Polytope &P) {
    if (P.is_v_rep()) return P;
    const std::vector<Outline> outlines = P.vertices();
    Piece v = outlines.empty() ? Piece{} : Piece(outlines.front().begin(), outlines.front().end());
    return priv_piece(v, P.A());
}

/// The regions of the boundary rows: the runs between NaN rows.
std::vector<Piece> aux_regions(const std::vector<double> &data) {
    std::vector<Piece> regions(1);
    for (std::size_t r = 0; 2 * r < data.size(); ++r) {
        const Point x = {data[2 * r], data[2 * r + 1]};
        if (std::isnan(x[0]) || std::isnan(x[1])) {
            if (!regions.back().empty()) regions.emplace_back();
        } else {
            regions.back().push_back(x);
        }
    }
    if (regions.back().empty()) regions.pop_back();
    return regions;
}

/// Whether every vertex of the region lies in one of the pieces.
bool aux_inside(const Piece &region, const std::vector<Piece> &pieces) {
    for (const Point &x : region) {
        bool held = false;
        for (const Piece &p : pieces) held = held || priv_piece_holds(p, x);
        if (!held) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The pieces -------------------------------------------------------------------------------------

Polygon::Polygon(Unchecked, std::vector<Polytope> pieces, Tensor like)
    : ContSet(kClassId), pieces_(std::move(pieces)), like_(std::move(like)) {}

Polygon priv_polygon(std::vector<Polytope> pieces, const Tensor &like) {
    return Polygon(Polygon::Unchecked{}, std::move(pieces), like);
}

Polygon::Polygon(const std::vector<Polytope> &pieces) : ContSet(kClassId) {
    detail::Entry entry;
    for (const Polytope &P : pieces) {
        CORA_CHECK(P.dim() == 2, Error::dimension_mismatch, [&] {
            return "polygon: every piece must be two-dimensional, got one of dimension " +
                   std::to_string(P.dim());
        });
        CORA_CHECK((P.is_v_rep() ? P.V() : P.A()).ndim() == 2, Error::not_supported, [] {
            return std::string("polygon: a piece must be a single polytope, not a batch");
        });
        CORA_CHECK(P.is_bounded(), Error::wrong_input_in_constructor,
                   [] { return std::string("polygon: every piece must be bounded"); });
    }
    // the pieces in vertex form
    pieces_.reserve(pieces.size());
    for (const Polytope &P : pieces) pieces_.push_back(aux_vertex_piece(P));
    like_ =
        pieces_.empty() ? Polytope::empty(2).A().zeros_like({0, 2}) : aux_like(pieces_.front().V());
}

// A boundary --------------------------------------------------------------------------------------

Polygon::Polygon(const Tensor &V) : ContSet(kClassId) {
    detail::Entry entry;
    CORA_CHECK(V.defined() && V.ndim() == 2 && V.size(-1) == 2, Error::wrong_input_in_constructor,
               [&] { return "polygon: the boundary must be a matrix (p, 2), one vertex per row"; });
    const std::vector<double> data = V.data();
    for (const double x : data)
        CORA_CHECK(std::isfinite(x) || std::isnan(x), Error::wrong_input_in_constructor, [] {
            return std::string("polygon: the boundary must be finite (NaN rows separate regions)");
        });
    like_ = aux_like(V);
    std::vector<std::vector<Piece>> split;
    for (const Piece &region : aux_regions(data)) split.push_back(priv_convex_split(region));
    // a region inside another is a hole in CORA, which is not ported
    const std::vector<Piece> regions = aux_regions(data);
    for (std::size_t i = 0; i < split.size(); ++i)
        for (std::size_t j = 0; j < regions.size(); ++j)
            CORA_CHECK(i == j || !aux_inside(regions[j], split[i]), Error::not_supported, [] {
                return std::string("polygon: a region inside another (a hole) is not supported");
            });
    for (const std::vector<Piece> &pieces : split)
        for (const Piece &piece : pieces) pieces_.push_back(priv_piece(piece, like_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
