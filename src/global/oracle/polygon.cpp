// polygon - lean::Polygon: the `polygon.*` ops of the oracle on a list of vertex polytopes
//
// The wire form is {"pieces": [{V, E}, ...]}, every piece a lean vertex polytope.
//
// Syntax:   lean::Polygon r = lean::Polygon(pgon).plus(other);   cora::Polygon p = r.nominal();
// Inputs:   pgon - a cora::Polygon, or the pieces as lean vertex polytopes
// Outputs:  the oracle's results: polygons, polytopes, boxes, Booleans and bounds
// See also: lean/polygon.h, lean/polytope.cpp, lean/request.h
//
// CORA.cpp only: the client of CORALean's polygon ops, which the oracle tests call
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/polygon.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The polygon `pgon` of the response.
Polygon aux_set(const Reply &reply) { return reply.get<Polygon>("pgon"); }

std::vector<Polytope> aux_pieces(const cora::Polygon &pgon) {
    std::vector<Polytope> out;
    for (const cora::Polytope &P : pgon.pieces()) out.emplace_back(P);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

Polygon::Polygon(const cora::Polygon &pgon)
    : dtype_(Tensor(pgon.like()).dtype()), pieces_(aux_pieces(pgon)) {}

Polygon::Polygon(std::string dtype, std::vector<Polytope> pieces)
    : dtype_(std::move(dtype)), pieces_(std::move(pieces)) {}

cora::Polygon Polygon::nominal() const {
    std::vector<cora::Polytope> pieces;
    pieces.reserve(pieces_.size());
    for (const Polytope &P : pieces_) pieces.push_back(P.gather());
    return cora::Polygon(pieces);
}

Json Polygon::to_json() const {
    Json list = Json::array();
    for (const Polytope &P : pieces_) list.push(P.to_json());
    return Json::object().set("pieces", list);
}

Polygon Polygon::from_json(const std::string &dtype, const Json &j) {
    std::vector<Polytope> pieces;
    for (const Json &p : j.at("pieces").items) pieces.push_back(Polytope::from_json(dtype, p));
    return {dtype, std::move(pieces)};
}

// The maps and sums -----------------------------------------------------------------------------

Polygon Polygon::linear_map(const Tensor &M) const {
    return aux_set(ask("polygon.mtimes", {{"pgon", *this}, {"M", M}}));
}

// the diagonal map and the sums of every pair of pieces
Polygon Polygon::times(const Tensor &factor) const {
    return aux_set(ask("polygon.times", {{"pgon", *this}, {"factor", factor}}));
}

Polygon Polygon::plus(const Polygon &other) const {
    return aux_set(ask("polygon.plus", {{"pgon", *this}, {"pgon2", other}}));
}

Polygon Polygon::minus(const Tensor &p) const {
    return aux_set(ask("polygon.minus", {{"pgon", *this}, {"point", p}}));
}

Polygon Polygon::unite(const Polygon &other) const {
    return aux_set(ask("polygon.or", {{"pgon", *this}, {"pgon2", other}}));
}

Polygon Polygon::linear_combination(const Polygon &other) const {
    return aux_set(ask("polygon.linComb", {{"pgon", *this}, {"pgon2", other}}));
}

Polygon Polygon::project(const std::vector<int64_t> &dims) const {
    return aux_set(ask("polygon.project", {{"pgon", *this}, {"dims", dims}}));
}

Polygon Polygon::empty() { return aux_set(ask("polygon.empty")); }

// The hull, the box, the pieces and the bounds ----------------------------------------------

Polytope Polygon::convex_hull() const {
    return ask("polygon.convHull", {{"pgon", *this}}).get<Polytope>("P");
}

Interval Polygon::interval() const {
    return ask("polygon.interval", {{"pgon", *this}}).get<Interval>("I");
}

std::vector<Polytope> Polygon::split_into_convex_sets() const {
    return ask("polygon.splitIntoConvexSets", {{"pgon", *this}})
        .get<std::vector<Polytope>>("pieces");
}

cora::Tensor Polygon::vertices() const {
    return ask("polygon.vertices", {{"pgon", *this}}).get<Tensor>("V").gather();
}

double Polygon::support_function(const Tensor &dir) const {
    return ask("polygon.supportFunc", {{"pgon", *this}, {"dir", dir}, {"type", "upper"}})
        .get<double>("value");
}

// The predicates --------------------------------------------------------------------------------

bool Polygon::contains(const Tensor &point) const {
    return ask("polygon.contains", {{"pgon", *this}, {"point", point}}).get<bool>("value");
}

bool Polygon::is_disjoint(const Polygon &other) const {
    return ask("polygon.isIntersecting", {{"pgon", *this}, {"pgon2", other}}).get<bool>("disjoint");
}

bool Polygon::isequal(const Polygon &other) const {
    return ask("polygon.isequal", {{"pgon", *this}, {"pgon2", other}}).get<bool>("value");
}

// the predicates of the pieces alone
bool Polygon::is_convex() const {
    return ask("polygon.isConvex", {{"pgon", *this}}).get<bool>("value");
}

bool Polygon::is_full_dim() const {
    return ask("polygon.isFullDim", {{"pgon", *this}}).get<bool>("value");
}

bool Polygon::is_bounded() const {
    return ask("polygon.isBounded", {{"pgon", *this}}).get<bool>("value");
}

bool Polygon::represents_empty_set() const {
    return ask("polygon.representsa", {{"pgon", *this}, {"type", "emptySet"}}).get<bool>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
