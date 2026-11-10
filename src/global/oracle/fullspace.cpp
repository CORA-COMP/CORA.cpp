// fullspace - lean::Fullspace, the whole space R^n, and the oracle's `fullspace.*` operations
//
// See also: lean/request.h, lean/interval.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/global/oracle/fullspace.h"
#include "cora/global/error.h"

#include "cora/global/oracle/empty_set.h"
#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The set `F` of the response to `op` on the operands.
Fullspace aux_space(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Fullspace>("F");
}

bool aux_bool(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<bool>("value");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Wire form and construction ---------------------------------------------------------------------

// The wire form is {n}.
Json Fullspace::to_json() const {
    Json j = Json::object();
    j.set("n", Json(n_));
    return j;
}

Fullspace Fullspace::from_json(const std::string &, const Json &j) {
    return Fullspace(std::stoll(j.at("n").text));
}

Fullspace Fullspace::inf(int64_t n) { return aux_space("fullspace.Inf", {{"n", n}}); }
Fullspace Fullspace::box() const { return aux_space("fullspace.box", {{"n", n_}}); }

Interval Fullspace::interval() const {
    return ask("fullspace.interval", {{"n", n_}}).get<Interval>("I");
}

// Set operations: the intersection returns the operand, the others the whole space.
Fullspace Fullspace::intersect(const Fullspace &F2) const {
    return ask("fullspace.and", {{"n", n_}, {"S", F2}}).get<Fullspace>("S");
}
Interval Fullspace::intersect(const Interval &I) const {
    return ask("fullspace.and", {{"n", n_}, {"S", I}}).get<Interval>("S");
}
Fullspace Fullspace::plus(const Fullspace &F2) const {
    return aux_space("fullspace.plus", {{"n", n_}, {"S", F2}});
}
Fullspace Fullspace::convex_hull(const Fullspace &F2) const {
    return aux_space("fullspace.convHull", {{"n", n_}, {"S", F2}});
}

// Certified predicates: a true proves what the operation names.
Certified Fullspace::contains(const Interval &I) const {
    return ask("fullspace.contains", {{"n", n_}, {"S", I}}).proved_contains();
}
Certified Fullspace::is_intersecting(const Interval &I) const {
    return ask("fullspace.isIntersecting", {{"n", n_}, {"S", I}}).proved_disjoint();
}
bool Fullspace::isequal(const Fullspace &F2, double tol) const {
    return aux_bool("fullspace.isequal", {{"n", n_}, {"S", F2}, {"tol", number(tol)}});
}

// Maps and support -------------------------------------------------------------------------------

// The image under a matrix is the whole space of its rows, whatever the rank.
Fullspace Fullspace::linear_map(const Tensor &M) const {
    return aux_space("fullspace.mtimes", {{"n", n_}, {"M", M}});
}
double Fullspace::support_function(const Tensor &dir, const std::string &type) const {
    return ask("fullspace.supportFunc", {{"n", n_}, {"dir", dir}, {"type", type}})
        .get<double>("value");
}

bool Fullspace::representsa(const std::string &type, double tol) const {
    return aux_bool("fullspace.representsa", {{"n", n_}, {"type", type}, {"tol", number(tol)}});
}
Tensor Fullspace::vertices() const {
    return ask("fullspace.vertices", {{"n", n_}}).get<Tensor>("V");
}

// Exact answers and the center, which is the origin.
bool Fullspace::is_bounded() const { return aux_bool("fullspace.isBounded", {{"n", n_}}); }
bool Fullspace::is_full_dim() const { return aux_bool("fullspace.isFullDim", {{"n", n_}}); }
Tensor Fullspace::center() const {
    return ask("fullspace.center", {{"n", n_}}).get<Tensor>("value");
}

// The dimension of the points is their number of rows.
Fullspace Fullspace::enclose_points(const Tensor &points) {
    return aux_space("fullspace.enclosePoints", {{"n", points.rows()}, {"points", points}});
}
Fullspace Fullspace::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_space("fullspace.lift", {{"n", n_}, {"N", N}, {"dims", dims}});
}

Fullspace Fullspace::project(const std::vector<int64_t> &dims) const {
    return aux_space("fullspace.project", {{"n", n_}, {"dims", dims}});
}
Fullspace Fullspace::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    return aux_space("fullspace.projectHighDim", {{"n", n_}, {"N", N}, {"dims", proj}});
}

// The radius and the volume are numbers, the complement a set of the other class.
double Fullspace::radius() const {
    return ask("fullspace.radius", {{"n", n_}}).get<double>("value");
}
double Fullspace::volume() const {
    return ask("fullspace.volume", {{"n", n_}}).get<double>("value");
}
EmptySet Fullspace::complement() const {
    return ask("fullspace.not", {{"n", n_}}).get<EmptySet>("E");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
