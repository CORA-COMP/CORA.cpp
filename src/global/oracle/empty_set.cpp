// empty_set - lean::EmptySet and its oracle operations
//
// See also: lean/empty_set.h, lean/request.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/global/oracle/empty_set.h"

#include "cora/global/oracle/fullspace.h"
#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The other operand on the wire: its dimension, and whether it is empty (a center without columns).
Wire aux_other(const cora::ContSet &S) {
    Json j = Json::object();
    j.set("n", Json(S.dim())).set("empty", Json(S.center().shape().back() == 0));
    return Wire(j, "");
}

EmptySet aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<EmptySet>("E");
}

bool aux_bool(const char *op, std::initializer_list<Operand> operands, const char *key = "value") {
    return ask(op, operands).get<bool>(key);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and set operations ----------------------------------------------------------------

// The result set is its dimension.
EmptySet EmptySet::empty(int64_t n) { return aux_set("emptySet.empty", {{"n", n}}); }

EmptySet EmptySet::intersect(const cora::ContSet &S) const {
    return aux_set("emptySet.and", {{"n", n_}, {"S", aux_other(S)}});
}

EmptySet EmptySet::plus(const cora::ContSet &S) const {
    return aux_set("emptySet.plus", {{"n", n_}, {"S", aux_other(S)}});
}

Interval EmptySet::convex_hull(const Interval &I) const {
    return ask("emptySet.convHull", {{"n", n_}, {"S", I}}).get<Interval>("S");
}

Zonotope EmptySet::convex_hull(const Zonotope &Z) const {
    return ask("emptySet.convHull", {{"n", n_}, {"S", Z}}).get<Zonotope>("S");
}

// The predicates, and the properties of the set.
bool EmptySet::contains(const cora::ContSet &S, double tol) const {
    return aux_bool("emptySet.contains", {{"n", n_}, {"S", aux_other(S)}, {"tol", number(tol)}},
                    "contains");
}

bool EmptySet::is_intersecting(const cora::ContSet &S, double tol) const {
    return !aux_bool("emptySet.isIntersecting",
                     {{"n", n_}, {"S", aux_other(S)}, {"tol", number(tol)}}, "disjoint");
}

bool EmptySet::isequal(const cora::ContSet &S, double tol) const {
    return aux_bool("emptySet.isequal", {{"n", n_}, {"S", aux_other(S)}, {"tol", number(tol)}});
}

// Maps and support -------------------------------------------------------------------------------

// The image under a matrix has the dimension of its rows; no point, so no support.
EmptySet EmptySet::linear_map(const Tensor &M) const {
    return aux_set("emptySet.mtimes", {{"n", n_}, {"M", M}});
}
double EmptySet::support_function(const Tensor &dir, const std::string &type) const {
    return ask("emptySet.supportFunc", {{"n", n_}, {"dir", dir}, {"type", type}})
        .get<double>("value");
}

bool EmptySet::representsa(const std::string &type, double tol) const {
    return aux_bool("emptySet.representsa", {{"n", n_}, {"type", type}, {"tol", number(tol)}});
}

// Boundedness and dimension of the interior.
bool EmptySet::is_bounded(double tol) const {
    return aux_bool("emptySet.isBounded", {{"n", n_}, {"tol", number(tol)}});
}

bool EmptySet::is_full_dim(double tol) const {
    return aux_bool("emptySet.isFullDim", {{"n", n_}, {"tol", number(tol)}});
}

// The center is nothing; the interval and the vertices are the oracle's empty ones.
std::optional<Tensor> EmptySet::center() const {
    return ask("emptySet.center", {{"n", n_}}).get<std::optional<Tensor>>("value");
}

// The numbers and sets the set stands for.
Interval EmptySet::interval() const {
    return ask("emptySet.interval", {{"n", n_}}).get<Interval>("I");
}

Tensor EmptySet::vertices() const { return ask("emptySet.vertices", {{"n", n_}}).get<Tensor>("V"); }

EmptySet EmptySet::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set("emptySet.lift", {{"n", n_}, {"N", N}, {"dims", dims}});
}

EmptySet EmptySet::project(const std::vector<int64_t> &dims) const {
    return aux_set("emptySet.project", {{"n", n_}, {"dims", dims}});
}

EmptySet EmptySet::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    return aux_set("emptySet.projectHighDim", {{"n", n_}, {"N", N}, {"dims", proj}});
}

// The radius and the volume are numbers, the complement a set of the other class.
double EmptySet::radius() const { return ask("emptySet.radius", {{"n", n_}}).get<double>("value"); }

double EmptySet::volume() const { return ask("emptySet.volume", {{"n", n_}}).get<double>("value"); }

Fullspace EmptySet::complement() const {
    return ask("emptySet.not", {{"n", n_}}).get<Fullspace>("F");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
