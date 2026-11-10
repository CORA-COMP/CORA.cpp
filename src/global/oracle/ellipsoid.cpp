// ellipsoid - lean::Ellipsoid, an ellipsoid of exact values, computed by CORALean
//
// See also: lean/con_zonotope.h, lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/ellipsoid.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Whether a wire-form tensor or box has only zero entries ("0*2^0" is the text of zero).
bool aux_zero(const Json &j) {
    if (j.kind == Json::Kind::object) {
        for (const auto &member : j.members)
            if (!aux_zero(member.second)) return false;
        return true;
    }
    if (j.kind == Json::Kind::array) {
        for (const Json &item : j.items)
            if (!aux_zero(item)) return false;
        return true;
    }
    return j.kind != Json::Kind::string || j.text.rfind("0*", 0) == 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// the wire form of an exact set is {Q, q}
Ellipsoid::Ellipsoid(const cora::Ellipsoid &E)
    : dtype_(Tensor(E.Q()).dtype()), json_(Json::object()) {
    json_.set("Q", Tensor(E.Q()).to_json()).set("q", Tensor(E.q()).to_json());
}

cora::Ellipsoid Ellipsoid::nominal() const {
    return {Tensor::from_json(dtype_, json_.at("Q")).gather(),
            Tensor::from_json(dtype_, json_.at("q")).gather()};
}

bool Ellipsoid::exact() const {
    const Json *error = json_.find("E");
    return !error || aux_zero(*error);
}

// Sets ---------------------------------------------------------------------------------------

// each op sends the ellipsoid (and its arguments) and reads `E` back

Ellipsoid Ellipsoid::origin(int64_t n) {
    return ask("ellipsoid.origin", {{"n", n}}).get<Ellipsoid>("E");
}

Ellipsoid Ellipsoid::project(const std::vector<int64_t> &dims) const {
    return ask("ellipsoid.project", {{"E", *this}, {"dims", dims}}).get<Ellipsoid>("E");
}

Ellipsoid Ellipsoid::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return ask("ellipsoid.lift", {{"E", *this}, {"N", N}, {"dims", dims}}).get<Ellipsoid>("E");
}

Ellipsoid Ellipsoid::linear_map(const Tensor &M) const {
    return ask("ellipsoid.mtimes", {{"E", *this}, {"M", M}}).get<Ellipsoid>("E");
}

// the answer is a Boolean (certified when true)
bool Ellipsoid::isequal(const Ellipsoid &E2, double tol) const {
    return ask("ellipsoid.isequal", {{"E", *this}, {"E2", E2}, {"tol", number(tol)}})
        .get<bool>("value");
}

// Bounds -------------------------------------------------------------------------------------

// numbers the oracle proves, rounded upward in a sound dtype

double Ellipsoid::norm(int k) const {
    return ask("ellipsoid.norm", {{"E", *this}, {"k", k}, {"type", 2}}).get<double>("value");
}

double Ellipsoid::radius(int p) const {
    return ask("ellipsoid.radius", {{"E", *this}, {"p", p}}).get<double>("value");
}

Zonotope Ellipsoid::zonotope() const {
    return ask("ellipsoid.zonotope", {{"E", *this}}).get<Zonotope>("Z");
}

Json Ellipsoid::to_json() const { return json_; }

Ellipsoid Ellipsoid::from_json(const std::string &dtype, const Json &j) { return {dtype, j}; }

// Operations ---------------------------------------------------------------------------------

Tensor Ellipsoid::center() const {
    return ask("ellipsoid.center", {{"E", *this}}).get<Tensor>("value");
}

bool Ellipsoid::is_full_dim() const {
    return ask("ellipsoid.isFullDim", {{"E", *this}}).get<bool>("value");
}

bool Ellipsoid::is_bigger(const Ellipsoid &E2) const {
    return ask("ellipsoid.isBigger", {{"E", *this}, {"E2", E2}}).get<bool>("value");
}

Ellipsoid Ellipsoid::cart_prod(const Tensor &v) const {
    return ask("ellipsoid.cartProd", {{"E", *this}, {"v", v}}).get<Ellipsoid>("E");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
