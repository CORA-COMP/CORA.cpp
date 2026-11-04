// zono_bundle - lean::ZonoBundle, a zonotope bundle of exact values, computed by CORALean
//
// See also: lean/zonotope.h, lean/request.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/zono_bundle.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The bundle `zB` of the response to `op` on the operands.
ZonoBundle aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<ZonoBundle>("zB");
}

std::vector<Zonotope> aux_members(const cora::ZonoBundle &zB) {
    std::vector<Zonotope> members;
    for (const cora::Zonotope &Z : zB.zonotopes()) members.emplace_back(Z);
    return members;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

ZonoBundle::ZonoBundle(const cora::ZonoBundle &zB) : Z_(aux_members(zB)) {}

ZonoBundle::ZonoBundle(std::vector<Zonotope> zonotopes) : Z_(std::move(zonotopes)) {}

cora::ZonoBundle ZonoBundle::nominal() const {
    std::vector<cora::Zonotope> members;
    for (const Zonotope &Z : Z_) members.push_back(Z.gather().first);
    return cora::ZonoBundle(std::move(members));
}

// the wire form is {"Zs": [zonotope, ...]}

Json ZonoBundle::to_json() const {
    Json list = Json::array();
    for (const Zonotope &Z : Z_) list.push(Z.to_json());
    return Json::object().set("Zs", list);
}

ZonoBundle ZonoBundle::from_json(const std::string &dtype, const Json &j) {
    std::vector<Zonotope> members;
    for (const Json &z : j.at("Zs").items) members.push_back(Zonotope::from_json(dtype, z));
    return ZonoBundle(std::move(members));
}

// Operations ---------------------------------------------------------------------------------

ZonoBundle ZonoBundle::intersect(const ZonoBundle &zB2) const {
    return aux_set("zonoBundle.and", {{"zB", *this}, {"zB2", zB2}});
}

ZonoBundle ZonoBundle::cart_prod(const ZonoBundle &zB2) const {
    return aux_set("zonoBundle.cartProd", {{"zB", *this}, {"zB2", zB2}});
}

ZonoBundle ZonoBundle::enclose(const Tensor &M, const ZonoBundle &zB2) const {
    return aux_set("zonoBundle.enclose", {{"zB", *this}, {"M", M}, {"zB2", zB2}});
}

ZonoBundle ZonoBundle::enlarge(const Tensor &factor) const {
    return aux_set("zonoBundle.enlarge", {{"zB", *this}, {"factor", factor}});
}

ZonoBundle ZonoBundle::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set("zonoBundle.lift", {{"zB", *this}, {"N", N}, {"dims", dims}});
}

ZonoBundle ZonoBundle::empty(int64_t n) { return aux_set("zonoBundle.empty", {{"n", n}}); }

ConZonotope ZonoBundle::con_zonotope() const {
    return ask("zonoBundle.conZonotope", {{"zB", *this}}).get<ConZonotope>("cZ");
}

Interval ZonoBundle::interval() const {
    return ask("zonoBundle.interval", {{"zB", *this}}).get<Interval>("I");
}

bool ZonoBundle::is_bounded() const {
    return ask("zonoBundle.isBounded", {{"zB", *this}}).get<bool>("value");
}

Certified ZonoBundle::is_intersecting(const ZonoBundle &zB2, double tol) const {
    return ask("zonoBundle.isIntersecting", {{"zB", *this}, {"zB2", zB2}, {"tol", number(tol)}})
        .proved_disjoint();
}

bool ZonoBundle::isequal(const ZonoBundle &zB2, double tol) const {
    return ask("zonoBundle.isequal", {{"zB", *this}, {"zB2", zB2}, {"tol", number(tol)}})
        .get<bool>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
