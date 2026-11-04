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

// The bundle `bundle` of the response to `op` on the operands.
ZonoBundle aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<ZonoBundle>("zB");
}

std::vector<Zonotope> aux_members(const cora::ZonoBundle &bundle) {
    std::vector<Zonotope> members;
    for (const cora::Zonotope &Z : bundle.zonotopes()) members.emplace_back(Z);
    return members;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

ZonoBundle::ZonoBundle(const cora::ZonoBundle &bundle) : Z_(aux_members(bundle)) {}

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

ZonoBundle ZonoBundle::intersect(const ZonoBundle &other) const {
    return aux_set("zonoBundle.and", {{"zB", *this}, {"zB2", other}});
}

ZonoBundle ZonoBundle::cart_prod(const ZonoBundle &other) const {
    return aux_set("zonoBundle.cartProd", {{"zB", *this}, {"zB2", other}});
}

ZonoBundle ZonoBundle::enclose(const Tensor &M, const ZonoBundle &other) const {
    return aux_set("zonoBundle.enclose", {{"zB", *this}, {"M", M}, {"zB2", other}});
}

// the oracle scales every member about its own center, and lifts it by a permutation
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

// the constant true, as a Boolean `value`
bool ZonoBundle::is_bounded() const {
    return ask("zonoBundle.isBounded", {{"zB", *this}}).get<bool>("value");
}

// a disjoint box pair is a proof; the rest is not
Certified ZonoBundle::is_intersecting(const ZonoBundle &other, double tol) const {
    return ask("zonoBundle.isIntersecting", {{"zB", *this}, {"zB2", other}, {"tol", number(tol)}})
        .proved_disjoint();
}

bool ZonoBundle::isequal(const ZonoBundle &other, double tol) const {
    return ask("zonoBundle.isequal", {{"zB", *this}, {"zB2", other}, {"tol", number(tol)}})
        .get<bool>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
