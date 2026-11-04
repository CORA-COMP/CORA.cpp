// capsule - lean::Capsule, a capsule of exact values, computed by CORALean
//
// See also: lean/con_zonotope.h, lean/request.h

#include "cora/global/oracle/capsule.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The capsule `C` of the response to `op` on the operands.
Capsule aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<Capsule>("C");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

// the wire form of an exact capsule is {c, g, r}, r the text of a number
Capsule::Capsule(const cora::Capsule &C) : dtype_(Tensor(C.c()).dtype()), json_(Json::object()) {
    json_.set("c", Tensor(C.c()).to_json()).set("g", Tensor(C.generator()).to_json());
    json_.set("r", Json(Tensor::encode(C.r().item())));
}

bool Capsule::exact() const {
    const Json *E = json_.find("E");
    if (!E) return true;
    const auto [inf, sup] = Interval::from_json(dtype_, *E).gather();
    return inf.abs().max().item() == 0 && sup.abs().max().item() == 0;
}

// the empty capsule has the radius -1, which the constructor of cora::Capsule refuses
cora::Capsule Capsule::nominal() const {
    const cora::Tensor c = Tensor::from_json(dtype_, json_.at("c")).gather();
    const double r = Tensor::decode(json_.at("r").text);
    if (r < 0) return cora::Capsule::empty(c.shape()[0]);
    return {c, Tensor::from_json(dtype_, json_.at("g")).gather(), r};
}

cora::Capsule Capsule::gather() const {
    if (!exact())
        CORA_FAIL(Error::wrong_value, "lean: the oracle's capsule has a nonempty error box");
    return nominal();
}

Json Capsule::to_json() const { return json_; }

Capsule Capsule::from_json(const std::string &dtype, const Json &j) { return {dtype, j}; }

// Operations ---------------------------------------------------------------------------------

Tensor Capsule::center() const {
    return ask("capsule.center", {{"C", *this}}).get<Tensor>("value");
}

Interval Capsule::interval() const {
    return ask("capsule.interval", {{"C", *this}}).get<Interval>("I");
}

// Certified answers: a true proves containment, a false proves disjointness.

Certified Capsule::contains(const Tensor &p) const {
    return ask("capsule.contains", {{"C", *this}, {"point", p}}).proved_contains();
}

Certified Capsule::is_intersecting(const Capsule &C2, int k) const {
    return ask("capsule.isIntersecting", {{"C", *this}, {"C2", C2}, {"k", k}}).proved_disjoint();
}

bool Capsule::is_bounded() const {
    return ask("capsule.isBounded", {{"C", *this}}).get<bool>("value");
}

bool Capsule::is_full_dim() const {
    return ask("capsule.isFullDim", {{"C", *this}}).get<bool>("value");
}

bool Capsule::isequal(const Capsule &C2, double tol) const {
    return ask("capsule.isequal", {{"C", *this}, {"C2", C2}, {"tol", number(tol)}})
        .get<bool>("value");
}

// Sets: each op sends the capsules (and k, the Newton steps of the roots) and reads `C` back.

Capsule Capsule::convex_hull(const Capsule &C2, int k) const {
    return aux_set("capsule.convHull", {{"C", *this}, {"C2", C2}, {"k", k}});
}

Capsule Capsule::linear_map(const Tensor &M, int k) const {
    return aux_set("capsule.mtimes", {{"C", *this}, {"M", M}, {"k", k}});
}

Capsule Capsule::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set("capsule.lift", {{"C", *this}, {"N", N}, {"dims", dims}});
}

Capsule Capsule::enclose(const Tensor &M, const Capsule &C2, int k) const {
    return aux_set("capsule.enclose", {{"C", *this}, {"M", M}, {"C2", C2}, {"k", k}});
}

Capsule Capsule::enclose_points(const Tensor &points, int k) {
    return aux_set("capsule.enclosePoints", {{"points", points}, {"k", k}});
}

Capsule Capsule::empty(int64_t n) { return aux_set("capsule.empty", {{"n", n}}); }

Capsule Capsule::origin(int64_t n) { return aux_set("capsule.origin", {{"n", n}}); }

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
