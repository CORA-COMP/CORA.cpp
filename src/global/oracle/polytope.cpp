// polytope - lean::Polytope, a polytope of exact values in either form, computed by CORALean
//
// See also: lean/con_zonotope.h, lean/request.h

#include "cora/global/oracle/polytope.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The polytope `P` of the response to `op` on the operands.
Polytope aux_set(const Reply &reply) { return reply.get<Polytope>("P"); }

// The member `key` of the wire form as a tensor of the dtype, back in doubles.
cora::Tensor aux_part(const std::string &dtype, const Json &j, const char *key) {
    return Tensor::from_json(dtype, j.at(key)).gather();
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

// the wire form of an exact set is {A, b, Ae, be} or {V}
Polytope::Polytope(const cora::Polytope &P) : dtype_(), json_(Json::object()) {
    if (P.is_h_rep()) {
        dtype_ = Tensor(P.A()).dtype();
        json_.set("A", Tensor(P.A()).to_json()).set("b", Tensor(P.b()).to_json());
        json_.set("Ae", Tensor(P.Ae()).to_json()).set("be", Tensor(P.be()).to_json());
    } else {
        dtype_ = Tensor(P.V()).dtype();
        json_.set("V", Tensor(P.V()).to_json());
    }
}

bool Polytope::exact() const {
    for (const char *key : {"E", "b_error", "be_error"})
        if (const Json *error = json_.find(key))
            if (!zero_valued(*error)) return false;
    return true;
}

CORA_FROM_NUMBERS cora::Polytope Polytope::nominal() const {
    if (vertices()) return cora::Polytope(aux_part(dtype_, json_, "V"));
    // the equalities are optional in a response of the oracle
    const int64_t n = Tensor::from_json(dtype_, json_.at("A")).cols();
    const cora::Tensor Ae = json_.find("Ae") ? aux_part(dtype_, json_, "Ae")
                                             : cora::Tensor::zeros({0, n}),
                       be = json_.find("be") ? aux_part(dtype_, json_, "be")
                                             : cora::Tensor::zeros({0, 1});
    return {aux_part(dtype_, json_, "A"), aux_part(dtype_, json_, "b"), Ae, be};
}

cora::Polytope Polytope::gather() const {
    if (!exact())
        CORA_FAIL(Error::wrong_value, "lean: the oracle's polytope has a nonempty error box");
    return nominal();
}

// Operations ---------------------------------------------------------------------------------

Polytope Polytope::inf(int64_t n) { return aux_set(ask("polytope.Inf", {{"n", n}})); }

Polytope Polytope::empty(int64_t n) { return aux_set(ask("polytope.empty", {{"n", n}})); }

Polytope Polytope::intersect(const Polytope &P2) const {
    return aux_set(ask("polytope.and", {{"P", *this}, {"P2", P2}}));
}

Polytope Polytope::cart_prod(const Polytope &P2) const {
    return aux_set(ask("polytope.cartProd", {{"P", *this}, {"P2", P2}}));
}

// a test of containment answers a Boolean `contains`

bool Polytope::contains(const Interval &I, double tol) const {
    return ask("polytope.contains", {{"P", *this}, {"I", I}, {"tol", number(tol)}})
        .get<bool>("contains");
}

bool Polytope::contains(const Zonotope &Z, double tol) const {
    return ask("polytope.contains", {{"P", *this}, {"Z", Z}, {"tol", number(tol)}})
        .get<bool>("contains");
}

Interval Polytope::interval() const {
    return ask("polytope.interval", {{"P", *this}}).get<Interval>("I");
}

Interval Polytope::center() const {
    return ask("polytope.center", {{"P", *this}}).get<Interval>("I");
}

// the vertex form: the box and the mean, the hull, the conversion and the cloud

Polytope Polytope::convex_hull(const Polytope &P2) const {
    return aux_set(ask("polytope.convHull", {{"P", *this}, {"P2", P2}}));
}

ConZonotope Polytope::con_zonotope() const {
    return ask("polytope.conZonotope", {{"P", *this}}).get<ConZonotope>("cZ");
}

Polytope Polytope::enclose_points(const Tensor &points) {
    return aux_set(ask("polytope.enclosePoints", {{"points", points}, {"method", "outer"}}));
}

// the conversion of a box

Polytope Polytope::from_interval(const Interval &I) {
    return aux_set(ask("interval.polytope", {{"I", I}}));
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
