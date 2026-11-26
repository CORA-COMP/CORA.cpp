// con_poly_zono - lean::ConPolyZono: the `conPolyZono.*` ops of the CORALean oracle
//
// The wire form is lean::PolyZonotope's {c, G, GI, Exp, E, id} with the rows {A, b, EC}: the
// polynomial part is written and read by lean::PolyZonotope, and so are the rows, as the
// polynomial zonotope of center b, generator columns A and exponents EC (A is CORA's matrix
// already). A reply carries no identifiers; gather() gives them 1..p, as CORA after an operation.
//
// Syntax:   lean::ConPolyZono S(cpz);   cora::ConPolyZono r = S.intersect(S2).gather();
// See also: con_poly_zono.h, poly_zonotope.cpp
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/con_poly_zono.h"
#include "cora/global/error.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The identifiers 1..p as a JSON array.
Json aux_ids(int64_t p) {
    Json ids = Json::array();
    for (int64_t i = 1; i <= p; ++i) ids.push(Json(i));
    return ids;
}

/// The number of factors of a wire set: the rows of its exponents.
int64_t aux_factors(const Json &j) { return std::stoll(j.at("Exp").at("rows").text); }


} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

// the polynomial part, then the rows as the polynomial zonotope of center b and columns A
ConPolyZono::ConPolyZono(const cora::ConPolyZono &S)
    : dtype_(Tensor(S.c().expand_dims(-1)).dtype()), json_(PolyZonotope(S.poly_part()).to_json()) {
    const cora::PolyLevelSet &C = S.constraints();
    const cora::PolyZonotope rows(C.b(), C.A().matrix_transpose(),
                                  C.b().zeros_like({0, C.num_rows()}), C.exponents(), S.ids(),
                                  detail::Verbatim{});
    const Json R = PolyZonotope(rows).to_json();
    json_.set("A", R.at("G")).set("b", R.at("c")).set("EC", R.at("Exp"));
}

ConPolyZono ConPolyZono::of_reply(const Reply &r, const char *key) {
    return {cora::lean::dtype(), r.get<Json>(key)};
}

cora::ConPolyZono ConPolyZono::gather() const {
    const int64_t p = aux_factors(json_);
    // the polynomial part through lean::PolyZonotope, with the identifiers 1..p
    Json part = json_;
    part.set("id", aux_ids(p));
    const cora::PolyZonotope P = PolyZonotope::from_json(dtype_, part).gather();
    // the rows: their slack must be zero, as the error box
    const cora::Tensor b = Tensor::from_json(dtype_, json_.at("b")).gather().squeeze(-1);
    if (const Json *slack = json_.find("b_error")) {
        for (const double v : Tensor::from_json(dtype_, *slack).gather().data())
            if (v != 0.0)
                CORA_FAIL(Error::wrong_value, "lean: a constraint row with a slack does not cross");
    }
    // the exponents EC through the same reader, a set of center b and columns A
    Json rows = Json::object();
    rows.set("c", json_.at("b")).set("G", json_.at("A")).set("Exp", json_.at("EC"));
    rows.set("GI", Tensor(b.zeros_like({0, b.shape()[0]})).to_json()).set("id", aux_ids(p));
    rows.set("E", Interval(cora::Interval(b.zeros_like(), b.zeros_like())).to_json());
    const cora::PolyZonotope R = PolyZonotope::from_json(dtype_, rows).gather();
    return {P, cora::PolyLevelSet(R.generators().matrix_transpose(), b, R.exponents(), p)};
}

// The conversions and constructors -----------------------------------------------------------

ConPolyZono ConPolyZono::from_zonotope(const cora::Zonotope &Z) {
    return of_reply(ask("conPolyZono.ofZonotope", {{"Z", Zonotope(Z)}}), "cPZ");
}

ConPolyZono ConPolyZono::from_poly_zonotope(const cora::PolyZonotope &P) {
    return of_reply(ask("conPolyZono.ofPolyZonotope", {{"P", PolyZonotope(P)}}));
}

ConPolyZono ConPolyZono::from_con_zonotope(const cora::ConZonotope &cz) {
    return of_reply(ask("conPolyZono.ofConZonotope", {{"cZ", ConZonotope(cz)}}));
}

ConPolyZono ConPolyZono::origin(int64_t n) {
    return of_reply(ask("conPolyZono.origin", {{"n", n}}));
}

ConPolyZono ConPolyZono::empty(int64_t n) { return of_reply(ask("conPolyZono.empty", {{"n", n}})); }

// Operations ---------------------------------------------------------------------------------

ConPolyZono ConPolyZono::linear_map(const Tensor &M) const {
    return of_reply(ask("conPolyZono.mtimes", {{"cPZ", Wire(json_, dtype_)}, {"M", M}}));
}

// the binary ops of two sets, each as the op of the oracle
ConPolyZono ConPolyZono::plus(const ConPolyZono &S) const {
    return of_reply(
        ask("conPolyZono.plus", {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

// the sum over shared factors, and the products
ConPolyZono ConPolyZono::exact_plus(const ConPolyZono &S) const {
    return of_reply(ask("conPolyZono.exactPlus",
                        {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

ConPolyZono ConPolyZono::cart_prod(const ConPolyZono &S) const {
    return of_reply(ask("conPolyZono.cartProd",
                        {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

ConPolyZono ConPolyZono::intersect(const ConPolyZono &S) const {
    return of_reply(
        ask("conPolyZono.and", {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

ConPolyZono ConPolyZono::linear_combination(const ConPolyZono &S) const {
    return of_reply(ask("conPolyZono.linComb",
                        {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

ConPolyZono ConPolyZono::convex_hull(const ConPolyZono &S) const {
    return of_reply(ask("conPolyZono.convHull",
                        {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}}));
}

// the coordinates, the reductions and the splits
ConPolyZono ConPolyZono::project(const std::vector<int64_t> &dims) const {
    return of_reply(ask("conPolyZono.project", {{"cPZ", Wire(json_, dtype_)}, {"dims", dims}}));
}

ConPolyZono ConPolyZono::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return of_reply(
        ask("conPolyZono.lift", {{"cPZ", Wire(json_, dtype_)}, {"N", N}, {"dims", dims}}));
}

// the reductions
ConPolyZono ConPolyZono::compact() const {
    return of_reply(ask("conPolyZono.compact", {{"cPZ", Wire(json_, dtype_)}}));
}

ConPolyZono ConPolyZono::reduce(int order) const {
    return of_reply(ask("conPolyZono.reduce",
                        {{"cPZ", Wire(json_, dtype_)}, {"order", order}, {"method", "girard"}}));
}

std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split(int64_t dim, const Interval &I) const {
    const Reply r =
        ask("conPolyZono.split", {{"cPZ", Wire(json_, dtype_)}, {"dim", dim}, {"I", I}});
    return {of_reply(r, "cPZ1"), of_reply(r, "cPZ2")};
}

ConPolyZono ConPolyZono::get_subset(int64_t id, double cen, double rad) const {
    return of_reply(
        ask("conPolyZono.getSubset", {{"cPZ", Wire(json_, dtype_)},
                                      {"factor", id},
                                      {"cen", Wire(Json(Tensor::encode(cen)), dtype_)},
                                      {"rad", Wire(Json(Tensor::encode(rad)), dtype_)}}));
}

// the halves along a factor
std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split_dep_factor(int64_t id) const {
    const Reply r =
        ask("conPolyZono.splitDepFactor", {{"cPZ", Wire(json_, dtype_)}, {"factor", id}});
    return {of_reply(r, "cPZ1"), of_reply(r, "cPZ2")};
}

std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split_longest_gen() const {
    const Reply r = ask("conPolyZono.splitLongestGen", {{"cPZ", Wire(json_, dtype_)}});
    return {of_reply(r, "cPZ1"), of_reply(r, "cPZ2")};
}

// The queries --------------------------------------------------------------------------------

int64_t ConPolyZono::dim() const {
    return ask("conPolyZono.dim", {{"cPZ", Wire(json_, dtype_)}}).get<int64_t>("value");
}

Tensor ConPolyZono::center() const {
    return ask("conPolyZono.center", {{"cPZ", Wire(json_, dtype_)}}).get<Tensor>("c");
}

Interval ConPolyZono::interval() const {
    return ask("conPolyZono.interval", {{"cPZ", Wire(json_, dtype_)}}).get<Interval>("I");
}

Zonotope ConPolyZono::zonotope() const {
    return ask("conPolyZono.zonotope", {{"cPZ", Wire(json_, dtype_)}}).get<Zonotope>("Z");
}

// the polynomial part, with the identifiers 1..p it lacks
PolyZonotope ConPolyZono::poly_zonotope() const {
    Json P = ask("conPolyZono.polyZonotope", {{"cPZ", Wire(json_, dtype_)}}).get<Json>("P");
    P.set("id", aux_ids(aux_factors(P)));
    return PolyZonotope::from_json(dtype_, P);
}

bool ConPolyZono::isequal(const ConPolyZono &S) const {
    return ask("conPolyZono.isequal",
               {{"cPZ", Wire(json_, dtype_)}, {"cPZ2", Wire(S.json_, dtype_)}})
        .get<bool>("value");
}

bool ConPolyZono::representsa_point() const {
    return ask("conPolyZono.representsa", {{"cPZ", Wire(json_, dtype_)}, {"type", "point"}})
        .get<bool>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
