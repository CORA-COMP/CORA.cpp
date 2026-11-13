// poly_zonotope - lean::PolyZonotope, a polynomial zonotope computed by CORALean
//
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/poly_zonotope.h"

#include "cora/global/error.h"
#include "cora/global/oracle/request.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The wire form of the exponents: p rows, h columns, row by row.
Json aux_exponents_json(const cora::Exponents &Exp, std::size_t p) {
    Json data = Json::array();
    for (std::size_t i = 0; i < p; ++i)
        for (const std::vector<int> &col : Exp) data.push(Json(int64_t(col[i])));
    return Json::object()
        .set("rows", int64_t(p))
        .set("cols", int64_t(Exp.size()))
        .set("data", data);
}

/// The exponents of the wire form, one column per generator.
cora::Exponents aux_exponents_of(const Json &j) {
    const int64_t p = std::stoll(j.at("rows").text), h = std::stoll(j.at("cols").text);
    const Json &data = j.at("data");
    cora::Exponents Exp(std::size_t(h), std::vector<int>(std::size_t(p), 0));
    for (int64_t i = 0; i < p; ++i)
        for (int64_t l = 0; l < h; ++l)
            Exp[std::size_t(l)][std::size_t(i)] =
                std::stoi(data.items[std::size_t(i * h + l)].text);
    return Exp;
}

/// The polynomial zonotope of the reply member `key`.
PolyZonotope aux_set(const char *op, std::initializer_list<Operand> operands) {
    return ask(op, operands).get<PolyZonotope>("P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction -------------------------------------------------------------------------------

PolyZonotope::PolyZonotope(const cora::PolyZonotope &P)
    : c_(Tensor::from(P.center())), G_(Tensor::from(P.generators())),
      GI_(Tensor::from(P.independent_generators())), E_(Interval::origin(P.center().shape()[0])),
      Exp_(P.exponents()), id_(P.ids()) {}

PolyZonotope::PolyZonotope(Tensor c, Tensor G, Tensor GI, Interval E, cora::Exponents Exp,
                           std::vector<int64_t> id)
    : c_(std::move(c)), G_(std::move(G)), GI_(std::move(GI)), E_(std::move(E)),
      Exp_(std::move(Exp)), id_(std::move(id)) {}

// the columns as given, so that the oracle's constructor does the merge and the order
PolyZonotope PolyZonotope::construct(const cora::Tensor &c, const cora::Tensor &G,
                                     const cora::Tensor &GI, const cora::Exponents &Exp,
                                     const std::vector<int64_t> &id) {
    const PolyZonotope raw(Tensor::from(c), Tensor::from(G), Tensor::from(GI),
                           Interval::origin(c.shape()[0]), Exp, id);
    return aux_set("polyZonotope.constructor", {{"P", raw}});
}

PolyZonotope PolyZonotope::from_zonotope(const cora::Zonotope &Z) {
    return aux_set("zonotope.polyZonotope", {{"Z", Zonotope(Z)}});
}

PolyZonotope PolyZonotope::from_interval(const cora::Interval &I) {
    return aux_set("interval.polyZonotope", {{"I", Interval(I)}});
}

// Operations ---------------------------------------------------------------------------------

// the sums: a polynomial zonotope, a zonotope, a box, a point
PolyZonotope PolyZonotope::plus(const PolyZonotope &Q) const {
    return aux_set("polyZonotope.plus", {{"P", *this}, {"Q", Q}});
}

PolyZonotope PolyZonotope::plus(const cora::Zonotope &Z) const {
    return aux_set("polyZonotope.plus", {{"P", *this}, {"Z", Zonotope(Z)}});
}

PolyZonotope PolyZonotope::plus(const cora::Interval &I) const {
    return aux_set("polyZonotope.plus", {{"P", *this}, {"I", Interval(I)}});
}

PolyZonotope PolyZonotope::plus(const cora::Tensor &v) const {
    return aux_set("polyZonotope.plus", {{"P", *this}, {"v", Tensor::from(v)}});
}

// the map, the reduction and the hull
PolyZonotope PolyZonotope::linear_map(const cora::Tensor &M) const {
    return aux_set("polyZonotope.mtimes", {{"M", Tensor::from(M)}, {"P", *this}});
}

PolyZonotope PolyZonotope::reduce(int order) const {
    return aux_set("polyZonotope.reduce", {{"P", *this}, {"order", order}, {"method", "girard"}});
}

PolyZonotope PolyZonotope::enclose(const PolyZonotope &Q) const {
    return aux_set("polyZonotope.enclose", {{"P", *this}, {"Q", Q}});
}

// the conversions and the dimension
Zonotope PolyZonotope::zonotope() const {
    return ask("polyZonotope.zonotope", {{"P", *this}}).get<Zonotope>("Z");
}

Interval PolyZonotope::interval() const {
    return ask("polyZonotope.interval", {{"P", *this}}).get<Interval>("I");
}

int64_t PolyZonotope::dim() const {
    return ask("polyZonotope.dim", {{"P", *this}}).get<int64_t>("value");
}

// Crossing -----------------------------------------------------------------------------------

cora::PolyZonotope PolyZonotope::gather() const {
    const auto [inf, sup] = E_.gather();
    const auto zero = [](const cora::Tensor &T) {
        const std::vector<double> x = T.data();
        return std::all_of(x.begin(), x.end(), [](double v) { return v == 0.0; });
    };
    if (!zero(inf) || !zero(sup))
        CORA_FAIL(Error::wrong_value,
                  "lean: a polynomial zonotope with an error box does not cross");
    return {c_.gather(), G_.gather(), GI_.gather(), Exp_, id_, detail::Verbatim{}};
}

// the wire form is {c, G, GI, Exp, E, id}
Json PolyZonotope::to_json() const {
    Json ids = Json::array();
    for (const int64_t i : id_) ids.push(Json(i));
    Json j = Json::object();
    j.set("c", c_.to_json())
        .set("G", G_.to_json())
        .set("GI", GI_.to_json())
        .set("Exp", aux_exponents_json(Exp_, id_.size()))
        .set("E", E_.to_json())
        .set("id", ids);
    return j;
}

PolyZonotope PolyZonotope::from_json(const std::string &dtype, const Json &j) {
    std::vector<int64_t> id;
    for (const Json &i : j.at("id").items) id.push_back(std::stoll(i.text));
    return {Tensor::from_json(dtype, j.at("c")),  Tensor::from_json(dtype, j.at("G")),
            Tensor::from_json(dtype, j.at("GI")), Interval::from_json(dtype, j.at("E")),
            aux_exponents_of(j.at("Exp")),        std::move(id)};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
