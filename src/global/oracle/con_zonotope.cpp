// con_zonotope - lean::ConZonotope, a constrained zonotope of exact values, computed by CORALean
//
// See also: lean/zonotope.h, lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The constrained zonotope `cZ` of the response to `op` on the operands.
ConZonotope aux_set(const Reply &reply) { return reply.get<ConZonotope>("cZ"); }

/// The multipliers of one end as the oracle's {y, zu, zl}, three columns.
CORA_FROM_NUMBERS Json aux_end(const cora::ConZonotope::CertWeights &w) {
    const auto column = [](const std::vector<double> &v) {
        return Tensor::from(cora::Tensor::from_data(v, {int64_t(v.size()), 1})).to_json();
    };
    Json end = Json::object();
    end.set("y", column(w.y)).set("zu", column(w.zu)).set("zl", column(w.zl));
    return end;
}

/// The multipliers of one end in the response.
cora::ConZonotope::CertWeights aux_read_end(const Reply &end) {
    return {end.get<Tensor>("y").gather().data(), end.get<Tensor>("zu").gather().data(),
            end.get<Tensor>("zl").gather().data()};
}

/// The box and the multipliers of the response.
ConZonotope::Cert aux_cert(const Reply &reply) {
    ConZonotope::Cert out{reply.get<Interval>("I"), {}};
    for (const Reply &entry : reply.list("weights")) {
        out.weights.up.push_back(aux_read_end(entry.member("up")));
        out.weights.lo.push_back(aux_read_end(entry.member("lo")));
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction and crossing ------------------------------------------------------------------

// the wire form of an exact set is {c, G, A, b}
ConZonotope::ConZonotope(const cora::ConZonotope &cz)
    : dtype_(Tensor(cz.c()).dtype()), json_(Json::object()) {
    json_.set("c", Tensor(cz.c()).to_json()).set("G", Tensor(cz.generators()).to_json());
    json_.set("A", Tensor(cz.A()).to_json()).set("b", Tensor(cz.b()).to_json());
}

// the wire form of a cut: the factor rows C x <= d are the polytope's inequalities
ConZonotope::ConZonotope(const cora::ConZonotope::Cut &cut)
    : dtype_(Tensor(cut.c).dtype()), json_(Json::object()) {
    json_.set("c", Tensor(cut.c).to_json()).set("G", Tensor(cut.G).to_json());
    Json P = Json::object();
    P.set("A", Tensor(cut.C).to_json()).set("b", Tensor(cut.d).to_json());
    P.set("Ae", Tensor(cut.A).to_json()).set("be", Tensor(cut.b).to_json());
    json_.set("P", std::move(P));
}

bool ConZonotope::exact() const {
    const Json *P = json_.find("P");
    if (!P) return true;
    if (!zero_valued(json_.at("E"))) return false;
    return zero_valued(P->at("E")) && zero_valued(P->at("be_error")) &&
           zero_valued(P->at("b_error")) && P->at("A").at("rows").text == "0";
}

// The equalities of an oracle result are P.Ae x = P.be; a set made of {c, G, A, b} has A, b.
cora::ConZonotope ConZonotope::nominal() const {
    const Json *P = json_.find("P");
    const Tensor c = Tensor::from_json(dtype_, json_.at("c"));
    const Tensor G = Tensor::from_json(dtype_, json_.at("G"));
    if (!P)
        return {c.gather(), G.gather(), Tensor::from_json(dtype_, json_.at("A")).gather(),
                Tensor::from_json(dtype_, json_.at("b")).gather()};
    return {c.gather(), G.gather(), Tensor::from_json(dtype_, P->at("Ae")).gather(),
            Tensor::from_json(dtype_, P->at("be")).gather()};
}

CORA_FROM_NUMBERS std::pair<cora::Tensor, cora::Tensor> ConZonotope::inequalities() const {
    // an input has no inequality rows
    const Json *P = json_.find("P");
    const int64_t m = Tensor::from_json(dtype_, json_.at("G")).cols();
    if (!P) return {cora::Tensor::zeros({0, m}), cora::Tensor::zeros({0, 1})};
    return {Tensor::from_json(dtype_, P->at("A")).gather(),
            Tensor::from_json(dtype_, P->at("b")).gather()};
}

cora::ConZonotope ConZonotope::gather() const {
    if (!exact())
        CORA_FAIL(Error::wrong_value,
                  "lean: the oracle's set has an inequality row or a nonempty error box");
    return nominal();
}

// the wire form of a result is the oracle's own object

Json ConZonotope::to_json() const { return json_; }

ConZonotope ConZonotope::from_json(const std::string &dtype, const Json &j) { return {dtype, j}; }

// Operations ---------------------------------------------------------------------------------

Interval ConZonotope::interval() const {
    return ask("conZonotope.interval", {{"cZ", *this}}).get<Interval>("I");
}

Interval ConZonotope::interval_tight(int64_t k) const {
    return ask("conZonotope.intervalTight", {{"cZ", *this}, {"k", k}}).get<Interval>("I");
}


ConZonotope::Cert ConZonotope::interval_cert(const cora::ConZonotope::Certificate &w) const {
    // the multipliers of both ends of every coordinate, as the oracle's `y`
    Json y = Json::array();
    for (std::size_t i = 0; i < w.up.size(); ++i) {
        Json both = Json::object();
        both.set("up", aux_end(w.up[i])).set("lo", aux_end(w.lo[i]));
        y.push(std::move(both));
    }
    return aux_cert(ask("conZonotope.intervalCert", {{"cZ", *this}, {"y", Wire(y, dtype_)}}));
}

ConZonotope::Cert ConZonotope::interval_cert() const {
    return aux_cert(ask("conZonotope.intervalCert", {{"cZ", *this}}));
}

// The bounds and predicates --------------------------------------------------------------------

double ConZonotope::support_function(const Tensor &dir) const {
    return ask("conZonotope.supportFunc", {{"cZ", *this}, {"dir", dir}, {"type", "upper"}})
        .get<double>("value");
}

bool ConZonotope::is_bounded() const {
    return ask("conZonotope.isBounded", {{"cZ", *this}}).get<bool>("value");
}

// The certified predicates: a true is a proof; the default certificate is a right inverse of G.
bool ConZonotope::contains(const Tensor &point, double tol) const {
    return ask("conZonotope.contains", {{"cZ", *this}, {"point", point}, {"tol", number(tol)}})
        .get<bool>("contains");
}

bool ConZonotope::is_full_dim(double tol) const {
    return ask("conZonotope.isFullDim", {{"cZ", *this}, {"tol", number(tol)}}).get<bool>("value");
}

ConZonotope ConZonotope::intersect(const ConZonotope &cz2) const {
    return aux_set(ask("conZonotope.and", {{"cZ", *this}, {"cZ2", cz2}}));
}

// each of these sends the operands as {cZ, cZ2, ...} and reads the set `cZ` back

ConZonotope ConZonotope::convex_hull(const ConZonotope &cz2) const {
    return aux_set(ask("conZonotope.convHull", {{"cZ", *this}, {"cZ2", cz2}}));
}

ConZonotope ConZonotope::cart_prod(const ConZonotope &cz2) const {
    return aux_set(ask("conZonotope.cartProd", {{"cZ", *this}, {"cZ2", cz2}}));
}

ConZonotope ConZonotope::con_intersect(const ConZonotope &cz2, const Tensor &R) const {
    return aux_set(ask("conZonotope.conIntersect", {{"cZ", *this}, {"cZ2", cz2}, {"R", R}}));
}

ConZonotope ConZonotope::enclose(const Tensor &M, const ConZonotope &cz2) const {
    return aux_set(ask("conZonotope.enclose", {{"cZ", *this}, {"M", M}, {"cZ2", cz2}}));
}

// the strips and the maps carry their own operands

ConZonotope ConZonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                         const Tensor &Lambda) const {
    return aux_set(ask("conZonotope.intersectStrip",
                       {{"cZ", *this}, {"C", C}, {"phi", phi}, {"y", y}, {"Lambda", Lambda}}));
}

ConZonotope ConZonotope::cub_map(const std::vector<std::vector<Tensor>> &T) const {
    return aux_set(ask("conZonotope.cubMap", {{"cZ", *this}, {"T", T}}));
}

ConZonotope ConZonotope::compact(const std::string &method, double tol) const {
    return aux_set(
        ask("conZonotope.compact", {{"cZ", *this}, {"method", method}, {"tol", number(tol)}}));
}

ConZonotope ConZonotope::empty(int64_t n) { return aux_set(ask("conZonotope.empty", {{"n", n}})); }

ConZonotope ConZonotope::origin(int64_t n) {
    return aux_set(ask("conZonotope.origin", {{"n", n}}));
}

// Maps and predicates ------------------------------------------------------------------------

ConZonotope ConZonotope::plus(const ConZonotope &cz2) const {
    return aux_set(ask("conZonotope.plus", {{"cZ", *this}, {"cZ2", cz2}}));
}

ConZonotope ConZonotope::mtimes(const Tensor &M) const {
    return aux_set(ask("conZonotope.mtimes", {{"cZ", *this}, {"M", M}}));
}

// the index maps send the dimensions as 0-based indices
ConZonotope ConZonotope::project(const std::vector<int64_t> &dims) const {
    return aux_set(ask("conZonotope.project", {{"cZ", *this}, {"dims", dims}}));
}

ConZonotope ConZonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    return aux_set(ask("conZonotope.lift", {{"cZ", *this}, {"N", N}, {"dims", dims}}));
}

// the maps with an order or matrices as operands
ConZonotope ConZonotope::reduce(int order) const {
    return aux_set(ask("conZonotope.reduce", {{"cZ", *this}, {"order", order}}));
}

ConZonotope ConZonotope::quad_map(const std::vector<Tensor> &Q) const {
    return aux_set(ask("conZonotope.quadMap", {{"cZ", *this}, {"Q", Q}}));
}

// a split answers with the two halves cZ1 and cZ2
std::pair<ConZonotope, ConZonotope> ConZonotope::split(int64_t dim) const {
    const Reply reply = ask("conZonotope.split", {{"cZ", *this}, {"dim", dim}});
    return {reply.get<ConZonotope>("cZ1"), reply.get<ConZonotope>("cZ2")};
}

// the predicates answer with a Boolean `value`; a proof of disjointness reads as Certified
bool ConZonotope::isequal(const ConZonotope &cz2, double tol) const {
    return ask("conZonotope.isequal", {{"cZ", *this}, {"cZ2", cz2}, {"tol", number(tol)}})
        .get<bool>("value");
}

bool ConZonotope::representsa_empty() const {
    return ask("conZonotope.representsa", {{"cZ", *this}, {"type", "emptySet"}, {"tol", number(0)}})
        .get<bool>("value");
}

Certified ConZonotope::is_intersecting(const ConZonotope &cz2, double tol) const {
    return ask("conZonotope.isIntersecting", {{"cZ", *this}, {"cZ2", cz2}, {"tol", number(tol)}})
        .proved_disjoint();
}

// the conversions of a zonotope and of a box

ConZonotope ConZonotope::from_zonotope(const Zonotope &Z) {
    return aux_set(ask("zonotope.conZonotope", {{"Z", Z}}));
}

ConZonotope ConZonotope::from_interval(const Interval &I) {
    return aux_set(ask("interval.conZonotope", {{"I", I}}));
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
