// taylm - lean::Taylm: each op sends the model (and its second operand) and reads the model back
//
// A model goes as {"vars", "maxOrder", "terms": [{"c", "e"}], "rem": {"inf", "sup"}}, the
// coefficients and the remainder as exact texts; the response has the error box "Ierr" as well.
//
// Syntax:   see cora/global/oracle/taylm.h
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle's `taylm.*` ops
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/oracle/taylm.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The interval [lo, hi] as {"inf", "sup"} texts.
Json aux_interval(const std::array<double, 2> &I) {
    Json j = Json::object();
    j.set("inf", Tensor::encode(I[0])).set("sup", Tensor::encode(I[1]));
    return j;
}

/// The interval {"inf", "sup"} of a reply.
std::array<double, 2> aux_read_interval(const Reply &reply) {
    return {reply.get<double>("inf"), reply.get<double>("sup")};
}

/// The model as an operand.
Wire aux_wire(const Taylm &T) { return {T.to_json(), dtype()}; }

} // namespace


// ===========================================  MAIN  =========================================== //

// The model ----------------------------------------------------------------------------------

Taylm::Taylm(const cora::Taylm &T)
    : vars_(T.num_vars()), max_order_(T.max_order()),
      terms_(T.coefficients().begin(), T.coefficients().end()), rem_(T.remainder()) {}

Json Taylm::to_json() const {
    // each term as its coefficient's text and its exponents
    Json terms = Json::array();
    for (const auto &[m, c] : terms_) {
        Json e = Json::array();
        for (const int x : m) e.push(int64_t(x));
        Json term = Json::object();
        term.set("c", Tensor::encode(c)).set("e", std::move(e));
        terms.push(std::move(term));
    }
    Json j = Json::object();
    j.set("vars", vars_).set("maxOrder", int64_t(max_order_)).set("terms", std::move(terms));
    j.set("rem", aux_interval(rem_));
    return j;
}

// the reply's model, its terms in the oracle's order
Taylm Taylm::from_reply(const Reply &reply) { return from_model(reply.member("T")); }

Taylm Taylm::from_model(const Reply &T) {
    Taylm out;
    out.vars_ = T.get<int64_t>("vars");
    out.max_order_ = int(T.get<int64_t>("maxOrder"));
    for (const Reply &term : T.list("terms")) {
        cora::Taylm::Monomial m;
        for (const int64_t e : term.get<std::vector<int64_t>>("e")) m.push_back(int(e));
        out.terms_.emplace_back(std::move(m), term.get<double>("c"));
    }
    out.rem_ = aux_read_interval(T.member("rem"));
    out.err_ = aux_read_interval(T.member("Ierr"));
    return out;
}

// Sums ---------------------------------------------------------------------------------------

Taylm Taylm::plus(const Taylm &Q) const {
    return from_reply(ask("taylm.plus", {{"T", aux_wire(*this)}, {"Q", aux_wire(Q)}}));
}
Taylm Taylm::plus(double s) const {
    return from_reply(ask("taylm.plus", {{"T", aux_wire(*this)}, {"s", s}}));
}
Taylm Taylm::minus(const Taylm &Q) const {
    return from_reply(ask("taylm.minus", {{"T", aux_wire(*this)}, {"Q", aux_wire(Q)}}));
}
Taylm Taylm::minus(double s) const {
    return from_reply(ask("taylm.minus", {{"T", aux_wire(*this)}, {"s", s}}));
}
Taylm Taylm::uminus() const { return from_reply(ask("taylm.uminus", {{"T", aux_wire(*this)}})); }
Taylm Taylm::uplus() const { return from_reply(ask("taylm.uplus", {{"T", aux_wire(*this)}})); }

// Products -----------------------------------------------------------------------------------

Taylm Taylm::times(const Taylm &Q, int max_order) const {
    return from_reply(
        ask("taylm.times",
            {{"T", aux_wire(*this)}, {"Q", aux_wire(Q)}, {"maxOrder", int64_t(max_order)}}));
}
Taylm Taylm::times(double s) const {
    return from_reply(ask("taylm.times", {{"T", aux_wire(*this)}, {"s", s}}));
}
Taylm Taylm::power(int k) const {
    return from_reply(ask("taylm.power", {{"T", aux_wire(*this)}, {"k", int64_t(k)}}));
}

// The range ----------------------------------------------------------------------------------

std::array<double, 2> Taylm::bounds() const {
    const Interval I = ask("taylm.interval", {{"T", aux_wire(*this)}}).get<Interval>("I");
    const auto [lo, hi] = I.gather();
    return {lo.data()[0], hi.data()[0]};
}

// The constructors ---------------------------------------------------------------------------

Taylm taylm_constant(double c, int64_t vars, int max_order) {
    return Taylm::from_model(
        ask("taylm.constant", {{"c", c}, {"vars", vars}, {"maxOrder", int64_t(max_order)}})
            .member("T"));
}

Taylm taylm_from_range(double lo, double hi, int64_t vars, int max_order) {
    return Taylm::from_model(
        ask("taylm.fromRange",
            {{"lo", lo}, {"hi", hi}, {"vars", vars}, {"maxOrder", int64_t(max_order)}})
            .member("T"));
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
