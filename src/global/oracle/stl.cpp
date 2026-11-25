// stl - the wire form of STL formulas and STL intervals for the oracle's stl ops (CORALean)
//
// See also: lean/stl.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/stl.h"

#include "cora/global/error.h"
#include "cora/global/oracle/tensor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The nodes ---------------------------------------------------------------------------------------

/// A node {"op", "args", more...} with the members in alphabetical order, as the oracle writes it.
Json aux_node(const char *op, std::vector<Json> args,
              std::vector<std::pair<std::string, Json>> more = {}) {
    Json list = Json::array();
    for (Json &a : args) list.push(std::move(a));
    more.emplace_back("args", std::move(list));
    more.emplace_back("op", Json(op));
    // the members sorted by name: args, exponent, from, op, to
    std::sort(more.begin(), more.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    Json node = Json::object();
    for (auto &[key, value] : more) node.set(key, std::move(value));
    return node;
}

/// CORA's name of a comparison.
const char *aux_cmp(StlCmp op) {
    switch (op) {
    case StlCmp::lt:
        return "<";
    case StlCmp::le:
        return "<=";
    case StlCmp::gt:
        return ">";
    case StlCmp::ge:
        return ">=";
    }
    CORA_FAIL(Error::report_to_dev, "stl_json: unknown comparison");
}

/// The largest variable index below an expression, -1 for none.
int64_t aux_top(const StlExpr &e) {
    if (e.kind() == StlExprKind::variable) return e.index();
    if (e.kind() == StlExprKind::number) return -1;
    const int64_t l = aux_top(e.left());
    return e.kind() == StlExprKind::pow ? l : std::max(l, aux_top(e.right()));
}

/// The largest variable index below a formula, -1 for none.
int64_t aux_top(const Stl &phi) {
    const StlKind k = phi.kind();
    if (k == StlKind::true_ || k == StlKind::false_) return -1;
    if (k == StlKind::comparison) return std::max(aux_top(phi.expr_lhs()), aux_top(phi.expr_rhs()));
    const bool unary =
        k == StlKind::not_ || k == StlKind::next || k == StlKind::finally || k == StlKind::globally;
    return unary ? aux_top(phi.lhs()) : std::max(aux_top(phi.lhs()), aux_top(phi.rhs()));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The expressions and formulas --------------------------------------------------------------------

Json stl_json(const StlExpr &e) {
    Json leaf = Json::object();
    switch (e.kind()) {
    case StlExprKind::variable:
        return leaf.set("x", Json(e.index()));
    case StlExprKind::number:
        return leaf.set("const", Tensor::encode(e.value()));
    case StlExprKind::add:
        return aux_node("add", {stl_json(e.left()), stl_json(e.right())});
    case StlExprKind::sub:
        return aux_node("sub", {stl_json(e.left()), stl_json(e.right())});
    case StlExprKind::mul:
        return aux_node("mul", {stl_json(e.left()), stl_json(e.right())});
    case StlExprKind::pow:
        return aux_node("pow", {stl_json(e.left())}, {{"exponent", Json(e.exponent())}});
    }
    CORA_FAIL(Error::report_to_dev, "stl_json: unknown expression kind");
}

Json stl_json(const Stl &phi) {
    // the window of a temporal node: from (and to, but not for next)
    const auto window = [&](bool to) {
        std::vector<std::pair<std::string, Json>> w{{"from", Tensor::encode(phi.from())}};
        if (to) w.emplace_back("to", Tensor::encode(phi.to()));
        return w;
    };
    switch (phi.kind()) {
    case StlKind::true_:
        return aux_node("true", {});
    case StlKind::false_:
        return aux_node("false", {});
    case StlKind::comparison:
        return aux_node(aux_cmp(phi.cmp()), {stl_json(phi.expr_lhs()), stl_json(phi.expr_rhs())});
    case StlKind::not_:
        return aux_node("not", {stl_json(phi.lhs())});
    case StlKind::and_:
        return aux_node("and", {stl_json(phi.lhs()), stl_json(phi.rhs())});
    case StlKind::or_:
        return aux_node("or", {stl_json(phi.lhs()), stl_json(phi.rhs())});
    // the temporal operators with their windows
    case StlKind::until:
        return aux_node("until", {stl_json(phi.lhs()), stl_json(phi.rhs())}, window(true));
    case StlKind::release:
        return aux_node("release", {stl_json(phi.lhs()), stl_json(phi.rhs())}, window(true));
    case StlKind::finally:
        return aux_node("finally", {stl_json(phi.lhs())}, window(true));
    case StlKind::globally:
        return aux_node("globally", {stl_json(phi.lhs())}, window(true));
    case StlKind::next:
        return aux_node("next", {stl_json(phi.lhs())}, window(false));
    }
    CORA_FAIL(Error::report_to_dev, "stl_json: unknown formula kind");
}

// The intervals -----------------------------------------------------------------------------------

Json stl_json(const StlInterval &I) {
    Json j = Json::object();
    if (I.is_empty_object()) return j.set("empty", true);
    j.set("lb", Tensor::encode(I.lower())).set("lc", I.left_closed()).set("rc", I.right_closed());
    return j.set("ub", std::isinf(I.upper()) ? Json("inf") : Json(Tensor::encode(I.upper())));
}

int64_t stl_variables(const Stl &phi) { return std::max<int64_t>(1, aux_top(phi) + 1); }

// The requests ------------------------------------------------------------------------------------

Reply ask_stl(const char *op, const Stl &phi, std::vector<Operand> more) {
    more.insert(more.begin(),
                {Operand{"n", Wire(stl_variables(phi))}, Operand{"phi", Wire(stl_json(phi), "")}});
    return ask(op, more);
}

Reply ask_stl_interval(const char *op, const StlInterval &I, std::vector<Operand> more) {
    more.insert(more.begin(), Operand{"I", Wire(stl_json(I), "")});
    return ask(op, more);
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
