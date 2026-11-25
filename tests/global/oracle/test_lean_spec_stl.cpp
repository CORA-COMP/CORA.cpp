// test_lean_spec_stl - the formulas of Stl against the CORALean oracle's stl ops
//
// Bit for bit at nearest on random formulas over x1, x2: the node builders, the comparisons and
// the expression arithmetic, desugar, the negation, conjunctive and disjunctive normal forms,
// eliminateTrueFalse, combineNext, the numbered parts of assignIdentifiers, getClauses,
// getVariables, isPredicate, isLinearPredicate, the latest and earliest times, isequal, and
// getTimes on the numbered formula. A formula is compared as its wire form (lean/stl.h), so the
// structure and every number must be the same; the identifiers are not part of it.
//
// ORACLE: specification/@stl/and.m
// ORACLE: specification/@stl/assignIdentifiers.m
// ORACLE: specification/@stl/combineNext.m
// ORACLE: specification/@stl/conjunctiveNormalForm.m
// ORACLE: specification/@stl/desugar.m
// ORACLE: specification/@stl/disjunctiveNormalForm.m
// ORACLE: specification/@stl/eliminateTrueFalse.m
// ORACLE: specification/@stl/finally.m
// ORACLE: specification/@stl/ge.m
// ORACLE: specification/@stl/getClauses.m
// ORACLE: specification/@stl/getTimes.m
// ORACLE: specification/@stl/getVariables.m
// ORACLE: specification/@stl/globally.m
// ORACLE: specification/@stl/gt.m
// ORACLE: specification/@stl/implies.m
// ORACLE: specification/@stl/isLinearPredicate.m
// ORACLE: specification/@stl/isPredicate.m
// ORACLE: specification/@stl/isequal.m
// ORACLE: specification/@stl/le.m
// ORACLE: specification/@stl/lt.m
// ORACLE: specification/@stl/maximumTime.m
// ORACLE: specification/@stl/minimumTime.m
// ORACLE: specification/@stl/minus.m
// ORACLE: specification/@stl/negationNormalForm.m
// ORACLE: specification/@stl/next.m
// ORACLE: specification/@stl/not.m
// ORACLE: specification/@stl/or.m
// ORACLE: specification/@stl/plus.m
// ORACLE: specification/@stl/power.m
// ORACLE: specification/@stl/release.m
// ORACLE: specification/@stl/stl.m
// ORACLE: specification/@stl/times.m
// ORACLE: specification/@stl/until.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/stl.h"
#include "cora/global/oracle/tensor.h"
#include "cora/specification/stl.h"
#include "oracle_testing.h"

#include <algorithm>
#include <iostream>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Random formulas ---------------------------------------------------------------------------------

/// A whole number in [0, k).
int aux_pick(oracle::Random &rnd, int k) {
    return std::min(k - 1, static_cast<int>(rnd.uniform(0.0, static_cast<double>(k))));
}

/// A random expression over x of the given depth.
StlExpr aux_expr(oracle::Random &rnd, const std::vector<StlExpr> &x, int depth) {
    if (depth == 0)
        return aux_pick(rnd, 3) == 0 ? StlExpr(rnd.scalar()) : x[aux_pick(rnd, int(x.size()))];
    const StlExpr a = aux_expr(rnd, x, depth - 1), b = aux_expr(rnd, x, depth - 1);
    switch (aux_pick(rnd, 4)) {
    case 0:
        return a.plus(b);
    case 1:
        return a.minus(b);
    case 2:
        return a.times(b);
    default:
        return a.power(2.0);
    }
}

/// A random comparison: an expression against a number or against another expression.
Stl aux_predicate(oracle::Random &rnd, const std::vector<StlExpr> &x) {
    const StlExpr a = aux_expr(rnd, x, aux_pick(rnd, 2));
    const StlExpr b = aux_pick(rnd, 2) == 0 ? StlExpr(rnd.scalar()) : aux_expr(rnd, x, 1);
    return Stl::compare(static_cast<StlCmp>(aux_pick(rnd, 4)), a, b);
}

/// A closed window [a, a + w].
StlInterval aux_window(oracle::Random &rnd) {
    const double a = rnd.uniform(0.0, 1.0);
    return StlInterval(a, a + rnd.uniform(0.0, 2.0));
}

/// A random formula of the given depth, with true and false among the leaves.
Stl aux_formula(oracle::Random &rnd, const std::vector<StlExpr> &x, int depth) {
    if (depth == 0) {
        const int leaf = aux_pick(rnd, 8);
        return leaf == 0 ? Stl(true) : leaf == 1 ? Stl(false) : aux_predicate(rnd, x);
    }
    // an operator over two random sub-formulas (the unary ones take the first)
    const Stl a = aux_formula(rnd, x, depth - 1), b = aux_formula(rnd, x, depth - 1);
    switch (aux_pick(rnd, 9)) {
    case 0:
        return a.not_();
    case 1:
        return a.and_(b);
    case 2:
        return a.or_(b);
    // the temporal operators, each with a random window
    case 3:
        return a.until(b, aux_window(rnd));
    case 4:
        return a.release(b, aux_window(rnd));
    case 5:
        return a.finally(aux_window(rnd));
    case 6:
        return a.globally(aux_window(rnd));
    case 7:
        return a.next(rnd.uniform(0.0, 1.0));
    default:
        return a.implies(b);
    }
}

// The comparisons ---------------------------------------------------------------------------------

/// The same wire form, under the name `what`.
void aux_same(const std::string &what, const lean::Json &mine, const lean::Json &theirs) {
    const bool ok = mine.dump() == theirs.dump();
    if (!ok)
        std::cerr << "  " << what << ":\n    CORA.cpp " << mine.dump() << "\n    oracle   "
                  << theirs.dump() << "\n";
    test::check(ok, what + ": the oracle's structure");
}

/// A list of formulas as a JSON array.
lean::Json aux_list(const std::vector<Stl> &list) {
    lean::Json a = lean::Json::array();
    for (const Stl &phi : list) a.push(lean::stl_json(phi));
    return a;
}

/// The operand `key` holding a formula or an expression.
lean::Operand aux_operand(const char *key, const lean::Json &j) { return {key, lean::Wire(j, "")}; }

/// The times of get_times in the oracle's form: per identifier the windows and the points.
std::pair<lean::Json, lean::Json> aux_times(const StlTimes &times) {
    lean::Json ints = lean::Json::array(), points = lean::Json::array();
    for (std::size_t k = 0; k < times.time_int.size(); ++k) {
        lean::Json w = lean::Json::array(), p = lean::Json::array();
        for (const StlInterval &I : times.time_int[k]) w.push(lean::stl_json(I));
        for (const double t : times.time_point[k]) p.push(lean::Tensor::encode(t));
        ints.push(std::move(w));
        points.push(std::move(p));
    }
    return {ints, points};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<StlExpr> x = StlExpr::variables("x", 2);
    constexpr int kTrials = 24;
    // the constructor: true, false and the variables
    oracle::run("stl.stl", [](oracle::Random &) {
        for (const bool value : {true, false})
            aux_same("stl", lean::stl_json(Stl(value)),
                     lean::ask("stl.stl", {{"value", value}}).get<lean::Json>("phi"));
        lean::Json vars = lean::Json::array();
        for (const StlExpr &v : StlExpr::variables("x", 3)) vars.push(lean::stl_json(v));
        aux_same("stl", vars, lean::ask("stl.stl", {{"n", 3}}).get<lean::Json>("vars"));
    });
    // the logic and temporal builders of two formulas
    oracle::run("stl.and", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const Stl a = aux_formula(rnd, x, 1), b = aux_formula(rnd, x, 1);
            const StlInterval w = aux_window(rnd);
            const lean::Operand psi = aux_operand("psi", lean::stl_json(b));
            const lean::Operand from{"from", lean::Wire(w.lower())},
                to{"to", lean::Wire(w.upper())};
            // phi = a and psi = b, n counted over both
            const Stl ab = a.and_(b);
            const auto ask2 = [&](const char *op, std::vector<lean::Operand> more) {
                more.insert(more.begin(), {lean::Operand{"n", lean::Wire(lean::stl_variables(ab))},
                                           aux_operand("phi", lean::stl_json(a)), psi});
                return lean::ask(op, more).get<lean::Json>("phi");
            };
            aux_same("and", lean::stl_json(a.and_(b)), ask2("stl.and", {}));
            aux_same("or", lean::stl_json(a.or_(b)), ask2("stl.or", {}));
            aux_same("implies", lean::stl_json(a.implies(b)), ask2("stl.implies", {}));
            aux_same("until", lean::stl_json(a.until(b, w)), ask2("stl.until", {from, to}));
            aux_same("release", lean::stl_json(a.release(b, w)), ask2("stl.release", {from, to}));
            // the builders of one formula
            aux_same("not", lean::stl_json(a.not_()),
                     lean::ask_stl("stl.not", a).get<lean::Json>("phi"));
            aux_same("finally", lean::stl_json(a.finally(w)),
                     lean::ask_stl("stl.finally", a, {from, to}).get<lean::Json>("phi"));
            aux_same("globally", lean::stl_json(a.globally(w)),
                     lean::ask_stl("stl.globally", a, {from, to}).get<lean::Json>("phi"));
            aux_same("next", lean::stl_json(a.next(w.lower())),
                     lean::ask_stl("stl.next", a, {from}).get<lean::Json>("phi"));
        }
    });
    // the comparisons and the arithmetic of two expressions
    oracle::run("stl.lt", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            // an expression against a number or a second expression
            const StlExpr a = aux_expr(rnd, x, aux_pick(rnd, 2)),
                          b = aux_pick(rnd, 2) == 0 ? StlExpr(rnd.scalar()) : aux_expr(rnd, x, 1);
            const auto ask2 = [&](const char *op, const char *key) {
                return lean::ask(op, {{"n", 2},
                                      aux_operand("e1", lean::stl_json(a)),
                                      aux_operand("e2", lean::stl_json(b))})
                    .get<lean::Json>(key);
            };
            // the comparison nodes, then the arithmetic nodes
            aux_same("lt", lean::stl_json(a.lt(b)), ask2("stl.lt", "phi"));
            aux_same("le", lean::stl_json(a.le(b)), ask2("stl.le", "phi"));
            aux_same("gt", lean::stl_json(a.gt(b)), ask2("stl.gt", "phi"));
            aux_same("ge", lean::stl_json(a.ge(b)), ask2("stl.ge", "phi"));
            aux_same("plus", lean::stl_json(a.plus(b)), ask2("stl.plus", "e"));
            aux_same("minus", lean::stl_json(a.minus(b)), ask2("stl.minus", "e"));
            aux_same("times", lean::stl_json(a.times(b)), ask2("stl.times", "e"));
            aux_same(
                "power", lean::stl_json(a.power(3.0)),
                lean::ask("stl.power", {{"n", 2}, aux_operand("e1", lean::stl_json(a)), {"k", 3}})
                    .get<lean::Json>("e"));
        }
    });
    // the rewritings of a formula
    oracle::run("stl.desugar", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const Stl phi = aux_formula(rnd, x, 1 + aux_pick(rnd, 3));
            const auto formula = [&](const char *op, std::vector<lean::Operand> more = {}) {
                return lean::ask_stl(op, phi, std::move(more)).get<lean::Json>("phi");
            };
            aux_same("desugar", lean::stl_json(phi.desugar()), formula("stl.desugar"));
            aux_same("negationNormalForm", lean::stl_json(phi.negation_normal_form()),
                     formula("stl.negationNormalForm"));
            aux_same("eliminateTrueFalse", lean::stl_json(phi.eliminate_true_false()),
                     formula("stl.eliminateTrueFalse"));
            aux_same("combineNext", lean::stl_json(phi.combine_next()), formula("stl.combineNext"));
            aux_same("conjunctiveNormalForm", lean::stl_json(phi.conjunctive_normal_form()),
                     formula("stl.conjunctiveNormalForm"));
            aux_same("disjunctiveNormalForm", lean::stl_json(phi.disjunctive_normal_form()),
                     formula("stl.disjunctiveNormalForm"));
            // the normal forms of a formula in negation normal form already
            const Stl nnf = phi.negation_normal_form();
            const lean::Operand given{"nnf", true};
            aux_same(
                "conjunctiveNormalForm(nnf)",
                lean::stl_json(nnf.conjunctive_normal_form(StlNnf::given)),
                lean::ask_stl("stl.conjunctiveNormalForm", nnf, {given}).get<lean::Json>("phi"));
            aux_same(
                "disjunctiveNormalForm(nnf)",
                lean::stl_json(nnf.disjunctive_normal_form(StlNnf::given)),
                lean::ask_stl("stl.disjunctiveNormalForm", nnf, {given}).get<lean::Json>("phi"));
        }
    });
    // the parts of a formula and its times
    oracle::run("stl.assignIdentifiers", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const Stl phi = aux_formula(rnd, x, 1 + aux_pick(rnd, 3));
            aux_same("assignIdentifiers", aux_list(phi.assign_identifiers().second),
                     lean::ask_stl("stl.assignIdentifiers", phi).get<lean::Json>("list"));
            for (const char *type : {"cnf", "dnf"})
                aux_same(
                    std::string("getClauses ") + type, aux_list(phi.get_clauses(type)),
                    lean::ask_stl("stl.getClauses", phi, {{"type", type}}).get<lean::Json>("list"));
            // the variables: x<i+1> is index i
            std::vector<std::string> theirs;
            for (const int64_t i :
                 lean::ask_stl("stl.getVariables", phi).get<std::vector<int64_t>>("vars"))
                theirs.push_back("x" + std::to_string(i + 1));
            test::check(phi.get_variables() == theirs, "getVariables: the oracle's names");
            oracle::check("isPredicate", phi.is_predicate(),
                          lean::ask_stl("stl.isPredicate", phi).get<bool>("value"));
            oracle::check("isLinearPredicate", phi.is_linear_predicate(),
                          lean::ask_stl("stl.isLinearPredicate", phi).get<bool>("value"));
            oracle::check("maximumTime", phi.maximum_time(),
                          lean::ask_stl("stl.maximumTime", phi).get<double>("value"));
            oracle::check("minimumTime", phi.minimum_time(),
                          lean::ask_stl("stl.minimumTime", phi).get<double>("value"));
            // the times of the numbered formula
            const auto [ints, points] = aux_times(phi.assign_identifiers().first.get_times());
            const lean::Reply times = lean::ask_stl("stl.getTimes", phi);
            aux_same("getTimes (windows)", ints, times.get<lean::Json>("timeInt"));
            aux_same("getTimes (points)", points, times.get<lean::Json>("timePoint"));
        }
    });
    // isequal: a formula, the same with & and | swapped, and another
    oracle::run("stl.isequal", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const Stl a = aux_formula(rnd, x, 2), b = aux_formula(rnd, x, 2);
            for (const Stl &other : {b.and_(a), a.and_(b), a.or_(b), b})
                oracle::check("isequal", a.and_(b).isequal(other),
                              lean::ask_stl("stl.isequal", a.and_(b),
                                            {aux_operand("psi", lean::stl_json(other))})
                                  .get<bool>("value"));
        }
    });
    return test::finish("lean spec stl");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
