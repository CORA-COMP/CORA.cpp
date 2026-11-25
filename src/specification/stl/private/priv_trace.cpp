// priv_trace - a sampled trace of states read against the variables of a formula
//
// Syntax:   StlTrace tr = priv_trace(phi, x, t, "robustness");
// See also: priv_trace.h
//
// CORA.cpp only: CORA reads the columns of x through contains(vars, name)
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "specification/stl/private/priv_trace.h"

#include <algorithm>
#include <map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The variables -----------------------------------------------------------------------------------

/// The variables below an expression: the names, and the largest index per stem.
void aux_expr_vars(const StlExpr &e, std::map<std::string, int64_t> &stems,
                   std::vector<std::string> &plain) {
    const StlExprKind k = e.kind();
    if (k == StlExprKind::number) return;
    if (k == StlExprKind::variable) {
        const std::string digits = std::to_string(e.index() + 1);
        const std::string &name = e.name();
        // x3 of index 2 belongs to the stem x; a name of another form stands alone
        if (name.size() > digits.size() &&
            name.compare(name.size() - digits.size(), digits.size(), digits) == 0) {
            int64_t &top = stems[name.substr(0, name.size() - digits.size())];
            top = std::max(top, e.index() + 1);
        } else {
            plain.push_back(name);
        }
        return;
    }
    aux_expr_vars(e.left(), stems, plain);
    if (k != StlExprKind::pow) aux_expr_vars(e.right(), stems, plain);
}

/// The variables below a formula.
void aux_formula_vars(const Stl &phi, std::map<std::string, int64_t> &stems,
                      std::vector<std::string> &plain) {
    const StlKind k = phi.kind();
    if (k == StlKind::true_ || k == StlKind::false_) return;
    if (k == StlKind::comparison) {
        aux_expr_vars(phi.expr_lhs(), stems, plain);
        aux_expr_vars(phi.expr_rhs(), stems, plain);
        return;
    }
    aux_formula_vars(phi.lhs(), stems, plain);
    const bool unary =
        k == StlKind::not_ || k == StlKind::next || k == StlKind::finally || k == StlKind::globally;
    if (!unary) aux_formula_vars(phi.rhs(), stems, plain);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<std::string> priv_trace_names(const Stl &phi) {
    std::map<std::string, int64_t> stems;
    std::vector<std::string> names;
    aux_formula_vars(phi, stems, names);
    for (const auto &[stem, top] : stems)
        for (int64_t i = 1; i <= top; ++i) names.push_back(stem + std::to_string(i));
    stl_sort_names(names);
    return names;
}

StlTrace priv_trace(const Stl &phi, const Tensor &x, const std::vector<double> &t, const char *op) {
    StlTrace tr;
    // the trace is a matrix with one time per row and a column for each variable
    tr.names = priv_trace_names(phi);
    if (!(x.ndim() == 2 && static_cast<std::size_t>(x.shape()[0]) == t.size() && !t.empty()))
        CORA_FAIL(Error::wrong_value,
                  std::string(op) + ": give x as (m, n) and one time per row, m >= 1");
    if (static_cast<std::size_t>(x.shape()[1]) < tr.names.size())
        CORA_FAIL(Error::wrong_value,
                  std::string(op) + ": x has fewer columns than the formula has variables");
    // the numbers on the host, where the algorithms of the trace read them
    tr.m = t.size();
    tr.n = static_cast<std::size_t>(x.shape()[1]);
    tr.x = x.data();
    tr.t = t;
    return tr;
}

// The values of the expressions -------------------------------------------------------------------

double priv_eval(const StlExpr &e, const std::vector<std::string> &names, const double *state) {
    const StlExprKind k = e.kind();
    if (k == StlExprKind::number) return e.value();
    if (k == StlExprKind::variable) {
        const auto it = std::find(names.begin(), names.end(), e.name());
        return state[it - names.begin()];
    }
    const double a = priv_eval(e.left(), names, state);
    if (k == StlExprKind::pow) {
        // the repeated product from 1, no libm pow (bit for bit with CORALean); 1/x^k below 0
        double prod = 1;
        const int64_t k_abs = e.exponent() < 0 ? -e.exponent() : e.exponent();
        for (int64_t i = 0; i < k_abs; ++i) prod *= a;
        return e.exponent() < 0 ? 1 / prod : prod;
    }
    const double b = priv_eval(e.right(), names, state);
    if (k == StlExprKind::add) return a + b;
    if (k == StlExprKind::sub) return a - b;
    return a * b;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
