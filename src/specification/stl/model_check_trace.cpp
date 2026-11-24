// model_check_trace - whether a sampled trace satisfies a formula
//
// Syntax:   bool res = phi.model_check_trace(x, t);
// Inputs:   x - the states of the trace (m, n), one per row
//           t - the times of the trace (m), uniformly sampled
// Outputs:  res - whether the trace satisfies the formula at its first sample
// See also: stl.h
//
// CORA: specification/@stl/modelCheckTrace.m
// CORA authors: Niklas Kochdumper, Benedikt Seidl
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/within_tol.h"
#include "cora/specification/stl.h"
#include "specification/stl/private/priv_trace.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The until operator ------------------------------------------------------------------------------

using Bools = std::vector<char>;

/// CORA's aux_until: the samples from which a witness of rhs within the window comes before lhs
/// fails.
Bools aux_until(const Bools &lhs, const Bools &rhs, const std::vector<double> &t, double from,
                double to) {
    const std::size_t m = t.size();
    std::vector<std::size_t> ind;
    for (std::size_t i = 0; i < m; ++i)
        if (t[i] >= from && t[i] <= to) ind.push_back(i);
    Bools res(m, 0);
    // the window slides one sample per start sample cnt
    for (std::size_t cnt = 0; !ind.empty() && ind.front() < m; ++cnt) {
        ind.erase(std::remove_if(ind.begin(), ind.end(), [&](std::size_t i) { return i >= m; }),
                  ind.end());
        // the first failure of lhs from cnt on, the first witness of rhs in the window
        std::size_t a1 = 0, a2 = 0;
        bool has1 = false, has2 = false;
        for (std::size_t k = cnt; k <= ind.back() && !has1; ++k)
            if (!lhs[k]) a1 = k, has1 = true;
        for (std::size_t j = 0; j < ind.size() && !has2; ++j)
            if (rhs[ind[j]]) a2 = ind.front() + j, has2 = true;
        if (has2 && (!has1 || (a2 < a1 && a1 + 1 >= ind.front()))) res[cnt] = 1;
        for (std::size_t &i : ind) ++i;
    }
    return res;
}

/// The negation of every entry.
Bools aux_not(Bools v) {
    for (char &c : v) c = !c;
    return v;
}

// The recursion over the formula ------------------------------------------------------------------

/// The values of the formula at every sample (CORA's aux_recursive).
Bools aux_recursive(const Stl &phi, const StlTrace &tr) {
    const StlKind k = phi.kind();
    const std::size_t m = tr.m;
    if (k == StlKind::true_) return Bools(m, 1);
    if (k == StlKind::false_) return Bools(m, 0);
    // a comparison reads both sides at every sample
    if (k == StlKind::comparison) {
        Bools res(m, 0);
        for (std::size_t i = 0; i < m; ++i) {
            const double a = priv_eval(phi.expr_lhs(), tr.names, tr.row(i));
            const double b = priv_eval(phi.expr_rhs(), tr.names, tr.row(i));
            const StlCmp op = phi.cmp();
            res[i] = op == StlCmp::lt   ? a < b
                     : op == StlCmp::le ? a <= b
                     : op == StlCmp::gt ? a > b
                                        : a >= b;
        }
        return res;
    }
    Bools lhs = aux_recursive(phi.lhs(), tr);
    if (k == StlKind::not_) return aux_not(lhs);
    if (k == StlKind::next) {
        // the first sample at the time of the shift, within tolerance
        std::size_t first = 0;
        while (first < m && !(tr.t[first] >= phi.from() || within_tol(tr.t[first], phi.from())))
            ++first;
        Bools res(m, 0);
        for (std::size_t i = first; i < m; ++i) res[i - first] = lhs[i];
        return res;
    }
    // F and G are until with true on the left
    if (k == StlKind::finally) return aux_until(Bools(m, 1), lhs, tr.t, phi.from(), phi.to());
    if (k == StlKind::globally)
        return aux_not(aux_until(Bools(m, 1), aux_not(lhs), tr.t, phi.from(), phi.to()));
    // the operators of two formulas
    const Bools rhs = aux_recursive(phi.rhs(), tr);
    if (k == StlKind::and_ || k == StlKind::or_) {
        for (std::size_t i = 0; i < m; ++i)
            lhs[i] = k == StlKind::and_ ? lhs[i] && rhs[i] : lhs[i] || rhs[i];
        return lhs;
    }
    if (k == StlKind::until) return aux_until(lhs, rhs, tr.t, phi.from(), phi.to());
    if (k != StlKind::release) CORA_FAIL(Error::not_supported, "model_check_trace: unknown node");
    Bools res = aux_not(aux_until(aux_not(lhs), aux_not(rhs), tr.t, phi.from(), phi.to()));
    // the last samples, from the first one at the end of the window on, do not satisfy it
    std::size_t idx = 0;
    while (idx < m && !(tr.t[idx] >= phi.to())) ++idx;
    const std::size_t count = idx < m ? std::min(m, idx + 1) : m;
    for (std::size_t i = m - count; i < m; ++i) res[i] = 0;
    return res;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Stl::model_check_trace(const Tensor &x, const std::vector<double> &t) const {
    detail::Entry entry;
    const StlTrace tr = priv_trace(*this, x, t, "model_check_trace");
    // only uniformly sampled traces
    if (tr.m >= 2) {
        double lo = INFINITY, hi = -INFINITY;
        for (std::size_t i = 1; i < tr.m; ++i) {
            lo = std::min(lo, t[i] - t[i - 1]);
            hi = std::max(hi, t[i] - t[i - 1]);
        }
        CORA_CHECK(std::abs(lo - hi) <= machine_eps(), Error::not_supported, [] {
            return std::string("model_check_trace: only uniformly sampled traces are supported");
        });
    }
    return aux_recursive(*this, tr)[0] != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
