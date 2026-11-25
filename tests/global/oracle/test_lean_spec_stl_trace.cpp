// test_lean_spec_stl_trace - Stl::model_check_trace against the CORALean oracle
//
// Bit for bit at nearest: random formulas over x1, x2 on random traces sampled every 1/8 (an exact
// step, so the trace is uniform), the windows on and between the samples. The predicates are
// evaluated at every sample in the same order on both sides (an integer power is the repeated
// product), then CORA's until windows decide the verdict at the first sample.
//
// ORACLE: specification/@stl/modelCheckTrace.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/stl.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/specification/stl.h"
#include "oracle_testing.h"

#include <algorithm>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A whole number in [0, k).
int aux_pick(oracle::Random &rnd, int k) {
    return std::min(k - 1, static_cast<int>(rnd.uniform(0.0, static_cast<double>(k))));
}

/// A time on the grid of 1/16 in [0, 2): half of them samples of the trace, half between two.
double aux_time(oracle::Random &rnd) { return aux_pick(rnd, 32) / 16.0; }

/// A random predicate: a linear or quadratic expression of x against a number.
Stl aux_predicate(oracle::Random &rnd, const std::vector<StlExpr> &x) {
    StlExpr e = x[aux_pick(rnd, 2)];
    if (aux_pick(rnd, 2) == 0) e = e.times(StlExpr(rnd.scalar())).plus(x[aux_pick(rnd, 2)]);
    if (aux_pick(rnd, 3) == 0) e = e.power(2.0 + aux_pick(rnd, 2));
    return Stl::compare(static_cast<StlCmp>(aux_pick(rnd, 4)), e, StlExpr(rnd.scalar()));
}

/// A random formula of the given depth, its windows on the grid of aux_time.
Stl aux_formula(oracle::Random &rnd, const std::vector<StlExpr> &x, int depth) {
    if (depth == 0) return aux_predicate(rnd, x);
    // an operator over two random sub-formulas, its window from..to
    const Stl a = aux_formula(rnd, x, depth - 1), b = aux_formula(rnd, x, depth - 1);
    const double from = aux_time(rnd), to = from + aux_time(rnd);
    switch (aux_pick(rnd, 7)) {
    case 0:
        return a.not_().and_(b);
    case 1:
        return a.or_(b);
    case 2:
        return a.until(b, StlInterval(from, to));
    case 3:
        return a.release(b, StlInterval(from, to));
    case 4:
        return a.finally(StlInterval(from, to));
    case 5:
        return a.globally(StlInterval(from, to));
    default:
        return a.next(from);
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<StlExpr> x = StlExpr::variables("x", 2);
    oracle::run("stl.modelCheckTrace", [&](oracle::Random &rnd) {
        for (int trial = 0; trial < 40; ++trial) {
            // a trace of m samples every 1/8, the states random
            const int64_t m = 8 + aux_pick(rnd, 24);
            std::vector<double> t(m);
            for (int64_t i = 0; i < m; ++i) t[i] = static_cast<double>(i) / 8;
            const Tensor states = rnd.matrix(m, 2);
            const Stl phi = aux_formula(rnd, x, 1 + aux_pick(rnd, 2));
            const Tensor times = Tensor::from_data(t, {m, 1});
            const bool theirs =
                lean::ask("stl.modelCheckTrace", {{"n", 2},
                                                  {"phi", lean::Wire(lean::stl_json(phi), "")},
                                                  {"x", lean::Tensor(states)},
                                                  {"t", lean::Tensor(times)},
                                                  {"tol", default_tol(DType::float64)}})
                    .get<bool>("value");
            oracle::check(phi.model_check_trace(states, t), theirs);
        }
    });
    return test::finish("lean spec stl trace");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
