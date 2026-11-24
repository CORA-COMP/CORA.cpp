// test_lean_expr_elementary - Expr elementary functions against the CORALean oracle
//
// log, asin, acos, atan, sinh, cosh, tanh, asinh and acosh of a variable, enclosed over a box: the
// range is the box operation of the interval, so the bounds agree bit for bit with the oracle's
// at dtype nearest. atanh stays out as in test_lean_interval_elementary (the oracle keeps the
// bracket of the unchecked roundings there).
//
// ORACLE: contSet/@interval/log.m
// ORACLE: contSet/@interval/asin.m
// ORACLE: contSet/@interval/acos.m
// ORACLE: contSet/@interval/atan.m
// ORACLE: contSet/@interval/sinh.m
// ORACLE: contSet/@interval/cosh.m
// ORACLE: contSet/@interval/tanh.m
// ORACLE: contSet/@interval/asinh.m
// ORACLE: contSet/@interval/acosh.m
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/expr.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kTerms = 20;

/// One function on boxes of the domain [low, high]: the range of f(x) over the box, the oracle's.
template <class F, class G>
void aux_function(const std::string &name, double low, double high, F f, G theirs) {
    test::oracle::run("interval." + name, [&](test::oracle::Random &rnd) {
        for (const int n : {1, 4, 6}) {
            std::vector<double> a(n), b(n);
            for (int i = 0; i < n; ++i) {
                a[i] = rnd.uniform(low, high);
                b[i] = rnd.uniform(low, high);
                if (a[i] > b[i]) std::swap(a[i], b[i]);
            }
            const Interval I(test::vec(a), test::vec(b));
            const Range box(I.inf().reshape({n, 1}), I.sup().reshape({n, 1}));
            const Range r = f(Expr::var(0)).enclose({box});
            test::oracle::check(Interval(r.lo.reshape({n}), r.hi.reshape({n})),
                                theirs(lean::Interval(I)));
        }
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    using L = const lean::Interval &;
    using E = const Expr &;
    aux_function(
        "log", 0.01, 50, [](E x) { return log(x); }, [](L I) { return I.log(kTerms, kTerms); });
    aux_function(
        "atan", -10, 10, [](E x) { return atan(x); }, [](L I) { return I.atan(kTerms, kTerms); });
    // the hyperbolic functions
    aux_function("sinh", -5, 5, [](E x) { return sinh(x); }, [](L I) { return I.sinh(kTerms); });
    aux_function("cosh", -5, 5, [](E x) { return cosh(x); }, [](L I) { return I.cosh(kTerms); });
    aux_function("tanh", -5, 5, [](E x) { return tanh(x); }, [](L I) { return I.tanh(kTerms); });
    // the inverse functions
    aux_function(
        "asin", -1, 1, [](E x) { return asin(x); }, [](L I) { return I.asin(kTerms, kTerms); });
    aux_function(
        "acos", -1, 1, [](E x) { return acos(x); }, [](L I) { return I.acos(kTerms, kTerms); });
    aux_function(
        "asinh", -10, 10, [](E x) { return asinh(x); },
        [](L I) { return I.asinh(kTerms, kTerms); });
    aux_function(
        "acosh", 1, 30, [](E x) { return acosh(x); }, [](L I) { return I.acosh(kTerms, kTerms); });
    return test::finish("lean expr elementary");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
