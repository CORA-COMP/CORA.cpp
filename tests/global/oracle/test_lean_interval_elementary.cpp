// test_lean_interval_elementary - the elementary functions of Interval against the CORALean oracle
//
// exp, log, tan, atan, sinh, cosh, tanh, asin, acos, asinh and acosh of a box. At dtype nearest
// the oracle takes the platform's function at the end points (its ElemOverride), where CORA.cpp on
// the lean backend does: the bounds agree bit for bit. The boxes are drawn from the domain of each
// function. sin, cos, sqrt and atanh stay out: CORALean bounds sin and cos by a Lipschitz form
// where CORA takes the exact case analysis, and sqrt and atanh keep the bracket of the unchecked
// roundings that the neural-network relaxations of CORA copy.
//
// ORACLE: contSet/@interval/exp.m
// ORACLE: contSet/@interval/log.m
// ORACLE: contSet/@interval/tan.m
// ORACLE: contSet/@interval/atan.m
// ORACLE: contSet/@interval/sinh.m
// ORACLE: contSet/@interval/cosh.m
// ORACLE: contSet/@interval/tanh.m
// ORACLE: contSet/@interval/asin.m
// ORACLE: contSet/@interval/acos.m
// ORACLE: contSet/@interval/asinh.m
// ORACLE: contSet/@interval/acosh.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
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

/// A box of n dimensions with bounds drawn from [low, high].
Interval aux_box(test::oracle::Random &rnd, int n, double low, double high) {
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rnd.uniform(low, high);
        b[i] = rnd.uniform(low, high);
        if (a[i] > b[i]) std::swap(a[i], b[i]);
    }
    return {test::vec(a), test::vec(b)};
}

/// One function on boxes of the domain [low, high]: the same bounds as the oracle's.
template <class F, class G>
void aux_function(const std::string &name, double low, double high, F mine, G theirs) {
    test::oracle::run("interval." + name, [&](test::oracle::Random &rnd) {
        for (const int n : {1, 4, 6}) {
            const Interval I = aux_box(rnd, n, low, high);
            test::oracle::check(mine(I), theirs(lean::Interval(I)));
        }
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    using L = const lean::Interval &;
    using M = const Interval &;
    aux_function("exp", -5, 5, [](M I) { return I.exp(); }, [](L I) { return I.exp(kTerms); });
    aux_function(
        "log", 0.01, 50, [](M I) { return I.log(); }, [](L I) { return I.log(kTerms, kTerms); });
    // the functions of a bounded domain next
    aux_function("tan", -1.5, 1.5, [](M I) { return I.tan(); }, [](L I) { return I.tan(kTerms); });
    aux_function(
        "atan", -10, 10, [](M I) { return I.atan(); }, [](L I) { return I.atan(kTerms, kTerms); });
    aux_function("sinh", -5, 5, [](M I) { return I.sinh(); }, [](L I) { return I.sinh(kTerms); });
    aux_function("cosh", -5, 5, [](M I) { return I.cosh(); }, [](L I) { return I.cosh(kTerms); });
    aux_function("tanh", -5, 5, [](M I) { return I.tanh(); }, [](L I) { return I.tanh(kTerms); });
    // the inverse functions
    aux_function(
        "asin", -1, 1, [](M I) { return I.asin(); }, [](L I) { return I.asin(kTerms, kTerms); });
    aux_function(
        "acos", -1, 1, [](M I) { return I.acos(); }, [](L I) { return I.acos(kTerms, kTerms); });
    aux_function(
        "asinh", -10, 10, [](M I) { return I.asinh(); },
        [](L I) { return I.asinh(kTerms, kTerms); });
    aux_function(
        "acosh", 1, 30, [](M I) { return I.acosh(); }, [](L I) { return I.acosh(kTerms, kTerms); });
    return test::finish("lean interval elementary");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
