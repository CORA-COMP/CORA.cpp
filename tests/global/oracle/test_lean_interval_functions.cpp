// test_lean_interval_functions - the elementary functions of Interval against the CORALean oracle
//
// exp, log, cos, cosh, acos, asin, atan, acosh, asinh, atanh, atan2, sqrt, tan and tanh, as
// agreement within a bound. The oracle's floats are a tuned sound enclosure (bisection with k
// steps, series with p terms), not libm, so no bit-exact test exists. CORA.cpp's bounds (libm, as
// CORA) lie in the enclosure (sound mode). For acos and atan2 they also differ from the oracle's
// bounds by at most kAgree, relative, at p = 20, k = 24 (measured: 1e-6). exp, log and cosh
// differ by at most kLooseAgree (measured: 3e-5 for exp, 1e-5 for log on [1, 40]; the oracle is
// looser than libm), tanh too (3e-7). sqrt and tan are taken on boxes where the oracle's
// principal branch and CORA agree (inf >= 0, inside (-1.4, 1.4)): they differ by at most
// kSqrtAgree and kTanAgree (measured: 2e-16, 2e-14 at p = 20). Only the enclosure is checked for
// acosh, asinh and atanh (their logarithm is loose: up to 0.3 relative), asin and atan (kept
// short for the run time) and cos (the oracle's midpoint form is far wider: 0.63 against 0.70).
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kAgree = 1e-6, kLooseAgree = 1e-3, kSqrtAgree = 1e-15, kTanAgree = 1e-13;
constexpr int kTerms = 20, kSteps = 24, kFewSteps = 6;

/// A box of n dimensions with bounds drawn from [low, high].
Interval aux_box(test::oracle::Random &rnd, int n, double low, double high) {
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rnd.uniform(low, high);
        b[i] = rnd.uniform(low, high);
        if (a[i] > b[i]) std::swap(a[i], b[i]);
    }
    return {test::column(a), test::column(b)};
}

/// Whether the bounds of a and b differ by at most `agree`, relatively.
bool aux_agree(const std::string &what, const Interval &a, const lean::Interval &b, double agree) {
    bool ok = true;
    const std::pair<std::vector<double>, std::vector<double>> mine{a.inf().data(), a.sup().data()};
    const std::pair<Tensor, Tensor> theirs = b.gather();
    const std::vector<double> lo = theirs.first.data(), hi = theirs.second.data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        for (const auto &[x, y] : {std::pair{mine.first[i], lo[i]}, {mine.second[i], hi[i]}})
            if (std::abs(x - y) > agree * std::max(1.0, std::abs(y))) {
                std::cerr << "  " << what << ": " << x << " in CORA.cpp, " << y
                          << " in the oracle\n";
                ok = false;
            }
    return ok;
}

/// One function: CORA.cpp's result on boxes drawn from [low, high] against the oracle's, in
/// agreement within `agree` (relative) if it is positive, else only inside the sound enclosure.
void aux_function(const std::string &name, double agree, double low, double high,
                  const std::function<Interval(const Interval &)> &mine,
                  const std::function<lean::Interval(const lean::Interval &)> &theirs) {
    test::oracle::run("interval." + name, [&](test::oracle::Random &rnd) {
        const bool sound = test::oracle::mode() == test::oracle::Mode::sound;
        if (agree == 0 && !sound) return; // nearest gives no enclosure to check against
        for (const int n : {1}) {
            const Interval I = aux_box(rnd, n, low, high);
            const Interval R = mine(I);
            const lean::Interval O = theirs(lean::Interval(I));
            if (agree > 0)
                test::check(aux_agree(name, R, O, agree), name + ": within the agreement bound");
            if (sound) test::oracle::check_enclosed(name, R, O);
        }
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    aux_function(
        "exp", kLooseAgree, -5, 5, [](const Interval &I) { return I.exp(); },
        [](const lean::Interval &I) { return I.exp(kTerms); });
    // the oracle's log needs 1 <= inf <= 2^p / 2
    aux_function(
        "log", kLooseAgree, 1, 40, [](const Interval &I) { return I.log(); },
        [](const lean::Interval &I) { return I.log(kTerms, kSteps); });
    aux_function(
        "cos", 0, -6, 6, [](const Interval &I) { return I.cos(); },
        [](const lean::Interval &I) { return I.cos(kTerms); });
    aux_function(
        "cosh", kLooseAgree, -3, 3, [](const Interval &I) { return I.cosh(); },
        [](const lean::Interval &I) { return I.cosh(kTerms); });
    aux_function(
        "acos", kAgree, -1, 1, [](const Interval &I) { return I.acos(); },
        [](const lean::Interval &I) { return I.acos(kTerms, kSteps); });
    aux_function(
        "asin", 0, -1, 1, [](const Interval &I) { return I.asin(); },
        [](const lean::Interval &I) { return I.asin(kTerms, kFewSteps); });
    aux_function(
        "atan", 0, -5, 5, [](const Interval &I) { return I.atan(); },
        [](const lean::Interval &I) { return I.atan(kTerms, kFewSteps); });
    // inverse hyperbolic cosine
    aux_function(
        "acosh", 0, 1, 40, [](const Interval &I) { return I.acosh(); },
        [](const lean::Interval &I) { return I.acosh(kTerms, kFewSteps); });
    aux_function(
        "asinh", 0, -5, 5, [](const Interval &I) { return I.asinh(); },
        [](const lean::Interval &I) { return I.asinh(kTerms, kFewSteps); });
    aux_function(
        "atanh", 0, -0.9, 0.9, [](const Interval &I) { return I.atanh(); },
        [](const lean::Interval &I) { return I.atanh(kTerms, kFewSteps); });
    // atan2 off the branch cut (x > 0), where the corner hull is an enclosure
    aux_function(
        "atan2", kAgree, 0.5, 4, [](const Interval &I) { return I.atan2(I.plus(I)); },
        [](const lean::Interval &I) { return I.atan2(I.plus(I), kTerms, kSteps); });
    aux_function(
        "sqrt", kSqrtAgree, 0, 40, [](const Interval &I) { return I.sqrt(); },
        [](const lean::Interval &I) { return I.sqrt(kTerms); });
    aux_function(
        "tan", kTanAgree, -1.4, 1.4, [](const Interval &I) { return I.tan(); },
        [](const lean::Interval &I) { return I.tan(kTerms); });
    aux_function(
        "tanh", kLooseAgree, -3, 3, [](const Interval &I) { return I.tanh(); },
        [](const lean::Interval &I) { return I.tanh(kTerms); });
    return test::finish("lean interval functions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
