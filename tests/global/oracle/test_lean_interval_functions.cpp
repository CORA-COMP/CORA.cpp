// test_lean_interval_functions - acos, asin, atan, acosh, asinh, atanh and atan2 of intervals
// against the CORALean oracle, as agreement within a bound
//
// The oracle's floats are a tuned sound enclosure (bisection with k steps, series with p terms),
// not libm, so no bit-exact test exists. CORA.cpp's bounds (libm, as CORA) lie in the enclosure
// (sound mode). For acos, asin, atan and atan2 they also differ from the oracle's bounds by at
// most kAgree, relative, at p = 20, k = 60, in both modes (measured: 1e-13). The oracle's acosh,
// asinh and atanh go through a logarithm whose bound is loose (up to 0.3 relative at p = k = 20,
// and not better for larger p), so for these only the enclosure is checked.

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

using namespace cora;

namespace {

constexpr double kAgree = 1e-11;
constexpr int kTerms = 20, kSteps = 60;

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

/// Whether the bounds of a and b differ by at most kAgree, relatively.
bool aux_agree(const std::string &what, const Interval &a, const lean::Interval &b) {
    bool ok = true;
    const std::pair<std::vector<double>, std::vector<double>> mine{a.inf().data(), a.sup().data()};
    const std::pair<Tensor, Tensor> theirs = b.gather();
    const std::vector<double> lo = theirs.first.data(), hi = theirs.second.data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        for (const auto [x, y] : {std::pair{mine.first[i], lo[i]}, {mine.second[i], hi[i]}})
            if (std::abs(x - y) > kAgree * std::max(1.0, std::abs(y))) {
                std::cerr << "  " << what << ": " << x << " in CORA.cpp, " << y
                          << " in the oracle\n";
                ok = false;
            }
    return ok;
}

/// One function: CORA.cpp's result on boxes drawn from [low, high] against the oracle's, in
/// agreement within kAgree if `tight`, else only inside the sound enclosure.
void aux_function(const std::string &name, bool tight, double low, double high,
                  const std::function<Interval(const Interval &)> &mine,
                  const std::function<lean::Interval(const lean::Interval &)> &theirs) {
    test::oracle::run("interval." + name, [&](test::oracle::Random &rnd) {
        const bool sound = test::oracle::mode() == test::oracle::Mode::sound;
        if (!tight && !sound) return; // nearest gives no enclosure to check against
        for (const int n : {1, 4}) {
            const Interval I = aux_box(rnd, n, low, high);
            const Interval R = mine(I);
            const lean::Interval O = theirs(lean::Interval(I));
            if (tight) test::check(aux_agree(name, R, O), name + ": within the agreement bound");
            if (sound) test::oracle::check_enclosed(name, R, O);
        }
    });
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    aux_function(
        "acos", true, -1, 1, [](const Interval &I) { return I.acos(); },
        [](const lean::Interval &I) { return I.acos(kTerms, kSteps); });
    aux_function(
        "asin", true, -1, 1, [](const Interval &I) { return I.asin(); },
        [](const lean::Interval &I) { return I.asin(kTerms, kSteps); });
    aux_function(
        "atan", true, -5, 5, [](const Interval &I) { return I.atan(); },
        [](const lean::Interval &I) { return I.atan(kTerms, kSteps); });
    aux_function(
        "acosh", false, 1, 40, [](const Interval &I) { return I.acosh(); },
        [](const lean::Interval &I) { return I.acosh(kTerms, kSteps); });
    aux_function(
        "asinh", false, -5, 5, [](const Interval &I) { return I.asinh(); },
        [](const lean::Interval &I) { return I.asinh(kTerms, kSteps); });
    aux_function(
        "atanh", false, -0.9, 0.9, [](const Interval &I) { return I.atanh(); },
        [](const lean::Interval &I) { return I.atanh(kTerms, kSteps); });
    // atan2 off the branch cut (x > 0), where the corner hull is an enclosure
    aux_function(
        "atan2", true, 0.5, 4, [](const Interval &I) { return I.atan2(I.plus(I)); },
        [](const lean::Interval &I) { return I.atan2(I.plus(I), kTerms, kSteps); });
    return test::finish("lean interval functions");
}
