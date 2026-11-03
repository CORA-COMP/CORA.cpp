// test_lean_interval_elementary - sign, sin and sinh against the CORALean oracle, as enclosure and
// agreement within a bound
//
// CORA evaluates sin and sinh with the platform math library, and CORALean's float definitions are
// tuned enclosures (a midpoint form for sin, exp-based halvings for sinh), so no bit-exact test
// exists. CORA.cpp's boxes lie in the sound enclosures. Where the enclosure is tight they also
// differ from the oracle's bounds by at most kAgree, relative: for sin on boxes inside
// [-1.2, 1.2], where the cosine is certified positive and the oracle takes the corner values
// (measured 3e-15 at p = 20); for sinh by at most kLooseAgree (measured 1.3e-5 on [-3, 3]). For
// sign, CORA's lower bound is 1 on a positive box where the oracle's is 0 (a float comparison
// certifies only <=): the boxes agree bit for bit across 0, and CORA.cpp's lies inside elsewhere.

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cora;

namespace {

constexpr double kAgree = 1e-12, kLooseAgree = 1e-3;
constexpr int kTerms = 20;

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
    const std::pair<Tensor, Tensor> theirs = b.gather();
    const std::vector<double> mine_lo = a.inf().data(), mine_hi = a.sup().data();
    const std::vector<double> lo = theirs.first.data(), hi = theirs.second.data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        for (const auto [x, y] : {std::pair{mine_lo[i], lo[i]}, {mine_hi[i], hi[i]}})
            if (std::abs(x - y) > agree * std::max(1.0, std::abs(y))) {
                std::cerr << "  " << what << ": " << x << " in CORA.cpp, " << y
                          << " in the oracle\n";
                ok = false;
            }
    return ok;
}

/// One function on boxes drawn from [low, high]: inside the sound enclosure, and within `agree`
/// of the oracle's bounds.
template <class F, class G>
void aux_function(const std::string &name, double agree, double low, double high, F mine,
                  G theirs) {
    test::oracle::run("interval." + name, [&](test::oracle::Random &rnd) {
        const bool sound = test::oracle::mode() == test::oracle::Mode::sound;
        for (const int n : {1, 4}) {
            const Interval I = aux_box(rnd, n, low, high);
            const Interval R = mine(I);
            const lean::Interval O = theirs(lean::Interval(I));
            test::check(aux_agree(name, R, O, agree), name + ": within the agreement bound");
            if (sound) test::oracle::check_enclosed(name, R, O);
        }
    });
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    aux_function(
        "sin", kAgree, -1.2, 1.2, [](const Interval &I) { return I.sin(); },
        [](const lean::Interval &I) { return I.sin(kTerms); });
    aux_function(
        "sinh", kLooseAgree, -3, 3, [](const Interval &I) { return I.sinh(); },
        [](const lean::Interval &I) { return I.sinh(kTerms); });
    // across 0 the oracle's bounds are -1 and 1 as CORA's; the sound mode adds every other box
    test::oracle::run("interval.sign", [](test::oracle::Random &rnd) {
        for (const int n : {1, 4}) {
            const bool sound = test::oracle::mode() == test::oracle::Mode::sound;
            const Interval I = aux_box(rnd, n, -2, 2);
            const Interval across(I.inf().abs().negative() - 0.5, I.sup().abs() + 0.5);
            test::oracle::check_exact("sign across 0", across.sign(),
                                      lean::Interval(across).sign());
            if (sound) test::oracle::check_enclosed("sign", I.sign(), lean::Interval(I).sign());
        }
    });
    return test::finish("lean interval elementary");
}
