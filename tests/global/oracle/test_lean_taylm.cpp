// test_lean_taylm - Taylm against the CORALean oracle (its `taylm.*` ops)
//
// ORACLE: contSet/@taylm/plus.m
// ORACLE: contSet/@taylm/minus.m
// ORACLE: contSet/@taylm/uminus.m
// ORACLE: contSet/@taylm/uplus.m
// ORACLE: contSet/@taylm/times.m
// ORACLE: contSet/@taylm/mtimes.m
// ORACLE: contSet/@taylm/power.m
// ORACLE: contSet/@taylm/mpower.m
// ORACLE: contSet/@taylm/interval.m
//
// The models of random boxes and zonotopes, and products of them truncated at their order, so
// that the remainders are not zero; every term (its monomial in the map's order, its coefficient)
// and the remainder is compared, and the oracle's error box is zero at nearest. CORA.cpp's
// mtimes and mpower of a scalar model are its times and power. The oracle's wire form has no -0,
// so the cases are chosen without one; the test checks that.
//
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "cora/global/oracle/taylm.h"
#include "oracle_testing.h"

#include <cmath>
#include <string>
#include <vector>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a value is -0, which the oracle's wire form cannot carry.
bool aux_negative_zero(double x) { return x == 0.0 && std::signbit(x); }

/// CORA.cpp's model against the oracle's: the terms in order, the remainder, no error box.
void aux_check(const std::string &what, const cora::Taylm &mine, const lean::Taylm &theirs) {
    const auto &terms = theirs.terms();
    test::check(mine.coefficients().size() == terms.size(), what + ": the number of terms");
    std::size_t i = 0;
    for (const auto &[m, c] : mine.coefficients()) {
        if (i >= terms.size()) break;
        test::check(m == terms[i].first, what + ": the monomials in the map's order");
        test::check(!aux_negative_zero(c), what + ": a case without -0");
        oracle::check(what + " coefficient " + std::to_string(i), c, terms[i].second);
        ++i;
    }
    const auto r = mine.remainder();
    test::check(!aux_negative_zero(r[0]) && !aux_negative_zero(r[1]), what + ": no -0");
    oracle::check(what + " remainder inf", r[0], theirs.remainder()[0]);
    oracle::check(what + " remainder sup", r[1], theirs.remainder()[1]);
    // the roundings of the sound operations cost nothing at nearest
    test::check(theirs.error()[0] == 0.0 && theirs.error()[1] == 0.0, what + ": no error box");
}

/// The operands: the coordinates of a box and of a zonotope, and products of them, which
/// truncate at the order and leave a remainder.
std::vector<cora::Taylm> aux_models(oracle::Random &rnd, int max_order) {
    std::vector<cora::Taylm> out;
    const std::vector<cora::Taylm> x = cora::Taylm::from_interval(rnd.box(3), max_order);
    out.insert(out.end(), x.begin(), x.end());
    out.push_back(x[0].times(x[1]).plus(x[2]));
    out.push_back(x[0].plus(x[1]).power(max_order + 1));
    out.push_back(x[2].times(x[2]).times(x[0].minus(x[1])).power(2));
    const std::vector<cora::Taylm> z = cora::Taylm::from_zonotope(rnd.zonotope(2, 3), max_order);
    out.push_back(z[0].times(z[1]).power(2));
    return out;
}

/// The operands with a nonzero remainder at both ends: their negation has no -0.
bool aux_signed(const cora::Taylm &T) { return T.remainder()[0] != 0 && T.remainder()[1] != 0; }

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the sums: equal monomials merged in the map, the remainders added
    oracle::run("taylm.plus", [](oracle::Random &rnd) {
        const std::vector<cora::Taylm> T = aux_models(rnd, 3);
        for (std::size_t a = 0; a < T.size(); ++a)
            for (std::size_t b = a % 2; b < T.size(); b += 2)
                aux_check("plus", T[a].plus(T[b]), lean::Taylm(T[a]).plus(lean::Taylm(T[b])));
        const double s = rnd.scalar();
        for (const cora::Taylm &t : T) aux_check("plus s", t.plus(s), lean::Taylm(t).plus(s));
    });

    // the differences: the second negated, then the sum
    oracle::run("taylm.minus", [](oracle::Random &rnd) {
        const std::vector<cora::Taylm> T = aux_models(rnd, 3);
        for (std::size_t a = 0; a < T.size(); ++a)
            for (std::size_t b = 0; b < T.size(); ++b)
                if (a != b && aux_signed(T[b]))
                    aux_check("minus", T[a].minus(T[b]),
                              lean::Taylm(T[a]).minus(lean::Taylm(T[b])));
        const double s = rnd.scalar();
        for (const cora::Taylm &t : T) aux_check("minus s", t.minus(s), lean::Taylm(t).minus(s));
    });

    oracle::run("taylm.uminus", [](oracle::Random &rnd) {
        for (const cora::Taylm &t : aux_models(rnd, 3))
            if (aux_signed(t)) aux_check("uminus", t.uminus(), lean::Taylm(t).uminus());
    });

    oracle::run("taylm.uplus", [](oracle::Random &rnd) {
        for (const cora::Taylm &t : aux_models(rnd, 3))
            aux_check("uplus", t.uplus(), lean::Taylm(t).uplus());
    });

    // the products at the first model's order, which CORA.cpp truncates at, and with a number
    oracle::run("taylm.times", [](oracle::Random &rnd) {
        for (const int order : {2, 3, 6}) {
            const std::vector<cora::Taylm> T = aux_models(rnd, order);
            for (std::size_t a = 0; a < T.size(); ++a)
                for (std::size_t b = 0; b < T.size(); ++b)
                    aux_check("times", T[a].times(T[b]),
                              lean::Taylm(T[a]).times(lean::Taylm(T[b]), T[a].max_order()));
            for (const cora::Taylm &t : T) {
                const double s = rnd.scalar();
                if (s > 0 || aux_signed(t))
                    aux_check("times s", t.times(s), lean::Taylm(t).times(s));
            }
        }
    });

    // the powers by binary exponentiation, the zeroth one included
    oracle::run("taylm.power", [](oracle::Random &rnd) {
        for (const int order : {2, 6}) {
            const std::vector<cora::Taylm> T = aux_models(rnd, order);
            for (const cora::Taylm &t : T)
                for (const int k : {0, 1, 2, 3, 5, 8})
                    aux_check("power " + std::to_string(k), t.power(k), lean::Taylm(t).power(k));
        }
    });

    // the range by the monomial bounds, CORA's method "int"
    oracle::run("taylm.interval", [](oracle::Random &rnd) {
        for (const cora::Taylm &t : aux_models(rnd, 3)) {
            const auto mine = t.bounds();
            const auto theirs = lean::Taylm(t).bounds();
            oracle::check("interval inf", mine[0], theirs[0]);
            oracle::check("interval sup", mine[1], theirs[1]);
        }
    });

    return test::finish("lean taylm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
