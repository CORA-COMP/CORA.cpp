// test_lean_interval_structure - uminus, uplus, transpose, supremum, sum, tril, triu, vertcat,
// volume, split and zonotope against the CORALean oracle
//
// Each result equals the oracle's bit for bit at nearest and lies in its sound enclosure; volume
// is compared with the oracle's upper bound. The sum and the zonotope are evaluated to nearest by
// CORA.cpp (the sum in the order of its backend), so they lie in the sound enclosure up to kSlack,
// relative: the enclosure holds the real value, and a rounded result may be a few ulps beyond it.
//
// ORACLE: contSet/@interval/uminus.m
// ORACLE: contSet/@interval/uplus.m
// ORACLE: contSet/@interval/transpose.m
// ORACLE: contSet/@interval/supremum.m
// ORACLE: contSet/@interval/sum.m
// ORACLE: contSet/@interval/tril.m
// ORACLE: contSet/@interval/triu.m
// ORACLE: contSet/@interval/vertcat.m
// ORACLE: contSet/@interval/volume_.m
// ORACLE: contSet/@interval/split.m
// ORACLE: contSet/@interval/zonotope.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cora;

namespace {

constexpr double kSlack = 1e-14;

/// Whether the bounds of a lie in those of b widened by kSlack, relative.
bool aux_nearly_inside(const Interval &a, const lean::Interval &b) {
    const auto [blo, bhi] = b.gather();
    const std::vector<double> lo = a.inf().data(), hi = a.sup().data(), l = blo.data(),
                              h = bhi.data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (lo[i] < l[i] - kSlack * std::max(1.0, std::abs(l[i])) ||
            hi[i] > h[i] + kSlack * std::max(1.0, std::abs(h[i]))) {
            std::cerr << "  " << test::oracle::op_name() << " [" << i << "]: [" << lo[i] << ", "
                      << hi[i] << "] is not inside [" << l[i] << ", " << h[i] << "] up to kSlack\n";
            return false;
        }
    return true;
}

/// The check of the mode: the same bits, or inside the oracle's enclosure up to kSlack.
void aux_check(const Interval &a, const lean::Interval &b) {
    const std::string &op = test::oracle::op_name();
    if (test::oracle::mode() == test::oracle::Mode::exact)
        test::oracle::check_exact(op, a, b);
    else
        test::check(aux_nearly_inside(a, b), op + ": inside the enclosure up to kSlack");
}

/// An interval matrix of rows x cols entries: a center and a radius in [0, 1].
Interval aux_matrix(test::oracle::Random &rnd, int64_t rows, int64_t cols) {
    const Tensor c = rnd.matrix(rows, cols);
    std::vector<double> r(rows * cols);
    for (double &x : r) x = rnd.uniform(0, 1);
    const Tensor rad = Tensor::from_data(r, {rows, cols});
    return {c - rad, c + rad};
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.uminus", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.uminus(), lean::Interval(I).uminus());
        }
    });
    test::oracle::run("interval.uplus", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.uplus(), lean::Interval(I).uplus());
        }
    });
    test::oracle::run("interval.transpose", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), M = aux_matrix(rnd, n, 3);
            test::oracle::check(I.transpose(), lean::Interval(I).transpose());
            test::oracle::check(M.transpose(), lean::Interval(M).transpose());
        }
    });
    test::oracle::run("interval.supremum", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.supremum(), lean::Interval(I).supremum());
        }
    });
    test::oracle::run("interval.sum", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // the oracle returns a column also for the sums down the rows, CORA.cpp a row
            const Interval M = aux_matrix(rnd, n, 4), down = M.sum(1);
            aux_check(Interval(down.inf().matrix_transpose(), down.sup().matrix_transpose()),
                      lean::Interval(M).sum(1));
            aux_check(M.sum(2), lean::Interval(M).sum(2));
            const Interval I = rnd.box(n);
            aux_check(I.sum(), lean::Interval(I).sum(1));
        }
    });
    test::oracle::run("interval.tril", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 12}) {
            const Interval M = aux_matrix(rnd, n, n);
            for (const int K : {0, 1, 3}) test::oracle::check(M.tril(K), lean::Interval(M).tril(K));
        }
    });
    test::oracle::run("interval.triu", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 12}) {
            const Interval M = aux_matrix(rnd, n, n);
            for (const int K : {0, 1, 3}) test::oracle::check(M.triu(K), lean::Interval(M).triu(K));
        }
    });
    test::oracle::run("interval.vertcat", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n + 1), M = aux_matrix(rnd, n, 3);
            test::oracle::check(Interval::vertcat({I, J}),
                                lean::Interval(I).vertcat(lean::Interval(J)));
            test::oracle::check(Interval::vertcat({M, M}),
                                lean::Interval(M).vertcat(lean::Interval(M)));
        }
    });
    test::oracle::run("interval.volume", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.volume(), lean::Interval(I).volume());
        }
    });
    test::oracle::run("interval.split", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int j : {0, n / 2, n - 1})
                test::oracle::check(I.split(j), lean::Interval(I).split(j));
        }
    });
    // the sound oracle proves the containment of the zonotope, which takes minutes beyond n = 6
    test::oracle::run("interval.zonotope", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 6}) {
            const Interval I = rnd.box(n);
            const Zonotope Z = I.zonotope();
            const lean::Zonotope O = lean::Interval(I).zonotope();
            if (test::oracle::mode() == test::oracle::Mode::exact)
                test::oracle::check(Z, O);
            else
                test::check(aux_nearly_inside(Z.interval(), O.enclosure().interval()),
                            "interval.zonotope: inside the enclosure up to kSlack");
        }
    });
    return test::finish("lean interval structure");
}
