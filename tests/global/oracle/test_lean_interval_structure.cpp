// test_lean_interval_structure - the structural interval operations against the CORALean oracle
//
// uminus, uplus, transpose, supremum, sum, tril, triu, vertcat, volume, split and zonotope: each
// result equals the oracle's bit for bit at nearest.
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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// An interval matrix of rows x cols entries: a center and a radius in [0, 1].
Interval aux_matrix(test::oracle::Random &rnd, int64_t rows, int64_t cols) {
    const Tensor c = rnd.matrix(rows, cols);
    std::vector<double> r(rows * cols);
    for (double &x : r) x = rnd.uniform(0, 1);
    const Tensor rad = Tensor::from_data(r, {rows, cols});
    return {c - rad, c + rad};
}

} // namespace


// ===========================================  MAIN  =========================================== //

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
            // transposes
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
            test::oracle::check(
                Interval(down.inf().matrix_transpose(), down.sup().matrix_transpose()),
                lean::Interval(M).sum(1));
            test::oracle::check(M.sum(2), lean::Interval(M).sum(2));
            const Interval I = rnd.box(n);
            test::oracle::check(I.sum(), lean::Interval(I).sum(1));
        }
    });
    // the lower and upper triangles of square interval matrices
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
    // vertical concatenation
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
            // split at a coordinate
            for (const int j : {0, n / 2, n - 1})
                test::oracle::check(I.split(j), lean::Interval(I).split(j));
        }
    });
    // the box as a zonotope, up to n = 6
    test::oracle::run("interval.zonotope", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 6}) {
            const Interval I = rnd.box(n);
            const Zonotope Z = I.zonotope();
            const lean::Zonotope O = lean::Interval(I).zonotope();
            test::oracle::check(Z, O);
        }
    });
    return test::finish("lean interval structure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
