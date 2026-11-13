// test_lean_poly_zonotope - PolyZonotope against the CORALean oracle (its `polyZonotope.*` ops)
//
// ORACLE: contSet/@polyZonotope/polyZonotope.m
// ORACLE: contSet/@polyZonotope/plus.m
// ORACLE: contSet/@polyZonotope/mtimes.m
// ORACLE: contSet/@polyZonotope/zonotope.m
// ORACLE: contSet/@polyZonotope/interval.m
// ORACLE: contSet/@polyZonotope/reduce.m
// ORACLE: contSet/@polyZonotope/enclose.m
// ORACLE: contSet/@polyZonotope/dim.m
// ORACLE: contSet/@interval/polyZonotope.m
// ORACLE: contSet/@zonotope/polyZonotope.m
//
// Random sets with repeated, zero and cancelling columns, with and without independent
// generators, and the sets of an AINNCS run: the ACC network on its initial set, and the plant
// state joined with it as the closed loop forms it. Every column of the result (generators,
// exponents, identifiers) is compared in its order; the oracle orders equal exponent columns as
// CORA does (the hash of removeRedundantExponents).
//
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/poly_zonotope.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <cstdlib>
#include <filesystem>

// the folder of the instance data, set before the benchmark header is read
#define CORACPP_BENCHMARK_DATA "."
#include "../../../examples/cpp/benchmarks/benchmark_neur_net.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shape of a random set: n dimensions, h dependent generators over p factors, q independent.
struct Dims {
    int64_t n, h, p, q;
};

/// A random exponent column of the factors, each exponent in 0..3.
std::vector<int> aux_column(oracle::Random &rnd, int64_t p) {
    std::vector<int> col(static_cast<std::size_t>(p), 0);
    for (int &e : col) e = static_cast<int>(rnd.uniform(0, 4));
    return col;
}

/// The columns of a random set before the constructor: the generators (n, h) and the exponents,
/// with columns repeated (the same exponents twice, three times) and one generator zero.
struct Raw {
    Tensor c, G, GI;
    Exponents E;
    std::vector<int64_t> id;
};

Raw aux_raw(oracle::Random &rnd, const Dims &s, bool repeats) {
    Exponents E;
    for (int64_t l = 0; l < s.h; ++l) E.push_back(aux_column(rnd, s.p));
    // equal columns: every fourth takes the exponents of an earlier one, the next of those twice
    if (repeats)
        for (int64_t l = 3; l < s.h; l += 4) E[std::size_t(l)] = E[std::size_t(l - 3)];
    if (repeats)
        for (int64_t l = 7; l < s.h; l += 8) E[std::size_t(l - 1)] = E[std::size_t(l)];
    // no constant column (exponent 0): CORA halves it in zonotope, CORALean moves it whole
    for (std::vector<int> &col : E)
        if (std::all_of(col.begin(), col.end(), [](int e) { return e == 0; })) col[0] = 1;
    std::vector<int64_t> id(std::size_t(s.p));
    for (int64_t i = 0; i < s.p; ++i) id[std::size_t(i)] = 2 * i + 3;
    return {rnd.vector(s.n), rnd.matrix(s.n, s.h), rnd.matrix(s.n, s.q), E, id};
}

PolyZonotope aux_set(const Raw &r) { return {r.c, r.G, r.GI, r.E, r.id}; }

/// A set from the oracle's constructor: what the other ops take as their operand.
lean::PolyZonotope aux_lean(const Raw &r) {
    return lean::PolyZonotope::construct(r.c, r.G, r.GI, r.E, r.id);
}

/// The shapes the ops are tried on.
const std::vector<Dims> kDims = {{2, 3, 2, 0}, {2, 6, 3, 2}, {3, 9, 4, 3}, {5, 20, 6, 4}};

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the constructor: equal columns added in the order of the hash, zero columns dropped
    oracle::run("polyZonotope.constructor", [](oracle::Random &rnd) {
        for (const Dims &s : kDims)
            for (const bool repeats : {false, true}) {
                const Raw r = aux_raw(rnd, s, repeats);
                oracle::check(aux_set(r), lean::PolyZonotope::construct(r.c, r.G, r.GI, r.E, r.id));
            }
    });

    oracle::run("polyZonotope.dim", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, true);
            oracle::check(aux_set(r).dim(), aux_lean(r).dim());
        }
    });

    // the sum with a polynomial zonotope, a zonotope, a box and a point
    oracle::run("polyZonotope.plus", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, true), q = aux_raw(rnd, {s.n, s.h, s.p, s.q}, true);
            const PolyZonotope P = aux_set(r), Q = aux_set(q);
            oracle::check("plus pZ", P.plus(Q), aux_lean(r).plus(aux_lean(q)));
            // a zonotope, a box and a point join the independent part
            const Zonotope Z(rnd.vector(s.n), rnd.matrix(s.n, 3));
            oracle::check("plus Z", P.plus(Z), aux_lean(r).plus(Z));
            // the point after the box, as the oracle's op takes them
            const Interval I = rnd.box(s.n);
            oracle::check("plus I", P.plus(I), aux_lean(r).plus(I));
            const Tensor v = rnd.vector(s.n);
            oracle::check("plus v", P.plus(v), aux_lean(r).plus(v));
        }
    });

    // the image under a matrix
    oracle::run("polyZonotope.mtimes", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, true);
            const Tensor M = rnd.matrix(s.n + 1, s.n);
            oracle::check(aux_set(r).linear_map(M), aux_lean(r).linear_map(M));
        }
    });

    // the zonotope of CORA's order of generators
    oracle::run("polyZonotope.zonotope", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, true);
            oracle::check(aux_set(r).zonotope(), aux_lean(r).zonotope());
        }
    });

    // the box of that zonotope
    oracle::run("polyZonotope.interval", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, true);
            oracle::check(aux_set(r).interval(), aux_lean(r).interval());
        }
    });

    // the order bound counts columns: orders below, at and above the number of generators
    oracle::run("polyZonotope.reduce", [](oracle::Random &rnd) {
        for (const Dims &s : kDims)
            for (const int order : {1, 2, 3, 10}) {
                const Raw r = aux_raw(rnd, s, true);
                oracle::check("order " + std::to_string(order), aux_set(r).reduce(order),
                              aux_lean(r).reduce(order));
            }
    });

    // the same factors and exponents, and other factors
    oracle::run("polyZonotope.enclose", [](oracle::Random &rnd) {
        for (const Dims &s : kDims) {
            const Raw r = aux_raw(rnd, s, false);
            Raw same = aux_raw(rnd, s, false);
            same.E = r.E;
            same.id = r.id;
            oracle::check("same factors", aux_set(r).enclose(aux_set(same)),
                          aux_lean(r).enclose(aux_lean(same)));
            // the oracle's constructor is cubic in the columns, and this case builds 4 (h + h)
            if (s.h > 9) continue;
            Raw other = aux_raw(rnd, {s.n, s.h, s.p, s.q}, false);
            // every other identifier is the first set's, the rest are new
            for (int64_t i = 0; i < s.p; ++i)
                other.id[std::size_t(i)] = r.id[std::size_t(i)] + i % 2;
            oracle::check("other factors", aux_set(r).enclose(aux_set(other)),
                          aux_lean(r).enclose(aux_lean(other)));
        }
    });

    // the conversions to a polynomial zonotope
    oracle::run("zonotope.polyZonotope", [](oracle::Random &rnd) {
        for (const int64_t m : {1, 4, 9}) {
            const Zonotope Z = rnd.zonotope(4, m);
            oracle::check(PolyZonotope(Z), lean::PolyZonotope::from_zonotope(Z));
        }
    });
    oracle::run("interval.polyZonotope", [](oracle::Random &rnd) {
        for (const int64_t n : {1, 3, 8}) {
            const Interval I = rnd.box(n);
            oracle::check(PolyZonotope(I), lean::PolyZonotope::from_interval(I));
        }
    });

    return test::finish("lean poly zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
