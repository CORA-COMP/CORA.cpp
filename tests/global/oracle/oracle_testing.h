// oracle_testing - what a test against the CORALean oracle shares
//
// Syntax:     if (!test::oracle::available()) return 0;
//             test::oracle::run("zonotope.supportFunc", [](test::oracle::Random &rnd) { ... });
//             test::oracle::check(Z.support_function(d), lean::Zonotope(Z).support_function(d));
// Passes:     run calls the body once per pass with the same random operands: "lean" (CORA.cpp on
//             the lean backend against the oracle at dtype nearest: bit for bit) and one per other
//             backend (the same body there against the same oracle answer, within tol: the backend
//             sanity check); $CORACPP_ORACLE_PASS (lean, eigen, torch) runs one pass only
// Results:    double, Tensor, the sets, bool, indices, pairs; bools and counts exact in every pass
// Tolerance:  the backend pass compares by within_tol with the repository tolerance; a body that
//             needs another passes run(op, body, Tolerance{tol, "reason"})
// Skip:       available() is false, with a note, unless CORACPP_ORACLE is set; a test then returns
//             at once. run skips an op the oracle does not serve (`lean::serves`), naming it once
//             on stderr; serves(op) asks for another op, and nearest_is_strict() for a check that
//             needs the machine's strict order at nearest
// See also:   testing.h, cora/global/oracle/request.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   07-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/capsule.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/fullspace.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/interval_matrix.h"
#include "cora/global/oracle/mat_zonotope.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/polytope.h"
#include "cora/global/oracle/zono_bundle.h"
#include "cora/global/oracle/zonotope.h"
#include "cora/global/within_tol.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "testing.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::oracle {

namespace lean = cora::lean;

/// A tolerance of the backend pass other than the repository's, with the reason it is needed.
struct Tolerance {
    Tolerance(double value, const char *why) : tol(value), reason(why) {}
    double tol;
    const char *reason;
};

/// The pass of the running `run`: the backend's name, "lean" outside one; its op and tolerance.
inline std::string &pass() {
    static std::string name = "lean";
    return name;
}
inline std::string &op_name() {
    static std::string name;
    return name;
}
inline cora::Tol &tolerance() {
    static cora::Tol tol;
    return tol;
}

/// Whether the oracle is configured; says so on stderr when not.
inline bool available() {
    const bool ok = std::getenv("CORACPP_ORACLE") != nullptr;
    if (!ok) std::cerr << "skipped: set CORACPP_ORACLE\n";
    return ok;
}

/// Says on stderr that a check is skipped, once per reason: an older oracle is still a valid one,
/// its tests skip what it lacks.
inline void skipped(const std::string &reason) {
    static std::set<std::string> said;
    if (said.insert(reason).second) std::cerr << "skipped: " << reason << "\n";
}

/// Whether the oracle serves `op`; names an op it does not serve once.
inline bool serves(const std::string &op) {
    const bool ok = lean::serves(op);
    if (!ok) skipped("the oracle does not serve " + op);
    return ok;
}

/// Whether the oracle at nearest compares strictly by the machine's order, so that a box is not
/// below itself; the checks of degenerate cases are skipped without it.
inline bool nearest_is_strict() {
    static const bool strict = [] {
        const lean::DTypeScope dtype("nearest");
        const cora::Tensor one = cora::Tensor::from_data({1.0}, {1, 1}), two = one + one;
        const lean::Interval box(cora::Interval(one, two));
        return !box.lt(box);
    }();
    if (!strict) skipped("the oracle's strict order at nearest has a margin (older oracle)");
    return strict;
}

/// Random operands from a fixed seed: normal numbers, which are exact doubles, on the current
/// backend (so a lean-backend run computes with lean tensors).
class Random {
  public:
    explicit Random(uint64_t seed) : gen_(seed) {}

    double scalar() { return normal_(gen_); }

    /// A uniform number in [low, high).
    double uniform(double low, double high) {
        return low + (high - low) * std::uniform_real_distribution<double>()(gen_);
    }

    cora::Tensor matrix(int64_t rows, int64_t cols) {
        std::vector<double> data(rows * cols);
        for (double &x : data) x = scalar();
        return cora::Tensor::from_data(data, {rows, cols});
    }
    cora::Tensor vector(int64_t n) { return matrix(n, 1); }

    /// A direction (n, 1) of unit length up to rounding.
    cora::Tensor direction(int64_t n) { return vector(n); }

    cora::Zonotope zonotope(int64_t n, int64_t m) { return {vector(n), matrix(n, m)}; }

    /// A constrained zonotope of n dimensions and m generators with p constraints [I B] x = b; b is
    /// the image of a factor vector inside the box, so the set is not empty.
    cora::ConZonotope con_zonotope(int64_t n, int64_t m, int64_t p) {
        std::vector<double> A(p * m, 0.0), xi(m), b(p, 0.0);
        for (double &x : xi) x = uniform(-0.5, 0.5);
        for (int64_t i = 0; i < p; ++i) {
            A[i * m + i] = 1.0;
            b[i] = xi[i];
            for (int64_t j = p; j < m; ++j) {
                A[i * m + j] = 0.3 * scalar();
                b[i] += A[i * m + j] * xi[j];
            }
        }
        return {vector(n), matrix(n, m), cora::Tensor::from_data(A, {p, m}),
                cora::Tensor::from_data(b, {p, 1})};
    }

    /// A matrix zonotope of rows x cols matrices with h generators.
    cora::MatZonotope mat_zonotope(int64_t rows, int64_t cols, int64_t h) {
        std::vector<double> data(h * rows * cols);
        for (double &x : data) x = scalar();
        return {matrix(rows, cols), cora::Tensor::from_data(data, {h, rows, cols})};
    }

    /// A bundle of k zonotopes of n dimensions and m generators each, all containing one common
    /// point (the intersection is not empty).
    cora::ZonoBundle zono_bundle(int64_t n, int64_t k, int64_t m) {
        const cora::Tensor x0 = vector(n);
        std::vector<cora::Zonotope> members;
        for (int64_t i = 0; i < k; ++i) {
            const cora::Tensor G = matrix(n, m);
            std::vector<double> b(m);
            for (double &x : b) x = uniform(-1, 1);
            members.emplace_back(x0 - G.matmul(cora::Tensor::from_data(b, {m, 1})), G);
        }
        return cora::ZonoBundle(std::move(members));
    }

    /// A capsule of n dimensions: a center, a generator and a radius in [0.1, 1].
    cora::Capsule capsule(int64_t n) {
        const cora::Tensor c = vector(n), g = vector(n);
        return {c, g, uniform(0.1, 1.0)};
    }

    /// A box of n dimensions: a center and a radius in [0, 1].
    cora::Interval box(int64_t n) {
        const cora::Tensor c = vector(n);
        std::vector<double> r(n);
        for (double &x : r) x = uniform(0, 1);
        const cora::Tensor rad = cora::Tensor::from_data(r, {n, 1});
        return {c - rad, c + rad};
    }

    /// An interval matrix of rows x cols: a center and a radius in [0, 1] per entry.
    cora::Interval box_matrix(int64_t rows, int64_t cols) {
        const cora::Tensor c = matrix(rows, cols);
        std::vector<double> r(rows * cols);
        for (double &x : r) x = uniform(0, 1);
        const cora::Tensor rad = cora::Tensor::from_data(r, {rows, cols});
        return {c - rad, c + rad};
    }

  private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> normal_;
};

// Reading results ---------------------------------------------------------------------------

namespace detail {

/// The bits of a double as text, so that a diff shows -0 and 1 ulp.
inline std::string bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    std::ostringstream out;
    out.precision(17);
    out << x << " (0x" << std::hex << u << ")";
    return out.str();
}

inline std::vector<double> values(const cora::Tensor &T) { return T.data(); }
inline std::vector<double> values(const lean::Tensor &T) { return T.gather().data(); }

inline std::vector<int64_t> shape(const cora::Tensor &T) { return T.shape(); }
inline std::vector<int64_t> shape(const lean::Tensor &T) { return {T.rows(), T.cols()}; }

inline std::string describe(const std::vector<int64_t> &s) {
    return std::to_string(s[0]) + "x" + std::to_string(s[1]);
}

} // namespace detail

// Comparisons ------------------------------------------------------------------------------

/// Whether two doubles agree in the running pass: the same bits on lean (so -0 differs from 0),
/// within_tol of the pass's tolerance on another backend (two NaN agree).
inline bool agree_value(double x, double y) {
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison): the bits are the point (NaN, -0)
    if (pass() == "lean") return std::memcmp(&x, &y, sizeof(double)) == 0;
    return (std::isnan(x) && std::isnan(y)) || cora::within_tol(x, y, tolerance());
}

/// The shape and every double agree; the first difference is printed.
template <class A, class B>
bool agree(const std::string &what, const A &cora_result, const B &lean_result) {
    const std::vector<double> x = detail::values(cora_result), y = detail::values(lean_result);
    if (detail::shape(cora_result) != detail::shape(lean_result)) {
        std::cerr << "  " << what << ": shape " << detail::describe(detail::shape(cora_result))
                  << " in CORA.cpp, " << detail::describe(detail::shape(lean_result))
                  << " in the oracle\n";
        return false;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!agree_value(x[i], y[i])) {
            std::cerr << "  " << what << " [" << i << "]: " << detail::bits(x[i])
                      << " in CORA.cpp on " << pass() << ", " << detail::bits(y[i])
                      << " in the oracle\n";
            return false;
        }
    return true;
}

/// Two numbers, as 1 x 1 tensors.
inline bool agree(const std::string &what, double a, double b) {
    return agree(what, cora::Tensor::from_data({a}, {1, 1}), cora::Tensor::from_data({b}, {1, 1}));
}

inline bool agree(const std::string &what, const cora::Interval &a, const lean::Interval &b) {
    const bool inf = agree(what + ".inf", a.inf(), b.inf()); // both are reported
    return agree(what + ".sup", a.sup(), b.sup()) && inf;
}

/// A whole space is its dimension: the same one.
inline bool agree(const std::string &what, const cora::Fullspace &a, const lean::Fullspace &b) {
    if (a.dim() != b.dim())
        std::cerr << "  " << what << ": R^" << a.dim() << " in CORA.cpp, R^" << b.dim()
                  << " in the oracle\n";
    return a.dim() == b.dim();
}

/// An interval matrix against the oracle's interval matrix: its bounds.
inline bool agree(const std::string &what, const cora::IntervalMatrix &a, const lean::Interval &b) {
    return agree(what, a.interval(), b);
}

/// The nominal part of the oracle's set; its error box is not compared (at nearest it is empty).
inline bool agree(const std::string &what, const cora::Zonotope &a, const lean::Zonotope &b) {
    const bool center = agree(what + ".c", a.center(), b.center()); // both are reported
    return agree(what + ".G", a.generators(), b.generators()) && center;
}

/// The nominal part of the oracle's set (c, G and the equalities); its error boxes are not
/// compared.
inline bool agree(const std::string &what, const cora::ConZonotope &a, const lean::ConZonotope &b) {
    const cora::ConZonotope other = b.nominal();
    bool same = agree(what + ".c", a.c(), other.c()); // all are reported
    same = agree(what + ".G", a.generators(), other.generators()) && same;
    same = agree(what + ".A", a.A(), other.A()) && same;
    return agree(what + ".b", a.b(), other.b()) && same;
}

/// The nominal part of the oracle's matrix zonotope: the center and the generators in their order.
inline bool agree(const std::string &what, const cora::MatZonotope &a, const lean::MatZonotope &b) {
    const cora::MatZonotope other = b.gather().first;
    // both compared, so that both mismatches are reported
    const bool center = agree(what + ".C", a.center(), other.center());
    const bool generators = agree(what + ".G", a.generators(), other.generators());
    return center && generators;
}

/// The members of a bundle, one by one, and their number.
inline bool agree(const std::string &what, const cora::ZonoBundle &a, const lean::ZonoBundle &b) {
    if (a.parallel_sets() != int64_t(b.zonotopes().size())) {
        std::cerr << "  " << what << ": " << a.parallel_sets() << " members in CORA.cpp, "
                  << b.zonotopes().size() << " in the oracle\n";
        return false;
    }
    bool ok = true;
    for (std::size_t i = 0; i < b.zonotopes().size(); ++i)
        ok &= agree(what + "[" + std::to_string(i) + "]", a.zonotopes()[i], b.zonotopes()[i]);
    return ok;
}


/// The nominal part of the oracle's polytope (its error boxes are not compared): the vertices, or
/// the rows and the equalities.
inline bool agree(const std::string &what, const cora::Polytope &a, const lean::Polytope &b) {
    const cora::Polytope other = b.nominal();
    if (a.is_h_rep() != other.is_h_rep()) {
        std::cerr << "  " << what << ": the form differs\n";
        return false;
    }
    if (!a.is_h_rep()) return agree(what + ".V", a.V(), other.V());
    // all four compared, so that every mismatch is reported
    const bool rows = agree(what + ".A", a.A(), other.A());
    const bool offsets = agree(what + ".b", a.b(), other.b());
    const bool equalities = agree(what + ".Ae", a.Ae(), other.Ae());
    const bool equality_offsets = agree(what + ".be", a.be(), other.be());
    return rows && offsets && equalities && equality_offsets;
}


/// The nominal part of the oracle's capsule: center, generator and radius; the error box is not
/// compared (at nearest it is empty).
inline bool agree(const std::string &what, const cora::Capsule &a, const lean::Capsule &b) {
    const cora::Capsule other = b.nominal();
    // all three compared, so that every mismatch is reported
    const bool center = agree(what + ".c", a.c(), other.c());
    const bool generator = agree(what + ".g", a.generator(), other.generator());
    const bool radius = agree(what + ".r", a.r(), other.r());
    return center && generator && radius;
}

/// The two halves of a split.
template <class A, class B>
bool agree(const std::string &what, const std::pair<A, A> &a, const std::pair<B, B> &b) {
    const bool first = agree(what + ".1", a.first, b.first); // both are reported
    return agree(what + ".2", a.second, b.second) && first;
}

template <class A, class B>
bool agree(const std::string &what, const std::optional<A> &a, const std::optional<B> &b) {
    if (a.has_value() != b.has_value()) {
        std::cerr << "  " << what << ": " << (a ? "a result" : "none") << " in CORA.cpp, "
                  << (b ? "a result" : "none") << " in the oracle\n";
        return false;
    }
    return !a || agree(what, *a, *b);
}

inline bool agree(const std::string &what, bool a, bool b) {
    if (a != b)
        std::cerr << "  " << what << ": " << a << " in CORA.cpp, " << b << " in the oracle\n";
    return a == b;
}

/// A dimension or a count, exact.
inline bool agree(const std::string &what, int64_t a, int64_t b) {
    if (a != b)
        std::cerr << "  " << what << ": " << a << " in CORA.cpp, " << b << " in the oracle\n";
    return a == b;
}

inline bool agree(const std::string &what, bool a, const lean::Certified &b) {
    return agree(what, a, b.value);
}

inline bool agree(const std::string &what, const std::vector<int64_t> &a,
                  const std::vector<int64_t> &b) {
    if (a != b) std::cerr << "  " << what << ": the indices differ\n";
    return a == b;
}

/// A point result against a point interval of the oracle: both of its bounds.
inline bool agree(const std::string &what, const cora::Tensor &a, const lean::Interval &b) {
    const bool inf = agree(what + ".inf", a, b.inf()); // both are reported
    return agree(what + ".sup", a, b.sup()) && inf;
}

/// A 1 x 1 tensor (CORA.cpp's scalar results) against the oracle's number.
inline bool agree(const std::string &what, const cora::Tensor &a, double b) {
    return agree(what, a.data()[0], b);
}

/// The interval hull of CORA.cpp's result against the oracle's, to a relative `rel`, for an
/// operation that sums in another order than CORALean's (a few ulp of the sums, not bits), in
/// every pass: the box of CORA.cpp's zonotope and of the oracle's, bound by bound.
inline void check_hull(const cora::Zonotope &mine, const lean::Zonotope &theirs,
                       double rel = 1e-10) {
    const cora::Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i)
        ok = std::abs(a[i] - b[i]) <= rel * (1 + std::abs(b[i])) &&
             std::abs(c[i] - d[i]) <= rel * (1 + std::abs(d[i]));
    test::check(ok, op_name() + ": the box of the oracle's result");
}

/// CORA.cpp's box inside the oracle's, for an op whose oracle answer is a wider enclosure (the
/// tests `*_enclosure` and `*_bound`, which are not bit-exact).
inline void check_inside(const cora::Interval &mine, const lean::Interval &theirs) {
    const std::vector<double> lo = mine.inf().data(), hi = mine.sup().data();
    const std::vector<double> blo = theirs.inf().gather().data(),
                              bhi = theirs.sup().gather().data();
    bool ok = lo.size() == blo.size();
    for (std::size_t i = 0; ok && i < lo.size(); ++i) {
        // another backend may round a bound across the oracle's, within the pass's tolerance
        const auto below = [](double x, double y) {
            return x <= y || (pass() != "lean" && cora::within_tol(x, y, tolerance()));
        };
        ok = below(blo[i], lo[i]) && below(hi[i], bhi[i]);
        if (!ok)
            std::cerr << "  " << op_name() << " [" << i << "]: [" << detail::bits(lo[i]) << ", "
                      << detail::bits(hi[i]) << "] is not inside [" << detail::bits(blo[i]) << ", "
                      << detail::bits(bhi[i]) << "]\n";
    }
    test::check(ok, op_name() + ": inside the oracle's box");
}

// The checks a test calls --------------------------------------------------------------------

/// CORA.cpp's result agrees with the oracle's in the running pass, under the name `op`: bit for
/// bit on lean, within the tolerance on another backend.
template <class A, class B>
void check(const std::string &op, const A &cora_result, const B &lean_result) {
    test::check(agree(op, cora_result, lean_result),
                op + (pass() == "lean" ? ": exactly the oracle's result"
                                       : ": the oracle's result within tol on " + pass()));
}

/// The check of the running pass, for the op of `run`.
template <class A, class B>
void check(const A &cora_result, const B &lean_result) {
    check(op_name(), cora_result, lean_result);
}

/// The passes to run: $CORACPP_ORACLE_PASS, else lean and the default backend (if not lean).
inline std::vector<std::string> passes() {
    const char *only = std::getenv("CORACPP_ORACLE_PASS");
    if (only && *only) return {only};
    const std::string other = cora::backend().name();
    if (other == "lean") return {"lean"};
    return {"lean", other};
}

/// Calls body(Random&) once per pass with the same seed, the oracle at dtype nearest: on the
/// lean backend, and on each other backend of `passes()` within `tol` of the oracle's answer.
template <class F>
void run(const std::string &op, F body, cora::Tol tol = std::nullopt) {
    // a name with a dot is an op of the oracle (`zonotope.enlarge`); one without is a generic test
    if (op.find('.') != std::string::npos && !serves(op)) return;
    op_name() = op;
    tolerance() = tol;
    for (const std::string &name : passes()) {
        pass() = name;
        const cora::BackendScope backend(name);
        const lean::DTypeScope dtype("nearest");
        Random rnd(7);
        body(rnd);
    }
    pass() = "lean";
}

/// run with a tolerance of its own for the backend pass; `tol.reason` says why.
template <class F>
void run(const std::string &op, F body, const Tolerance &tol) {
    run(op, body, cora::Tol(tol.tol));
}

} // namespace test::oracle

// ---------------------------------------  END OF CODE  ---------------------------------------- //
