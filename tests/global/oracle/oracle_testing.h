// oracle_testing - what a test against the CORALean oracle shares
//
// Syntax:     if (!test::oracle::available()) return 0;
//             test::oracle::run("zonotope.supportFunc", [](test::oracle::Random &rnd) {
//                 const Zonotope Z = rnd.zonotope(3, 5);   const Tensor d = rnd.direction(3);
//                 test::oracle::check(Z.support_function(d), lean::Zonotope(Z).support_function(d));
//             });
// Modes:      run calls the body twice with the same random operands, "exact" (CORA.cpp on the lean
//             backend against the oracle at dtype nearest: bit for bit) and "sound" (CORA.cpp on
//             its default backend against the sound oracle at binary64: inside its enclosure);
//             check does the check of the mode, check_exact and check_enclosed one of them
// Results:    CORA.cpp's result and the oracle's, as double, Tensor, Interval, Zonotope, bool, the
//             indices of filter_out, a pair of halves, or Vertices; an oracle Certified answer
//             is a proof only when `proved`
// Skip:       available() is false, with a note, unless CORACPP_ORACLE and CORACPP_ORACLE_UNSOUND are
//             set; a test then returns at once
// See also:   testing.h, cora/global/oracle/request.h

#pragma once

#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/zonotope.h"
#include "testing.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>

namespace test::oracle {

namespace lean = cora::lean;

enum class Mode { exact, sound };

/// The mode of the running `run` (exact outside one), and the name of its op, for the messages.
inline Mode &mode() {
    static Mode m = Mode::exact;
    return m;
}
inline std::string &op_name() {
    static std::string name;
    return name;
}

/// Whether both oracles are configured; says so on stderr when not.
inline bool available() {
    const bool ok = std::getenv("CORACPP_ORACLE") && std::getenv("CORACPP_ORACLE_UNSOUND");
    if (!ok) std::cerr << "skipped: set CORACPP_ORACLE and CORACPP_ORACLE_UNSOUND\n";
    return ok;
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

    /// A box of n dimensions: a center and a radius in [0, 1].
    cora::Interval box(int64_t n) {
        const cora::Tensor c = vector(n);
        std::vector<double> r(n);
        for (double &x : r) x = uniform(0, 1);
        const cora::Tensor rad = cora::Tensor::from_data(r, {n, 1});
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

// Exact checks ------------------------------------------------------------------------------

/// The same bits: shape and every double, so -0 differs from 0; the first difference is printed.
template <class A, class B>
bool same_bits(const std::string &what, const A &cora_result, const B &lean_result) {
    const std::vector<double> x = detail::values(cora_result), y = detail::values(lean_result);
    if (detail::shape(cora_result) != detail::shape(lean_result)) {
        std::cerr << "  " << what << ": shape " << detail::describe(detail::shape(cora_result))
                  << " in CORA.cpp, " << detail::describe(detail::shape(lean_result))
                  << " in the oracle\n";
        return false;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::memcmp(&x[i], &y[i], sizeof(double)) != 0) {
            std::cerr << "  " << what << " [" << i << "]: " << detail::bits(x[i])
                      << " in CORA.cpp, " << detail::bits(y[i]) << " in the oracle\n";
            return false;
        }
    return true;
}

inline bool same_bits(const std::string &what, double a, double b) {
    return same_bits(what, cora::Tensor::from_data({a}, {1, 1}), cora::Tensor::from_data({b}, {1, 1}));
}

inline bool same_bits(const std::string &what, const cora::Interval &a, const lean::Interval &b) {
    return same_bits(what + ".inf", a.inf(), b.inf()) & same_bits(what + ".sup", a.sup(), b.sup());
}

/// The nominal part of the oracle's set; its error box is not compared (at nearest it is empty).
inline bool same_bits(const std::string &what, const cora::Zonotope &a, const lean::Zonotope &b) {
    return same_bits(what + ".c", a.center(), b.center()) &
           same_bits(what + ".G", a.generators(), b.generators());
}

/// The two halves of a split.
template <class A, class B>
bool same_bits(const std::string &what, const std::pair<A, A> &a, const std::pair<B, B> &b) {
    return same_bits(what + ".1", a.first, b.first) & same_bits(what + ".2", a.second, b.second);
}

template <class A, class B>
bool same_bits(const std::string &what, const std::optional<A> &a, const std::optional<B> &b) {
    if (a.has_value() != b.has_value()) {
        std::cerr << "  " << what << ": " << (a ? "a result" : "none") << " in CORA.cpp, "
                  << (b ? "a result" : "none") << " in the oracle\n";
        return false;
    }
    return !a || same_bits(what, *a, *b);
}

inline bool same_bits(const std::string &what, bool a, bool b) {
    if (a != b) std::cerr << "  " << what << ": " << a << " in CORA.cpp, " << b << " in the oracle\n";
    return a == b;
}

inline bool same_bits(const std::string &what, bool a, const lean::Certified &b) {
    return same_bits(what, a, b.value);
}

inline bool same_bits(const std::string &what, const std::vector<int64_t> &a,
                      const std::vector<int64_t> &b) {
    if (a != b) std::cerr << "  " << what << ": the indices differ\n";
    return a == b;
}

/// A 1 x 1 tensor (CORA.cpp's scalar results) against the oracle's number.
inline bool same_bits(const std::string &what, const cora::Tensor &a, double b) {
    return same_bits(what, a.data()[0], b);
}

// Enclosure checks --------------------------------------------------------------------------

/// A scalar of the oracle is an upper bound, which CORA.cpp's number may not exceed.
inline bool enclosed(const std::string &what, double a, double bound) {
    if (a > bound) std::cerr << "  " << what << ": " << detail::bits(a) << " exceeds the bound "
                             << detail::bits(bound) << "\n";
    return a <= bound;
}

inline bool enclosed(const std::string &what, const cora::Interval &a, const lean::Interval &b) {
    const std::vector<double> lo = a.inf().data(), hi = a.sup().data();
    const std::vector<double> blo = b.inf().gather().data(), bhi = b.sup().gather().data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (!(blo[i] <= lo[i] && hi[i] <= bhi[i])) {
            std::cerr << "  " << what << " [" << i << "]: [" << detail::bits(lo[i]) << ", "
                      << detail::bits(hi[i]) << "] is not inside [" << detail::bits(blo[i])
                      << ", " << detail::bits(bhi[i]) << "]\n";
            return false;
        }
    return true;
}

/// Inside nominal + error box: the oracle proves the containment, or else (when it cannot) the
/// interval hull of CORA.cpp's set at least lies in the hull of the enclosure.
inline bool enclosed(const std::string &what, const cora::Zonotope &a, const lean::Zonotope &b) {
    const lean::Zonotope mine(a), hull = b.enclosure();
    if (hull.contains(mine).proved) return true;
    return enclosed(what + " (hull)", a.interval(), hull.interval());
}

/// A certified answer must not be contradicted: if the oracle proves it, CORA.cpp says the same.
inline bool enclosed(const std::string &what, bool a, const lean::Certified &b) {
    if (b.proved && a != b.value)
        std::cerr << "  " << what << ": CORA.cpp says " << a << ", the oracle proves " << b.value
                  << "\n";
    return !b.proved || a == b.value;
}

/// An exact answer: a true of the oracle must not be contradicted.
inline bool enclosed(const std::string &what, bool a, bool b) {
    return enclosed(what, a, lean::Certified{b, b});
}

template <class A, class B>
bool enclosed(const std::string &what, const std::pair<A, A> &a, const std::pair<B, B> &b) {
    return enclosed(what + ".1", a.first, b.first) & enclosed(what + ".2", a.second, b.second);
}

template <class A, class B>
bool enclosed(const std::string &what, const std::optional<A> &a, const std::optional<B> &b) {
    return !a || !b || enclosed(what, *a, *b);
}

/// A point-valued result has no enclosure: nothing to check against the sound oracle.
inline bool enclosed(const std::string &, const cora::Tensor &, const lean::Tensor &) { return true; }
inline bool enclosed(const std::string &, const std::vector<int64_t> &, const std::vector<int64_t> &) {
    return true;
}

/// A 1 x 1 tensor against the oracle's bound.
inline bool enclosed(const std::string &what, const cora::Tensor &a, double bound) {
    return enclosed(what, a.data()[0], bound);
}

// The checks a test calls --------------------------------------------------------------------

/// CORA.cpp's result equals the oracle's bit for bit (run it on the lean backend, oracle at
/// dtype nearest).
template <class A, class B>
void check_exact(const std::string &op, const A &cora_result, const B &lean_result) {
    test::check(same_bits(op, cora_result, lean_result), op + ": exactly the oracle's result");
}

/// CORA.cpp's result lies in the sound oracle's.
template <class A, class B>
void check_enclosed(const std::string &op, const A &cora_result, const B &sound_result) {
    test::check(enclosed(op, cora_result, sound_result), op + ": inside the oracle's enclosure");
}

/// The check of the running mode, for the op of `run`.
template <class A, class B>
void check(const A &cora_result, const B &lean_result) {
    if (mode() == Mode::exact) check_exact(op_name(), cora_result, lean_result);
    else check_enclosed(op_name(), cora_result, lean_result);
}

/// Calls body(Random&) in both modes with the same seed: exact (lean backend, dtype nearest) and
/// sound (default backend, dtype binary64); what the body builds follows the mode.
template <class F>
void run(const std::string &op, F body, uint64_t seed = 7) {
    op_name() = op;
    for (const Mode m : {Mode::exact, Mode::sound}) {
        mode() = m;
        const cora::BackendScope backend(m == Mode::exact ? "lean" : cora::backend().name());
        const lean::DTypeScope dtype(m == Mode::exact ? "nearest" : "binary64");
        Random rnd(seed);
        body(rnd);
    }
}

} // namespace test::oracle
