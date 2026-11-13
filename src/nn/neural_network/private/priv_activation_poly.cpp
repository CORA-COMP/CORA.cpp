// priv_activation_poly - the polynomial that encloses an activation over a range, and its error
//
// See also: priv_activation_poly.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA: nn/layers/nonlinear/nnLeakyReLULayer.m
// CORA: nn/layers/nonlinear/nnSShapeLayer.m
// CORA authors: Tobias Ladner, Niklas Kochdumper, Sebastian Sigl
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "nn/neural_network/private/priv_activation_poly.h"

#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Poly = std::vector<double>; // highest power first, as MATLAB's polyval

/// The sample count of the error bound: at least this many, at most that many (CORA: 5e9, which
/// is 40 GB; the Lipschitz bound is recomputed from the count used, so a cap stays sound).
constexpr double kMinPoints = 1e4, kMaxPoints = 2e7;

// Polynomials -------------------------------------------------------------------------------------

Poly aux_derivative(const Poly &p) {
    Poly out;
    const std::size_t n = p.size();
    for (std::size_t i = 0; i + 1 < n; ++i) out.push_back(p[i] * double(n - 1 - i));
    return out.empty() ? Poly{0.0} : out;
}

/// The real roots of p in [a, b]: the roots of p' cut [a, b] into monotone pieces, and a piece
/// with a sign change is bisected.
std::vector<double> aux_roots(Poly p, double a, double b) {
    while (p.size() > 1 && p.front() == 0.0) p.erase(p.begin());
    if (p.size() <= 1) return {};
    if (p.size() == 2) {
        const double r = -p[1] / p[0];
        return a <= r && r <= b ? std::vector<double>{r} : std::vector<double>{};
    }
    // the pieces between the ends and the roots of the derivative
    std::vector<double> pieces{a};
    for (const double x : aux_roots(aux_derivative(p), a, b)) pieces.push_back(x);
    pieces.push_back(b);
    std::vector<double> out;
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
        double lo = pieces[i], hi = pieces[i + 1];
        const double f_lo = priv_polyval(p, lo), f_hi = priv_polyval(p, hi);
        if (f_lo == 0.0) out.push_back(lo);
        if (f_lo * f_hi >= 0.0) continue;
        // bisect the sign change until the ends meet
        for (int it = 0; it < 200; ++it) {
            const double mid = lo + (hi - lo) * 0.5;
            if (mid <= lo || mid >= hi) break;
            (priv_polyval(p, mid) * f_lo > 0 ? lo : hi) = mid;
        }
        out.push_back(lo + (hi - lo) * 0.5);
    }
    if (priv_polyval(p, b) == 0.0) out.push_back(b);
    return out;
}

/// The least and largest value of p at the ends and at the roots of its derivative.
std::pair<double, double> aux_range(const Poly &p, double a, double b) {
    std::vector<double> points{a, b};
    for (const double r : aux_roots(aux_derivative(p), a, b)) points.push_back(r);
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const double x : points) {
        lo = std::min(lo, priv_polyval(p, x));
        hi = std::max(hi, priv_polyval(p, x));
    }
    return {lo, hi};
}

/// The least-squares solution x of A x = b for a tall or square A (Householder QR; CORA's pinv).
Poly aux_solve(std::vector<std::vector<double>> A, std::vector<double> b) {
    const std::size_t rows = A.size(), k = A.front().size();
    for (std::size_t j = 0; j < k && j < rows; ++j) {
        // the reflection that clears the column j below the diagonal
        double norm = 0;
        for (std::size_t i = j; i < rows; ++i) norm += A[i][j] * A[i][j];
        norm = std::sqrt(norm);
        if (norm == 0) continue;
        std::vector<double> v(rows, 0.0);
        for (std::size_t i = j; i < rows; ++i) v[i] = A[i][j];
        v[j] -= A[j][j] > 0 ? -norm : norm;
        double vv = 0;
        for (std::size_t i = j; i < rows; ++i) vv += v[i] * v[i];
        if (vv == 0) continue;
        // applied to the columns j.. and to b (the column k)
        for (std::size_t c = j; c <= k; ++c) {
            double dot = 0;
            for (std::size_t i = j; i < rows; ++i) dot += v[i] * (c < k ? A[i][c] : b[i]);
            for (std::size_t i = j; i < rows; ++i) (c < k ? A[i][c] : b[i]) -= 2 * dot / vv * v[i];
        }
    }
    // back substitution
    Poly x(k, 0.0);
    for (std::size_t j = std::min(k, rows); j-- > 0;) {
        double s = b[j];
        for (std::size_t c = j + 1; c < k; ++c) s -= A[j][c] * x[c];
        x[j] = A[j][j] != 0 ? s / A[j][j] : 0.0;
    }
    return x;
}

/// The least-squares fit of degree n through (x, y), coefficients from the highest power down
/// (CORA's pinv(A) y); ridge > 0 appends sqrt(ridge) I to A, the ridge regression.
Poly aux_least_squares(const std::vector<double> &x, const std::vector<double> &y, int n,
                       double ridge) {
    // the Vandermonde matrix, with the ridge rows below it
    const std::size_t m = x.size(), k = std::size_t(n) + 1;
    std::vector<std::vector<double>> A(m, std::vector<double>(k, 0.0));
    std::vector<double> b = y;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) A[i][j] = std::pow(x[i], double(n) - double(j));
    for (std::size_t j = 0; ridge > 0 && j < k; ++j) {
        A.emplace_back(k, 0.0);
        A.back()[j] = std::sqrt(ridge);
        b.push_back(0.0);
    }
    return aux_solve(A, b);
}

std::vector<double> aux_linspace(double l, double u, std::size_t n) {
    // equidistant points, the last one exactly u
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = l + double(i) * ((u - l) / double(n - 1));
    x.back() = u;
    return x;
}

double aux_choose(int n, int k) {
    // the product form, rounded to the integer
    double r = 1;
    for (int i = 1; i <= k; ++i) r = r * double(n - k + i) / double(i);
    return std::round(r);
}

// Activations -------------------------------------------------------------------------------------

double aux_f(Activation a, double x) {
    switch (a) {
    case Activation::identity:
        return x;
    case Activation::relu:
        return std::max(x, 0.0);
    case Activation::sigmoid:
        return 1.0 / (1.0 + std::exp(-x));
    case Activation::tanh:
        return std::tanh(x);
    }
    CORA_FAIL(Error::wrong_value, "computeApproxPoly: an unknown activation");
}

/// The i-th derivative of the activation at x, i >= 0: a sigmoid's and a tanh's are polynomials in
/// the value s, with d/dx P(s) = P'(s) s (1 - s) for the sigmoid and P'(s) (1 - s^2) for the tanh.
double aux_df(Activation a, int i, double x) {
    if (i == 0) return aux_f(a, x);
    if (a == Activation::identity) return i == 1 ? 1.0 : 0.0;
    if (a == Activation::relu) return i == 1 && x > 0 ? 1.0 : 0.0;
    // P as ascending coefficients, starting with P(s) = s
    std::vector<double> P{0.0, 1.0};
    const std::vector<double> factor = a == Activation::sigmoid
                                           ? std::vector<double>{0.0, 1.0, -1.0}
                                           : std::vector<double>{1.0, 0.0, -1.0};
    for (int k = 1; k <= i; ++k) {
        std::vector<double> next(P.size() + 1, 0.0);
        for (std::size_t j = 1; j < P.size(); ++j)
            for (std::size_t m = 0; m < 3; ++m) next[j - 1 + m] += P[j] * double(j) * factor[m];
        P = next;
    }
    // Horner at the value s
    const double s = aux_f(a, x);
    double v = 0;
    for (std::size_t j = P.size(); j-- > 0;) v = v * s + P[j];
    return v;
}

/// Bounds of the derivative of an S-shaped activation over [l, u] (CORA's getDerBounds).
std::pair<double, double> aux_der_bounds(Activation a, double l, double u) {
    const auto df = [&](double x) { return aux_df(a, 1, x); };
    if (l <= 0 && u >= 0) return {df(std::abs(l) >= std::abs(u) ? l : u), df(0.0)};
    if (l > 0) return {df(u), df(l)};
    return {df(l), df(u)};
}

// The error ---------------------------------------------------------------------------------------

/// The least and largest value of p1 - p2 over [a, b] (CORA's minMaxDiffPoly).
std::pair<double, double> aux_min_max_diff(const Poly &p1, const Poly &p2, double a, double b) {
    Poly p(std::max(p1.size(), p2.size()), 0.0);
    for (std::size_t i = 0; i < p1.size(); ++i) p[p.size() - p1.size() + i] = p1[i];
    for (std::size_t i = 0; i < p2.size(); ++i) p[p.size() - p2.size() + i] -= p2[i];
    return aux_range(p, a, b);
}

/// The ReLU's error: p - relu on the two linear pieces, exact; the polynomial moves by the middle.
double aux_relu_error(Poly &coeffs, double l, double u) {
    const auto [lo1, hi1] = aux_min_max_diff(coeffs, {0.0, 0.0}, l, 0.0);
    const auto [lo2, hi2] = aux_min_max_diff(coeffs, {1.0, 0.0}, 0.0, u);
    const double lo = std::min(lo1, lo2), hi = std::max(hi1, hi2), mid = (hi + lo) / 2;
    coeffs.back() -= mid;
    return hi - mid;
}

/// The error of a sigmoid or tanh (CORA's minMaxDiffOrder): the grid maximum of a - p widened by
/// the Lipschitz bound of the grid; the polynomial moves by the middle. The derivative of a - p
/// is bounded by a' - p' (CORA's expression adds p' where this subtracts it).
double aux_sampled_error(Activation a, Poly &coeffs, double l, double u) {
    const auto [d1l, d1u] = aux_der_bounds(a, l, u);
    const auto [d2l, d2u] = aux_range(aux_derivative(coeffs), l, u);
    const double der = std::max(
        {std::abs(d1l - d2l), std::abs(d1l - d2u), std::abs(d1u - d2l), std::abs(d1u - d2u)});
    const double count = std::min(
        std::max(std::ceil((u - l) * der / numerics::kActivationFitTol), kMinPoints), kMaxPoints);
    const double tol = der * ((u - l) / count);
    // the grid, with both ends in any case
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    const auto visit = [&](double x) {
        const double diff = aux_f(a, x) - priv_polyval(coeffs, x);
        lo = std::min(lo, diff);
        hi = std::max(hi, diff);
    };
    visit(l);
    visit(u);
    for (std::size_t i = 0, n = std::size_t(count); i < n; ++i)
        visit(l + double(i) * ((u - l) / double(n - 1)));
    lo -= tol;
    hi += tol;
    coeffs.back() += (lo + hi) / 2;
    return hi - (lo + hi) / 2;
}

// The polynomials ---------------------------------------------------------------------------------

/// The Taylor polynomial of the activation at the middle of the range.
Poly aux_taylor(Activation a, double l, double u, int order) {
    const double c = (l + u) / 2;
    Poly coeffs(std::size_t(order) + 1, 0.0);
    double factorial = 1.0;
    for (int i = 0; i <= order; ++i) {
        if (i > 0) factorial *= double(i);
        // (x - c)^i f^(i)(c) / i! into the coefficients x^i .. x^0
        const double df = aux_df(a, i, c) / factorial;
        for (int k = 0; k <= i; ++k)
            coeffs[std::size_t(order - i) + std::size_t(k)] +=
                aux_choose(i, k) * std::pow(-c, k) * df;
    }
    return coeffs;
}

/// The Bernstein polynomial of degree n of the activation on [l, u]. CORA's findBernsteinPoly maps
/// the variable back with +l where t = (x - l) / (u - l) needs -l, so its polynomial is shifted.
Poly aux_bernstein(Activation a, double l, double u, int n) {
    // the polynomial in t, ascending: the sum of f(v / n) times the Bernstein basis b_{v,n}
    std::vector<double> asc(std::size_t(n) + 1, 0.0);
    for (int v = 0; v <= n; ++v) {
        const double fv = aux_f(a, l + (u - l) * (double(v) / double(n)));
        for (int k = v; k <= n; ++k)
            asc[std::size_t(k)] +=
                fv * aux_choose(n, k) * aux_choose(k, v) * ((k - v) % 2 ? -1.0 : 1.0);
    }
    // t^k = sum_j C(k, j) x^(k-j) (-l)^j / w^k with w = u - l
    Poly coeffs(std::size_t(n) + 1, 0.0);
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= k; ++j)
            coeffs[std::size_t(n - (k - j))] +=
                asc[std::size_t(k)] * aux_choose(k, j) * std::pow(-l, j) / std::pow(u - l, k);
    return coeffs;
}

/// The polynomial that matches the function and its derivatives alternately at the ends (CORA's
/// calcAlternatingDerCoeffs, as written there): the value and slope at l, then for every higher
/// derivative one row at u and one at l, solved in the least-squares sense.
Poly aux_throw_catch(Activation a, double l, double u, int order) {
    if (aux_df(a, 1, l) > aux_df(a, 1, u)) return aux_throw_catch(a, u, l, order);
    const std::size_t k = std::size_t(order) + 1;
    // the exponents order..0 and the weights 1; a derivative lowers them (not below 0)
    std::vector<double> exponents(k), weights(k, 1.0), coeffs(k, 1.0);
    for (std::size_t j = 0; j < k; ++j) exponents[j] = double(order) - double(j);
    const auto row = [&](double x) {
        std::vector<double> r(k);
        for (std::size_t j = 0; j < k; ++j) r[j] = weights[j] * std::pow(x, exponents[j]);
        return r;
    };
    const auto next = [&]() {
        for (double &e : exponents) e = std::max(0.0, e - 1);
        coeffs = aux_derivative(coeffs);
        weights = coeffs;
        weights.resize(k, 0.0);
    };
    // the rows at l: the value and the first derivative
    std::vector<std::vector<double>> X{row(l)};
    std::vector<double> Y{aux_f(a, l)};
    next();
    X.push_back(row(l));
    Y.push_back(aux_df(a, 1, l));
    // then the higher derivatives, one row at u and one at l each
    for (int i = 2; i <= (order + 2) / 2; ++i) {
        X.push_back(row(u));
        Y.push_back(aux_df(a, i, u));
        next();
        X.push_back(row(l));
        Y.push_back(aux_df(a, i, l));
    }
    return aux_solve(X, Y);
}

/// The values of the activation at the grid of 10 (order + 1) points of [l, u], fitted by degree n.
Poly aux_fit(Activation a, double l, double u, int order, double ridge) {
    const std::vector<double> x = aux_linspace(l, u, 10 * (std::size_t(order) + 1));
    std::vector<double> y;
    y.reserve(x.size());
    for (const double xi : x) y.push_back(aux_f(a, xi));
    return aux_least_squares(x, y, order, ridge);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The polynomial ----------------------------------------------------------------------------------

double priv_polyval(const std::vector<double> &p, double x) {
    double v = 0;
    for (const double c : p) v = v * x + c;
    return v;
}

PrivApprox priv_approx_poly(Activation a, double l, double u, int order, PolyMethod method) {
    if (!(l <= u)) CORA_FAIL(Error::wrong_value, "computeApproxPoly: the range must have l <= u");
    const bool relu = a == Activation::relu;
    // a range on which the activation is linear, and a range of one point (the tangent)
    if (a == Activation::identity) return {{1.0, 0.0}, 0.0};
    if (relu && u <= 0) return {{0.0, 0.0}, 0.0};
    if (relu && 0 <= l) return {{1.0, 0.0}, 0.0};
    if (l == u) {
        const double slope = aux_df(a, 1, l);
        return {{slope, aux_f(a, l) - slope * l}, 0.0};
    }
    // the polynomial of the method, and the error where the method gives it
    Poly coeffs;
    std::optional<double> d;
    switch (method) {
    // the fits on a grid of the range
    case PolyMethod::regression:
        coeffs = aux_fit(a, l, u, order, 0.0);
        break;
    case PolyMethod::ridge_regression:
        coeffs = aux_fit(a, l, u, order, numerics::kRidgeLambda);
        break;
    case PolyMethod::singh:
        if (order == 1 && !relu) {
            // Singh's line (Thm. 3.2 of the NeurIPS 2018 paper) of a monotone S-shape, as CORA's
            const double lambda = std::min(aux_df(a, 1, l), aux_df(a, 1, u));
            coeffs = {lambda, 0.5 * (aux_f(a, u) + aux_f(a, l) - lambda * (u + l))};
            d = 0.5 * (aux_f(a, u) - aux_f(a, l) - lambda * (u - l));
        } else if (order == 1) {
            const double lambda = u / (u - l);
            coeffs = {lambda, 0.5 * (u - lambda * u)};
            d = coeffs[1];
        } else if (order == 2 && relu) {
            const double ca = u / ((u - l) * (u - l));
            coeffs = {ca, -2 * ca * l, ca * l * l};
        }
        break;
    // the series and the interpolations
    case PolyMethod::taylor:
        coeffs = aux_taylor(a, l, u, order);
        break;
    case PolyMethod::bernstein:
        coeffs = aux_bernstein(a, l, u, order);
        break;
    case PolyMethod::bounds:
        coeffs = {(aux_f(a, u) - aux_f(a, l)) / (u - l), 0.0};
        break;
    case PolyMethod::throw_catch:
        if (!relu) coeffs = aux_throw_catch(a, l, u, order);
        break;
    }
    if (coeffs.empty())
        CORA_FAIL(Error::nn_layer_not_supported,
                  "computeApproxPoly: this poly_method has no polynomial of this order");
    if (d) return {coeffs, *d};
    // the error of the polynomial, which moves by the middle of the difference
    const double radius = relu ? aux_relu_error(coeffs, l, u) : aux_sampled_error(a, coeffs, l, u);
    return {coeffs, radius};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
