// priv_float_step - the linear step of the nonlinear algorithms, as CORALean's reachFloatInput
//
// See also: priv_float_step.h
//
// CORA.cpp only: CORA's oneStep puts the input curvature in the window; the step follows
// CORALean's LinearSys.reachFloatInput and NonlinearSys.oneStepFloat.
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_step.h"
#include "cont_set/interval/private/priv_matmul.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The scalars -------------------------------------------------------------------------------------
// Every rounded operation of CORALean's float layer is the machine's at nearest, in its order.

/// k!, the product (k) * ((k-1) * ...) of CORALean's factorialDown.
double aux_factorial(int k) {
    double f = 1;
    for (int i = 1; i <= k; ++i) f = std::max(1.0, double(i) * f);
    return f;
}

/// a^k as a * (a * (... * 1)), CORALean's powUp.
double aux_power(int k, double a) {
    double p = 1;
    for (int i = 0; i < k; ++i) p = a * p;
    return p;
}

/// [r^k / k!] as CORALean's taylorCoeffBounds: the bounds of the quotient with 1/k! enclosed.
std::pair<double, double> aux_taylor_coeff(double r, int k) {
    const double inv = 1.0 / aux_factorial(k), p = aux_power(k, r);
    return {p * std::max(0.0, inv), p * inv};
}

/// The remainder bound of the series at order eta for the norm nrm: nrm^(eta+1) / (eta+1)! * 2.
double aux_remainder_bound(double nrm, int eta) {
    return (aux_power(eta + 1, nrm) / aux_factorial(eta + 1)) * (1.0 + 1.0);
}

/// -(q q) with q = (i - 1) / i: CORALean's lower bound of the correction coefficient.
double aux_corr_coeff_low(int i) {
    const double q = double(i - 1) / std::max(1.0, double(i));
    return -(q * q);
}

// The interval matrices ---------------------------------------------------------------------------

/// The largest row sum of the magnitudes of the entries, each sum nested to the right, floored at
/// 0 (CORALean's normUpperBound).
double aux_norm_upper_bound(const Interval &M) {
    const std::vector<double> lo = M.inf().data(), hi = M.sup().data();
    const std::size_t n = std::size_t(M.inf().size(-1)), rows = lo.size() / n;
    double nrm = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0;
        for (std::size_t j = n; j-- > 0;)
            sum = std::max(std::abs(lo[i * n + j]), std::abs(hi[i * n + j])) + sum;
        nrm = std::max(nrm, sum);
    }
    return nrm;
}

/// The box scaled by a number within [lo, hi], the hull of the corner products.
Interval aux_smul(const Interval &M, double lo, double hi) {
    const Tensor a = Tensor::like(M.inf(), {lo}, {1, 1}), b = Tensor::like(M.inf(), {hi}, {1, 1});
    return M.times(Interval(a, b));
}

/// The box with every entry [-w, w].
Interval aux_symmetric(const Tensor &like, double w) {
    return {like.zeros_like() - w, like.zeros_like() + w};
}

/// The data of a step of length r: the powers of [A] and what the sets of the step are made of.
class Series {
  public:
    Series(const Tensor &A, double r, int eta)
        : r_(r), eta_(eta), point_(A, A), scaled_(aux_smul(point_, r, r)),
          remainder_(aux_remainder_bound(aux_norm_upper_bound(scaled_), eta)),
          spread_(aux_symmetric(A, r * remainder_)) {
        powers_.push_back(Interval(A.eye_like(), A.eye_like()));
        for (int i = 1; i <= eta; ++i) powers_.push_back(priv_matmul(point_, powers_.back()));
        // the box of every order of the input solution, and of the remainder
        for (int i = 0; i <= eta; ++i) {
            const auto [lo, hi] = aux_taylor_coeff(r, i + 1);
            terms_.push_back(aux_smul(powers_[std::size_t(i)], lo, hi));
        }
    }

    /// [e^{A r}], the Taylor sum of [r A] with the remainder in every entry.
    Interval exp() const {
        Interval P(scaled_.inf().eye_like(), scaled_.inf().eye_like());
        Interval S = aux_smul(P, 1.0, 1.0);
        for (int k = 0; k < eta_; ++k) {
            P = priv_matmul(scaled_, P);
            const double inv = 1.0 / aux_factorial(k + 1);
            S = aux_smul(P, inv, inv).plus(S);
        }
        return {S.inf() - remainder_, S.sup() + remainder_};
    }

    /// [F], the orders 2 to eta each spanning the interval between it and 0, then the remainder.
    Interval correction() const {
        Tensor lo = point_.inf().zeros_like(), hi = lo;
        for (int i = eta_; i >= 2; --i) {
            const double c = aux_corr_coeff_low(i) * aux_taylor_coeff(r_, i).second;
            const Interval T = aux_smul(powers_[std::size_t(i)], c, 0.0);
            lo = T.inf().minimum(lo.zeros_like()) + lo;
            hi = T.sup().maximum(hi.zeros_like()) + hi;
        }
        return {lo - remainder_, hi + remainder_};
    }

    /// The set the input V contributes over one step: the orders 0 to eta, one box each, and the
    /// remainder, added one after the other.
    Zonotope input_solution(const Zonotope &V) const {
        Zonotope S = V.linear_map(terms_[0]);
        for (std::size_t i = 1; i < terms_.size(); ++i) S = S.plus(V.linear_map(terms_[i]));
        return S.plus(V.linear_map(spread_));
    }

  private:
    double r_;
    int eta_;
    Interval point_, scaled_;
    double remainder_;
    std::vector<Interval> powers_, terms_;
    Interval spread_;
};

// The sets ----------------------------------------------------------------------------------------

/// The hull of the segments between matching points of Z1 and Z2 (CORALean's linComb): the paired
/// generators (G1 + G2) / 2, then (G1 - G2) / 2, the surplus of either copied as it is, and the
/// generator (c1 - c2) / 2. Equal counts of generators give linear_combination.
Zonotope aux_lin_comb(const Zonotope &Z1, const Zonotope &Z2) {
    const Tensor &G1 = Z1.generators(), &G2 = Z2.generators();
    const int64_t m1 = G1.size(-1), m2 = G2.size(-1), common = std::min(m1, m2);
    const Tensor a = G1.slice(-1, 0, common), b = G2.slice(-1, 0, common);
    std::vector<Tensor> parts{(a + b) / 2.0};
    if (m1 > common) parts.push_back(G1.slice(-1, common, m1));
    parts.push_back((a - b) / 2.0);
    if (m2 > common) parts.push_back(G2.slice(-1, common, m2));
    parts.push_back((Z1.center() - Z2.center()) / 2.0);
    return {(Z1.center() + Z2.center()) / 2.0, Tensor::concat(parts, -1)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<int> priv_float_order(const Tensor &A, double time_step, int taylor_terms) {
    const double nrm = aux_norm_upper_bound(aux_smul(Interval(A, A), time_step, time_step));
    const int first = std::max(taylor_terms, 2);
    for (int eta = first; eta < first + 4096; ++eta)
        if (nrm <= double(eta + 2) * 0.5) return eta;
    return std::nullopt;
}

// The matrices of the step: the powers and terms, [e^{A dt}] and [F]
struct FloatStepper::Matrices {
    Matrices(const Tensor &A, int order, double time_step)
        : series(A, time_step, order), e_adt(series.exp()), F(series.correction()), n(A.size(-1)),
          eye(A.eye_like()) {}
    Series series;
    Interval e_adt, F;
    int64_t n;
    Tensor eye;
};

// The order is searched once; without an admissible one there are no matrices
FloatStepper::FloatStepper(const Tensor &A, double time_step, int taylor_terms) {
    if (const std::optional<int> order = priv_float_order(A, time_step, taylor_terms))
        m_ = std::make_unique<Matrices>(A, *order, time_step);
}

FloatStepper::~FloatStepper() = default;
FloatStepper::FloatStepper(FloatStepper &&) noexcept = default;

bool FloatStepper::admissible() const { return m_ != nullptr; }

FloatStep FloatStepper::step(const Tensor &xs, const Zonotope &V, const Zonotope &R,
                             int zonotope_order) const {
    const Series &series = m_->series;
    const int64_t n = m_->n;
    // CORALean's canonical system maps V by B = I and adds c = 0 (-0 becomes 0)
    const Tensor zero = xs.zeros_like();
    const Zonotope Vi = V.linear_map(m_->eye);
    const Zonotope Vc(Vi.center() + zero, Vi.generators());
    // the input is V's center and V without it; a point has n zero generators (Interval.zonotope)
    const Zonotope u(Vc.center(), zero.zeros_like({n, n})), U(zero, Vc.generators());
    const Zonotope X(R.center() - xs, R.generators());
    const Zonotope Pu = series.input_solution(u);
    const Zonotope PuR =
        series.input_solution(aux_lin_comb(Zonotope(zero, zero.zeros_like({n, n})), u));
    const Zonotope PU = series.input_solution(U);
    // H(t_1) = reduce([e^{A dt}] X + Pu), H(tau_0) = hull(X, [e^{A dt}] X) + [F] X + P^R
    const Zonotope e_adt_x = X.linear_map(m_->e_adt);
    const Zonotope Htp = e_adt_x.plus(Pu).reduce(zonotope_order);
    const Zonotope Hti = aux_lin_comb(X, e_adt_x).plus(X.linear_map(m_->F)).plus(PuR);
    const auto back = [&](const Zonotope &Z) { return Zonotope(Z.center() + xs, Z.generators()); };
    return FloatStep{back(Hti.plus(PU)), back(Htp.plus(PU))};
}

std::optional<FloatStep> priv_float_one_step(const Tensor &A, const Tensor &xs, const Zonotope &V,
                                             const Zonotope &R, double time_step, int taylor_terms,
                                             int zonotope_order) {
    const FloatStepper stepper(A, time_step, taylor_terms);
    if (!stepper.admissible()) return std::nullopt;
    return stepper.step(xs, V, R, zonotope_order);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
