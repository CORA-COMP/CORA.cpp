// priv_taylor_lin_sys - cache of the Taylor quantities of a linear system, as CORA's taylorLinSys
//
// The methods of TaylorLinSys, and the correction matrices of priv_correctionMatrixState and
// priv_correctionMatrixInput for an adaptive truncation order: the series stops when a term no
// longer changes the sums in floating-point precision and fails after 75 terms.
//
// Syntax:   TaylorLinSys taylor(A);   Tensor e_adt = taylor.e_adt(0.1);   auto F = taylor.F(0.1);
// Inputs:   A - system matrix (n, n)
// Outputs:  the cached quantities of priv_taylorLinSys.h
// See also: priv_reach_adaptive, priv_verify_ra_zonotope

#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "cont_dynamics/linear_sys/private/priv_taylor_lin_sys.h"
#include "cont_dynamics/linear_sys/private/priv_verify.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kMaxEta = 75;                     // terms before a Taylor sum counts as diverging

/// Whether a <= b holds for every entry.
bool aux_all_le(const Tensor &a, const Tensor &b) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : (a - b).max(-1, true).data()) m = std::max(m, v);
    return m <= 0;
}

/// The weight of the i-th Taylor term in the correction matrices: i^(-i/(i-1)) - i^(-1/(i-1)).
double aux_weight(int i) {
    return std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Cache -------------------------------------------------------------------------------------------

TaylorLinSys::TaylorLinSys(Tensor A) : A_(std::move(A)) { Apower_.push_back(A_); }

const TaylorLinSys::PerStep *TaylorLinSys::find(double dt) const {
    for (const PerStep &s : steps_)
        if (within_tol(s.dt, dt, numerics::kTaylorStepMatch)) return &s;
    return nullptr;
}

TaylorLinSys::PerStep &TaylorLinSys::entry(double dt) {
    for (PerStep &s : steps_)
        if (within_tol(s.dt, dt, numerics::kTaylorStepMatch)) return s;
    steps_.push_back({dt, {}, std::nullopt, std::nullopt, std::nullopt});
    return steps_.back();
}

Tensor TaylorLinSys::Apower(int i) {
    while (static_cast<int>(Apower_.size()) < i) Apower_.push_back(Apower_.back().matmul(A_));
    return Apower_[i - 1];
}

// The positive part is cached like the power it belongs to.
Tensor TaylorLinSys::Apos(int i) {
    Apower(i);
    if (static_cast<int>(Apos_.size()) < i) Apos_.resize(i);
    if (!Apos_[i - 1].defined()) Apos_[i - 1] = Apower_[i - 1].maximum(0.0);
    return Apos_[i - 1];
}

Tensor TaylorLinSys::Aneg(int i) {
    Apower(i);
    if (static_cast<int>(Aneg_.size()) < i) Aneg_.resize(i);
    if (!Aneg_[i - 1].defined()) Aneg_[i - 1] = Apower_[i - 1].minimum(0.0);
    return Aneg_[i - 1];
}

// dt^i / i! is built up term by term, as it is used in every Taylor series.
double TaylorLinSys::dtoverfac(double dt, int i) {
    std::vector<double> &v = entry(dt).dtoverfac;
    if (v.empty()) v.push_back(dt);
    while (static_cast<int>(v.size()) < i) {
        const int k = static_cast<int>(v.size()) + 1;
        v.push_back(v.back() * dt / k);
    }
    return v[i - 1];
}

Tensor TaylorLinSys::e_adt(double dt) {
    PerStep &s = entry(dt);
    if (!s.e_adt) s.e_adt = (A_ * dt).expm();
    return *s.e_adt;
}

// A value cached by insert_e_adt (a product of matrices) is read like a computed one.
std::optional<Tensor> TaylorLinSys::read_e_adt(double dt) const {
    const PerStep *s = find(dt);
    return s ? s->e_adt : std::nullopt;
}

void TaylorLinSys::insert_e_adt(double dt, Tensor value) { entry(dt).e_adt = std::move(value); }

// A^-1 is computed on first use: A may be singular.
std::optional<Tensor> TaylorLinSys::Ainv() {
    if (!AinvDone_) {
        Ainv_ = priv_inverse(A_);
        AinvDone_ = true;
    }
    return Ainv_;
}

// Correction matrices -----------------------------------------------------------------------------

std::optional<Interval> TaylorLinSys::F(double dt) {
    if (const PerStep *s = find(dt); s && s->F) return s->F;
    Tensor sum_pos = A_.zeros_like(), sum_neg = A_.zeros_like();
    for (int eta = 2;; ++eta) {
        // The weight is negative: the negative part of A^eta bounds F from above.
        const double factor = aux_weight(eta) * dtoverfac(dt, eta);
        const Tensor add_pos = Apos(eta) * factor, add_neg = Aneg(eta) * factor;
        if (aux_all_le(add_neg, sum_pos * machine_eps()) &&
            aux_all_le(sum_neg * machine_eps(), add_pos))
            break;
        if (eta == kMaxEta) return std::nullopt;
        sum_pos = sum_pos + add_neg;
        sum_neg = sum_neg + add_pos;
    }
    entry(dt).F = Interval(sum_neg, sum_pos);
    return entry(dt).F;
}

std::optional<Interval> TaylorLinSys::G(double dt) {
    if (const PerStep *s = find(dt); s && s->G) return s->G;
    Tensor sum_pos = A_.zeros_like(), sum_neg = A_.zeros_like();
    for (int eta = 2;; ++eta) {
        // The terms of G use one power of A less than those of F for the same weight.
        const double factor = aux_weight(eta) * dtoverfac(dt, eta);
        const Tensor add_pos = Apos(eta - 1) * factor, add_neg = Aneg(eta - 1) * factor;
        if (aux_all_le(add_neg, sum_pos * machine_eps()) &&
            aux_all_le(sum_neg * machine_eps(), add_pos))
            break;
        if (eta == kMaxEta + 1) return std::nullopt;
        sum_pos = sum_pos + add_neg;
        sum_neg = sum_neg + add_pos;
    }
    entry(dt).G = Interval(sum_neg, sum_pos);
    return entry(dt).G;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
