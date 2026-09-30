// priv_taylorLinSys - cache of the Taylor quantities of a linear system, as CORA's taylorLinSys
//
// The methods of TaylorLinSys, and the correction matrices of priv_correctionMatrixState and
// priv_correctionMatrixInput for an adaptive truncation order: the series stops when a term no
// longer changes the sums in floating-point precision and fails after 75 terms.
//
// Syntax:   TaylorLinSys taylor(A);   Tensor eAdt = taylor.eAdt(0.1);   auto F = taylor.F(0.1);
// Inputs:   A - system matrix (n, n)
// Outputs:  the cached quantities of priv_taylorLinSys.h
// See also: priv_reach_adaptive, priv_verifyRA_zonotope

#include "contDynamics/linearSys/private/priv_taylorLinSys.h"
#include "contDynamics/linearSys/private/priv_verify.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kEps = 2.220446049250313e-16;  // as MATLAB's eps
constexpr int kMaxEta = 75;                     // terms before a Taylor sum counts as diverging

/// Whether a <= b holds for every entry.
bool aux_allLE(const Tensor &a, const Tensor &b) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : (a - b).maxLast().data()) m = std::max(m, v);
    return m <= 0;
}

/// CORA's withinTol: equal up to an absolute or a relative tolerance.
bool aux_withinTol(double a, double b, double tol) {
    const double diff = std::abs(a - b);
    return diff <= tol || diff / std::min(std::abs(a), std::abs(b)) <= tol;
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
        if (aux_withinTol(s.dt, dt, 1e-10)) return &s;
    return nullptr;
}

TaylorLinSys::PerStep &TaylorLinSys::entry(double dt) {
    for (PerStep &s : steps_)
        if (aux_withinTol(s.dt, dt, 1e-10)) return s;
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
    if (!Apos_[i - 1].defined()) Apos_[i - 1] = Apower_[i - 1].pos();
    return Apos_[i - 1];
}

Tensor TaylorLinSys::Aneg(int i) {
    Apower(i);
    if (static_cast<int>(Aneg_.size()) < i) Aneg_.resize(i);
    if (!Aneg_[i - 1].defined()) Aneg_[i - 1] = Apower_[i - 1].neg();
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

Tensor TaylorLinSys::eAdt(double dt) {
    PerStep &s = entry(dt);
    if (!s.eAdt) s.eAdt = (A_ * dt).expm();
    return *s.eAdt;
}

// A value cached by insertEAdt (a product of matrices) is read like a computed one.
std::optional<Tensor> TaylorLinSys::readEAdt(double dt) const {
    const PerStep *s = find(dt);
    return s ? s->eAdt : std::nullopt;
}

void TaylorLinSys::insertEAdt(double dt, Tensor value) { entry(dt).eAdt = std::move(value); }

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
    Tensor sumPos = A_.zerosLike(), sumNeg = A_.zerosLike();
    for (int eta = 2;; ++eta) {
        // The weight is negative: the negative part of A^eta bounds F from above.
        const double factor = aux_weight(eta) * dtoverfac(dt, eta);
        const Tensor addPos = Apos(eta) * factor, addNeg = Aneg(eta) * factor;
        if (aux_allLE(addNeg, sumPos * kEps) && aux_allLE(sumNeg * kEps, addPos)) break;
        if (eta == kMaxEta) return std::nullopt;
        sumPos = sumPos + addNeg;
        sumNeg = sumNeg + addPos;
    }
    entry(dt).F = Interval(sumNeg, sumPos);
    return entry(dt).F;
}

std::optional<Interval> TaylorLinSys::G(double dt) {
    if (const PerStep *s = find(dt); s && s->G) return s->G;
    Tensor sumPos = A_.zerosLike(), sumNeg = A_.zerosLike();
    for (int eta = 2;; ++eta) {
        // The terms of G use one power of A less than those of F for the same weight.
        const double factor = aux_weight(eta) * dtoverfac(dt, eta);
        const Tensor addPos = Apos(eta - 1) * factor, addNeg = Aneg(eta - 1) * factor;
        if (aux_allLE(addNeg, sumPos * kEps) && aux_allLE(sumNeg * kEps, addPos)) break;
        if (eta == kMaxEta + 1) return std::nullopt;
        sumPos = sumPos + addNeg;
        sumNeg = sumNeg + addPos;
    }
    entry(dt).G = Interval(sumNeg, sumPos);
    return entry(dt).G;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
