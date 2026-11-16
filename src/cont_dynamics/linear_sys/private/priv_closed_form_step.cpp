// priv_closed_form_step - CORA's oneStep in closed form, as CORALean's oneStepClosedFormFloat
//
// At nearest the interval matrices of the series are points: a box scaled by a point, a product of
// points and a sum of points have equal bounds. So the powers and the series are computed on plain
// (n, n) arrays, each product as the corner hull of the interval product would be (a NaN corner,
// 0 * Inf, is 0), and the boxes are formed where the bounds differ. [e^{A dt}] is the exception:
// its squarings multiply true intervals (the remainder box), as interval matrix products.
//
// See also: priv_closed_form_step.h
//
// CORA: contDynamics/@linearSys/oneStep.m
// CORA: contDynamics/@linearSys/taylorMatrices.m
// CORA evaluates oneStep with expm and A^-1; this is its closed form in CORALean's float
// definition (LinearSys.oneStepClosedFormFloat).
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The matrices of the series ----------------------------------------------------------------------

using Mat = std::vector<double>;

/// The first order from taylor_terms (within 4096 orders) with norm <= (order + 2) / 2.
std::optional<int> aux_admissible_order(double norm, int taylor_terms) {
    for (int eta = taylor_terms; eta < taylor_terms + 4096; ++eta)
        if (norm <= double(eta + 2) * 0.5) return eta;
    return std::nullopt;
}

/// B = |A| dt.
Mat aux_magnitude_scaled(const Tensor &A, double dt) {
    Mat B = A.abs().data();
    for (double &x : B) x = priv_float_mul(dt, x);
    return B;
}

/// The partial sums of the Taylor series of M, expIntervalTaylor: the powers by left
/// multiplication, the newest term added to the running sum. sums[k] is the sum up to order k.
std::vector<Mat> aux_taylor_sums(const Mat &M, std::size_t n, int orders) {
    Mat P = priv_float_dense_identity(n);
    std::vector<Mat> sums{priv_float_dense_scale(1.0, P)};
    for (int k = 0; k < orders; ++k) {
        P = priv_float_dense_matmul(M, P, n);
        const double inv = 1.0 / priv_float_factorial(k + 1);
        Mat S = priv_float_dense_scale(inv, P);
        for (std::size_t e = 0; e < S.size(); ++e) S[e] += sums.back()[e];
        sums.push_back(std::move(S));
    }
    return sums;
}

/// The sums of min(T, 0) and of max(T, 0) over the terms, each nested to the right (sumDown,
/// sumUp), widened by the radius r: subDown and addUp.
std::pair<Mat, Mat> aux_hull_of_terms(const std::vector<std::pair<Mat, Mat>> &terms,
                                      std::size_t entries, const Mat &r) {
    Mat lo(entries, 0.0), hi(entries, 0.0);
    for (std::size_t i = terms.size(); i-- > 0;)
        for (std::size_t e = 0; e < entries; ++e) {
            lo[e] = std::min(terms[i].first[e], 0.0) + lo[e];
            hi[e] = std::max(terms[i].second[e], 0.0) + hi[e];
        }
    for (std::size_t e = 0; e < entries; ++e) {
        lo[e] -= r[e];
        hi[e] += r[e];
    }
    return {std::move(lo), std::move(hi)};
}

/// W(j,k): e^{|A| r} enclosed at the order eta + 8, less its series at eta (CORALean !917).
Mat aux_entrywise_remainder(const Tensor &A, double r, int eta) {
    const std::size_t n = std::size_t(A.size(-1));
    const Mat B = aux_magnitude_scaled(A, r);
    const std::vector<Mat> B_sums = aux_taylor_sums(B, n, eta + 8);
    const double tail = priv_float_remainder_bound(priv_float_dense_norm(B, n), eta + 8);
    Mat W(n * n);
    for (std::size_t e = 0; e < W.size(); ++e)
        W[e] = (B_sums[std::size_t(eta) + 8][e] + tail) - B_sums[std::size_t(eta)][e];
    return W;
}

/// [e^{M r}] of the matrix a (n, n): the enclosure of order 16 at a r / 2^s, squared s times
/// (LinearSys.expSquaringFloat); the lower and the upper ends.
std::pair<Mat, Mat> aux_exp_squaring(const Mat &a, std::size_t n, double r, int halvings) {
    const Mat M = priv_float_dense_scale(priv_float_halved(r, halvings), a);
    const Mat T = aux_taylor_sums(M, n, 16).back();
    const double R = priv_float_remainder_bound(priv_float_dense_norm(M, n), 16);
    Mat inf(T), sup(T);
    for (std::size_t e = 0; e < T.size(); ++e) {
        inf[e] -= R;
        sup[e] += R;
    }
    for (int i = 0; i < halvings; ++i) {
        auto squared = priv_float_interval_matmul(inf, sup, inf, sup, n);
        inf = std::move(squared.first);
        sup = std::move(squared.second);
    }
    return {std::move(inf), std::move(sup)};
}

/// W(j,k): e^{|A| r} by scaling and squaring, less its series at eta
/// (LinearSys.expRemainderSquaringFloat).
Mat aux_squaring_remainder(const Tensor &A, double r, int eta, int halvings) {
    const std::size_t n = std::size_t(A.size(-1));
    const Mat X = aux_exp_squaring(A.abs().data(), n, r, halvings).second;
    const Mat T = aux_taylor_sums(aux_magnitude_scaled(A, r), n, eta).back();
    Mat W(n * n);
    for (std::size_t e = 0; e < W.size(); ++e) W[e] = X[e] - T[e];
    return W;
}

/// The powers A^0 to A^eta by left multiplication.
std::vector<Mat> aux_powers(const Mat &a, std::size_t n, int eta) {
    std::vector<Mat> powers{priv_float_dense_identity(n)};
    for (int i = 1; i <= eta; ++i) powers.push_back(priv_float_dense_matmul(a, powers.back(), n));
    return powers;
}

/// [F]: the orders 2 to eta of the state correction, widened by W.
std::pair<Mat, Mat> aux_state_correction(std::size_t n, double r, int eta,
                                         const std::vector<Mat> &powers, const Mat &W,
                                         double (*coeff)(int) = priv_float_corr_coeff_low) {
    std::vector<std::pair<Mat, Mat>> f_terms;
    for (int i = 2; i <= eta; ++i) {
        const double c = coeff(i) * priv_float_taylor_coeff(r, i).second;
        f_terms.push_back(priv_float_dense_box_scale(c, 0.0, powers[std::size_t(i)]));
    }
    return aux_hull_of_terms(f_terms, n * n, W);
}

/// [G]: the orders 1 to eta of the constant input, the power one above the order of the
/// coefficient, widened by r W.
std::pair<Mat, Mat> aux_input_correction(std::size_t n, double r, int eta,
                                         const std::vector<Mat> &powers, const Mat &scaled_w,
                                         double (*coeff)(int) = priv_float_corr_coeff_low) {
    std::vector<std::pair<Mat, Mat>> g_terms;
    for (int i = 0; i < eta; ++i) {
        const double c = coeff(i + 2) * priv_float_taylor_coeff(r, i + 2).second;
        g_terms.push_back(priv_float_dense_box_scale(c, 0.0, powers[std::size_t(i) + 1]));
    }
    return aux_hull_of_terms(g_terms, n * n, scaled_w);
}

/// Whether every entry of the box is finite.
bool aux_finite(const Interval &I) {
    for (const Tensor &T : {I.inf(), I.sup()})
        for (const double x : T.data())
            if (!std::isfinite(x)) return false;
    return true;
}

bool aux_finite(const Tensor &T) {
    for (const double x : T.data())
        if (!std::isfinite(x)) return false;
    return true;
}

/// r W.
Mat aux_scaled(const Mat &W, double r) {
    Mat out(W);
    for (double &x : out) x = r * x;
    return out;
}

/// The corrections of A at the order with the remainder W and the coefficient: F, G, W like A.
EntrywiseCorrections aux_corrections(const Tensor &A, double r, int order, const Mat &W,
                                     double (*coeff)(int)) {
    const std::size_t n = std::size_t(A.size(-1));
    const Shape shape{A.size(-1), A.size(-1)};
    const auto box = [&](const std::pair<Mat, Mat> &b) {
        return Interval(Tensor::like(A, b.first, shape), Tensor::like(A, b.second, shape));
    };
    const std::vector<Mat> powers = aux_powers(A.data(), n, order);
    return EntrywiseCorrections{
        order, Tensor::like(A, W, shape), box(aux_state_correction(n, r, order, powers, W, coeff)),
        box(aux_input_correction(n, r, order, powers, aux_scaled(W, r), coeff))};
}

// The sets ----------------------------------------------------------------------------------------

/// The input solution of V: the orders 0 to eta, one box each, then the remainder, added in turn.
Zonotope aux_input_solution(const std::vector<Interval> &terms, const Interval &remainder,
                            const Zonotope &V) {
    // the boxes added in turn: the centers one after the other, the generators side by side
    Zonotope first = V.linear_map(terms[0]);
    Tensor center = first.center();
    std::vector<Tensor> generators{first.generators()};
    generators.reserve(terms.size() + 1);
    const auto add = [&](const Interval &box) {
        const Zonotope Z = V.linear_map(box);
        center = center + Z.center();
        generators.push_back(Z.generators());
    };
    for (std::size_t i = 1; i < terms.size(); ++i) add(terms[i]);
    add(remainder);
    return {std::move(center), Tensor::concat(generators, -1)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<int> priv_closed_form_halvings(const Tensor &A, double time_step) {
    const std::size_t n = std::size_t(A.size(-1));
    const Mat a = A.data();
    for (int s = 0; s < 128; ++s)
        if (priv_float_dense_norm(priv_float_dense_scale(priv_float_halved(time_step, s), a), n) <=
            0.5)
            return s;
    return std::nullopt;
}

std::optional<int> priv_closed_form_order(const Tensor &A, double time_step, int taylor_terms) {
    const std::size_t n = std::size_t(A.size(-1));
    return aux_admissible_order(priv_float_dense_norm(aux_magnitude_scaled(A, time_step), n),
                                taylor_terms);
}


std::optional<EntrywiseCorrections> priv_entrywise_corrections(const Tensor &A, double time_step,
                                                               int taylor_terms) {
    // the admissible order, then the remainder at it and F and G
    const std::optional<int> order = priv_closed_form_order(A, time_step, taylor_terms);
    if (!order) return std::nullopt;
    return aux_corrections(A, time_step, *order, aux_entrywise_remainder(A, time_step, *order),
                           priv_float_corr_coeff_low);
}

std::optional<EntrywiseCorrections> priv_squaring_corrections(const Tensor &A, double time_step,
                                                              int taylor_terms) {
    // the halvings of |A| dt, then the remainder by squaring and F and G at the order asked for
    const std::optional<int> halvings = priv_closed_form_halvings(A.abs(), time_step);
    // exception 4: no count below 128 halves |A| dt to norm 1/2, the callers keep CORA's expm
    if (!halvings) return std::nullopt;
    // CORA's coefficient, bounded from below (corrCoeffLowTight)
    return aux_corrections(A, time_step, taylor_terms,
                           aux_squaring_remainder(A, time_step, taylor_terms, *halvings),
                           priv_float_corr_coeff_low_tight);
}

std::pair<std::vector<double>, std::vector<double>>
priv_exp_squaring(const std::vector<double> &a, std::size_t n, double r, int halvings) {
    return aux_exp_squaring(a, n, r, halvings);
}

std::optional<EntrywiseCorrections> priv_taylor_corrections(const Tensor &A, double time_step,
                                                            int taylor_terms) {
    // exception 1: a batch (..., n, n) keeps CORA's expm, the host matrices are one (n, n)
    if (A.ndim() != 2) return std::nullopt;
    // exception 2: a gradient keeps CORA's expm, the host matrices carry none
    if (A.requires_grad()) return std::nullopt;
    const char *choice = std::getenv("CORACPP_TAYLOR");
    const bool entrywise = choice != nullptr && std::string(choice) == "entrywise";
    if (choice != nullptr && !entrywise)
        CORA_FAIL(Error::wrong_value,
                  "CORACPP_TAYLOR: the only value is entrywise, got " + std::string(choice));
    // the last results are kept, as CORA keeps its own (eta_e is in the hundreds for a stiff A)
    struct Entry {
        std::vector<double> a;
        double dt;
        int terms;
        bool entrywise;
        std::optional<EntrywiseCorrections> value;
    };
    // one cache per thread: a shared one under a lock cost CVDP23 (8 threads) a third of its time
    thread_local std::deque<Entry> recent;
    const std::vector<double> a = A.data();
    {
        for (const Entry &e : recent)
            if (e.dt == time_step && e.terms == taylor_terms && e.entrywise == entrywise &&
                e.a == a)
                return e.value;
    }
    auto r = entrywise ? priv_entrywise_corrections(A, time_step, taylor_terms)
                       : priv_squaring_corrections(A, time_step, taylor_terms);
    // exception 3: where the powers of A or e^{|A| dt} overflow (a very stiff A), CORA's expm
    if (r && !(aux_finite(r->F) && aux_finite(r->G) && aux_finite(r->W))) r = std::nullopt;
    recent.push_back({a, time_step, taylor_terms, entrywise, r});
    if (recent.size() > 8) recent.pop_front();
    return r;
}

// The matrices ------------------------------------------------------------------------------------

// The matrices of the step: [e^{A dt}], [F], [G], the boxes of the input solution
struct ClosedFormStepper::Matrices {
    Matrices(const Tensor &A, double r, int eta, int halvings)
        : e_adt(A, A), F(A, A), G(A, A), remainder(A, A) {
        const std::size_t n = std::size_t(A.size(-1));
        const Shape shape{A.size(-1), A.size(-1)};
        const auto tensor = [&](const Mat &M) { return Tensor::like(A, M, shape); };
        const auto box = [&](const std::pair<Mat, Mat> &b) {
            return Interval(tensor(b.first), tensor(b.second));
        };
        const Mat a = A.data();
        const Mat W = aux_entrywise_remainder(A, r, eta);
        // [e^{A r}]: the enclosure of order 16 at A r / 2^s, squared s times
        e_adt = box(aux_exp_squaring(a, n, r, halvings));
        const std::vector<Mat> powers = aux_powers(a, n, eta);
        // the box of every order of the input solution
        for (int i = 0; i <= eta; ++i) {
            const auto [lo, hi] = priv_float_taylor_coeff(r, i + 1);
            terms.push_back(box(priv_float_dense_box_scale(lo, hi, powers[std::size_t(i)])));
        }
        // [-r W, r W]
        const Mat scaled_w = aux_scaled(W, r);
        Mat neg_scaled_w(scaled_w);
        for (double &x : neg_scaled_w) x = -x;
        remainder = box({neg_scaled_w, scaled_w});
        F = box(aux_state_correction(n, r, eta, powers, W));
        G = box(aux_input_correction(n, r, eta, powers, scaled_w));
    }
    Interval e_adt, F, G, remainder;
    std::vector<Interval> terms;
};

ClosedFormStepper::ClosedFormStepper(const Tensor &A, double time_step, int taylor_terms) {
    const std::optional<int> enclosed = priv_closed_form_order(A, time_step, taylor_terms);
    if (!enclosed) return;
    const std::optional<int> halvings = priv_closed_form_halvings(A, time_step);
    if (halvings) m_ = std::make_unique<Matrices>(A, time_step, *enclosed, *halvings);
}

// the matrices stay behind the pointer
ClosedFormStepper::~ClosedFormStepper() = default;
ClosedFormStepper::ClosedFormStepper(ClosedFormStepper &&) noexcept = default;

bool ClosedFormStepper::admissible() const { return m_ != nullptr; }

// the matrices of the step and the input solution, for a step from a polynomial zonotope
const Interval &ClosedFormStepper::e_adt() const { return m_->e_adt; }
const Interval &ClosedFormStepper::F() const { return m_->F; }
const Interval &ClosedFormStepper::G() const { return m_->G; }

Zonotope ClosedFormStepper::input_solution(const Zonotope &V) const {
    return aux_input_solution(m_->terms, m_->remainder, V);
}

// The step ----------------------------------------------------------------------------------------

ClosedFormStep ClosedFormStepper::step(const Zonotope &X, const Zonotope &U,
                                       const Tensor &u) const {
    const Matrices &m = *m_;
    // the constant input is a point set
    const Zonotope point(u, u.zeros_like({u.size(0), 0}));
    const Zonotope Pu = aux_input_solution(m.terms, m.remainder, point);
    const Zonotope PU = aux_input_solution(m.terms, m.remainder, U);
    // Htp = [e^{A dt}] X + Pu, Hti = hull(X, Htp) + [F] X
    const Zonotope Htp = X.linear_map(m.e_adt).plus(Pu);
    const Zonotope Hti = priv_lin_comb(X, Htp).plus(X.linear_map(m.F));
    return {Htp.plus(PU), Hti.plus(PU).plus(point.linear_map(m.G))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
