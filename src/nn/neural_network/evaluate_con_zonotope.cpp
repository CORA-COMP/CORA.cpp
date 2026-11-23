// evaluate_con_zonotope - the network on a constrained zonotope, as CORA's evaluateConZonotope
//
// The set becomes a star set c + G x with C x <= d and l <= x <= u (nnHelper.conversionConZonoStar
// Set): a particular solution of A x = b plus the null space of A. A linear or elementwise affine
// layer maps c and G. A ReLU is taken neuron by neuron on [lo, up], the bounds of the neuron by the
// box of the factors (options.bound_approx) or two linear programs: a neuron that is never
// positive is zeroed, an unstable one becomes a new factor y in [0, up] with the triangle's
// constraints y >= 0, y >= g x + c, y <= up / (up - lo) (g x + c - lo). The star set returns to a
// constrained zonotope (nnHelper.conversionStarSetConZono): rows that bound one factor become its
// box, the others equalities with a slack factor each. Differences to CORA, both for soundness: the
// center of the box bound is the middle of the factors' ranges (CORA's has their width) and a row
// bounds a factor only if it has no other entry. An infeasible set is the empty set. Single set.
//
// Syntax:   ConZonotope Y = nn.evaluate_con_zonotope(X);   nn.evaluate_con_zonotope(X, options);
// Inputs:   X - the input set
//           options - bound_approx (default true)
// Outputs:  Y - a constrained zonotope that contains the network's outputs on X
// See also: neural_network.h, evaluate.cpp, con_zonotope.h
// References:
//    [1] S. Bak, H.-D. Tran, K. Hobbs, T. T. Johnson. "Improved geometric path enumeration for
//        verifying ReLU neural networks", CAV 2020.
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateConZonotope.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/linprog.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/blocks.h"
#include "global/format.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The star set ------------------------------------------------------------------------------------

/// The star set c + G x (G (n, k)), C x <= d (C (r, k)), l <= x <= u, row-major on the host.
struct Star {
    std::size_t n = 0, k = 0;
    std::vector<double> c, G, C, d, l, u;
    std::size_t rows() const { return d.size(); }
};

/// The least value of f'x over C x <= d (x free); NaN if the set is empty or f is unbounded.
double aux_minimum(const std::vector<double> &f, const Star &S) {
    if (S.k == 0) return 0.0;
    LinProg lp;
    lp.n = int(S.k);
    lp.f = f;
    lp.Aineq = S.C;
    lp.bineq = S.d;
    const LinProgResult r = linprog(lp);
    return r.status == LinProgResult::Status::optimal ? r.fval
                                                      : std::numeric_limits<double>::quiet_NaN();
}

/// The star set of a constrained zonotope; false if its factors have no point (empty set).
bool aux_star(const ConZonotope &X, Star &S) {
    const int64_t m = X.nr_generators(), n = X.dim();
    const Tensor Gt = X.generators().matrix_transpose();
    S.n = std::size_t(n);
    if (X.A().shape()[X.A().ndim() - 2] == 0) {
        // the factors in the unit box
        S.k = std::size_t(m);
        S.c = X.c().data();
        S.G = Gt.data();
        S.C.assign(std::size_t(2 * m * m), 0.0);
        for (int64_t i = 0; i < m; ++i) {
            S.C[std::size_t(i * m + i)] = 1.0;
            S.C[std::size_t((m + i) * m + i)] = -1.0;
        }
        S.d.assign(std::size_t(2 * m), 1.0);
        S.l.assign(std::size_t(m), -1.0);
        S.u.assign(std::size_t(m), 1.0);
        return true;
    }
    // without generators the constraints are rows 0 = b: the set is empty unless b is zero
    if (m == 0) {
        S.k = 0;
        S.c = X.c().data();
        for (const double v : X.b().data())
            if (v != 0.0) return false;
        return true;
    }
    // the factors p + T y of the constraints, y free in the box rows [T; -T] y <= [1 - p; 1 + p]
    const Tensor p = X.A().pinv().matmul(X.b().expand_dims(-1));
    const Tensor T = null_space(X.A());
    S.k = std::size_t(T.shape()[1]);
    const std::vector<double> t = T.data(), pv = p.data();
    S.C.resize(std::size_t(2 * m) * S.k);
    S.d.resize(std::size_t(2 * m));
    for (std::size_t i = 0; i < std::size_t(m); ++i) {
        S.d[i] = 1.0 - pv[i];
        S.d[std::size_t(m) + i] = 1.0 + pv[i];
        for (std::size_t j = 0; j < S.k; ++j) {
            S.C[i * S.k + j] = t[i * S.k + j];
            S.C[(std::size_t(m) + i) * S.k + j] = -t[i * S.k + j];
        }
    }
    S.c = (X.c().expand_dims(-1) + Gt.matmul(p)).squeeze(-1).data();
    S.G = Gt.matmul(T).data();
    // the range of every factor
    S.l.resize(S.k);
    S.u.resize(S.k);
    for (std::size_t j = 0; j < S.k; ++j) {
        std::vector<double> f(S.k, 0.0);
        f[j] = 1.0;
        S.l[j] = aux_minimum(f, S);
        f[j] = -1.0;
        S.u[j] = -aux_minimum(f, S);
    }
    for (const double v : S.d)
        if (S.k == 0 && v < 0.0) return false;
    for (std::size_t j = 0; j < S.k; ++j)
        if (std::isnan(S.l[j]) || std::isnan(S.u[j])) return false;
    return true;
}

// The layers --------------------------------------------------------------------------------------

/// A linear layer: c = W c + b, G = W G.
Star aux_affine(const NeuralNetwork::Layer &L, const Star &S) {
    const std::vector<double> W = L.dense().data(), b = L.b.data();
    Star R = S;
    R.n = b.size();
    R.c.assign(R.n, 0.0);
    R.G.assign(R.n * S.k, 0.0);
    for (std::size_t i = 0; i < R.n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < S.n; ++j) {
            acc += W[i * S.n + j] * S.c[j];
            for (std::size_t q = 0; q < S.k; ++q)
                R.G[i * S.k + q] += W[i * S.n + j] * S.G[j * S.k + q];
        }
        R.c[i] = acc + b[i];
    }
    return R;
}

/// One ReLU neuron j (CORA's nnReLULayer.evaluateConZonotopeNeuron); false if the set is empty.
void aux_relu(Star &S, std::size_t j, bool bound_approx) {
    const std::size_t k = S.k;
    const std::vector<double> g(S.G.begin() + std::ptrdiff_t(j * k),
                                S.G.begin() + std::ptrdiff_t((j + 1) * k));
    // the bounds of the neuron: the box of the factors, or two linear programs
    double lo = 0.0, up = 0.0;
    if (bound_approx) {
        double mid = S.c[j], rad = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            mid += 0.5 * g[i] * (S.u[i] + S.l[i]);
            rad += std::abs(0.5 * g[i] * (S.u[i] - S.l[i]));
        }
        lo = mid - rad;
        up = mid + rad;
    } else {
        std::vector<double> minus(k);
        for (std::size_t i = 0; i < k; ++i) minus[i] = -g[i];
        lo = S.c[j] + aux_minimum(g, S);
        up = S.c[j] - aux_minimum(minus, S);
    }
    if (!(lo < 0.0)) return;
    const double cj = S.c[j];
    // the neuron zeroed in the center and the generators
    S.c[j] = 0.0;
    for (std::size_t i = 0; i < k; ++i) S.G[j * k + i] = 0.0;
    if (up <= 0.0) return;
    // a new factor y in [0, up]: y >= 0, y >= g x + c, y <= up / (up - lo) (g x + c - lo)
    const std::size_t r = S.rows();
    std::vector<double> C((r + 3) * (k + 1), 0.0);
    for (std::size_t a = 0; a < r; ++a)
        for (std::size_t i = 0; i < k; ++i) C[a * (k + 1) + i] = S.C[a * k + i];
    C[(r + 0) * (k + 1) + k] = -1.0;
    for (std::size_t i = 0; i < k; ++i) {
        C[(r + 1) * (k + 1) + i] = g[i];
        C[(r + 2) * (k + 1) + i] = -up / (up - lo) * g[i];
    }
    C[(r + 1) * (k + 1) + k] = -1.0;
    C[(r + 2) * (k + 1) + k] = 1.0;
    S.C = C;
    S.d.push_back(0.0);
    S.d.push_back(-cj);
    S.d.push_back(up / (up - lo) * (cj - lo));
    // the generators gain a column, the unit vector of the neuron
    std::vector<double> G(S.n * (k + 1), 0.0);
    for (std::size_t a = 0; a < S.n; ++a)
        for (std::size_t i = 0; i < k; ++i) G[a * (k + 1) + i] = S.G[a * k + i];
    G[j * (k + 1) + k] = 1.0;
    S.G = G;
    S.l.push_back(0.0);
    S.u.push_back(up);
    S.k = k + 1;
}

// Back to a constrained zonotope ------------------------------------------------------------------

/// Whether row a of C has one nonzero entry, equal to sign (at column col).
bool aux_unit_row(const Star &S, std::size_t a, double sign, std::size_t &col) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < S.k; ++i)
        if (S.C[a * S.k + i] != 0.0) {
            ++count;
            col = i;
        }
    return count == 1 && S.C[a * S.k + col] == sign;
}

/// The constrained zonotope of a star set (CORA's conversionStarSetConZono); empty if the
/// constraints cannot hold.
ConZonotope aux_con_zonotope(Star S, const Tensor &like) {
    const std::size_t n = S.n, k = S.k;
    // the rows scaled by their length (as CORA does)
    for (std::size_t a = 0; a < S.rows(); ++a) {
        double sum = 0.0;
        for (std::size_t i = 0; i < k; ++i) sum += S.C[a * k + i] * S.C[a * k + i];
        const double norm = std::sqrt(sum);
        for (std::size_t i = 0; i < k; ++i) S.C[a * k + i] = norm * S.C[a * k + i];
        S.d[a] = norm * S.d[a];
    }
    // the rows that bound one factor give its range; the others stay constraints
    std::vector<double> u = S.u, l = S.l;
    std::vector<bool> bound_u(k, false), bound_l(k, false), keep(S.rows(), true);
    for (std::size_t a = 0; a < S.rows(); ++a) {
        std::size_t col = 0;
        if (aux_unit_row(S, a, 1.0, col)) {
            u[col] = bound_u[col] ? std::min(u[col], S.d[a]) : S.d[a];
            bound_u[col] = true;
            keep[a] = false;
        } else if (aux_unit_row(S, a, -1.0, col)) {
            l[col] = bound_l[col] ? std::max(l[col], -S.d[a]) : -S.d[a];
            bound_l[col] = true;
            keep[a] = false;
        }
    }
    // a factor without a value leaves the empty set
    for (std::size_t i = 0; i < k; ++i)
        if (l[i] > u[i]) return ConZonotope::empty(int64_t(n));
    // the factors in the unit box: x = cen + R y
    std::vector<double> cen(k), R(k);
    for (std::size_t i = 0; i < k; ++i) {
        cen[i] = 0.5 * (l[i] + u[i]);
        R[i] = 0.5 * (u[i] - l[i]);
    }
    // the center and the generators in the unit box
    std::vector<double> c = S.c, G = S.G;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t i = 0; i < k; ++i) {
            c[a] += S.G[a * k + i] * cen[i];
            G[a * k + i] = S.G[a * k + i] * R[i];
        }
    // each remaining row C y <= d becomes an equality with a slack factor in [-1, 1]
    std::vector<std::size_t> rows;
    for (std::size_t a = 0; a < S.rows(); ++a)
        if (keep[a]) rows.push_back(a);
    const std::size_t p = rows.size(), m = k + p;
    std::vector<double> A(p * m, 0.0), b(p), Gt(m * n, 0.0);
    for (std::size_t r = 0; r < p; ++r) {
        double dr = S.d[rows[r]], lr = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            dr -= S.C[rows[r] * k + i] * cen[i];
            A[r * m + i] = S.C[rows[r] * k + i] * R[i];
            lr -= std::abs(A[r * m + i]);
        }
        if (lr > dr) return ConZonotope::empty(int64_t(n));
        b[r] = 0.5 * (lr + dr);
        A[r * m + k + r] = 0.5 * (dr - lr);
    }
    // the generators as rows
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t i = 0; i < k; ++i) Gt[i * n + a] = G[a * k + i];
    const Shape gs{int64_t(m), int64_t(n)};
    if (p == 0) return ConZonotope(Tensor::like(like, c, {int64_t(n)}), Tensor::like(like, Gt, gs));
    return ConZonotope(Tensor::like(like, c, {int64_t(n)}), Tensor::like(like, Gt, gs),
                       Tensor::like(like, A, {int64_t(p), int64_t(m)}),
                       Tensor::like(like, b, {int64_t(p)}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The evaluation ----------------------------------------------------------------------------------

ConZonotope NeuralNetwork::evaluate_con_zonotope(const ConZonotope &X,
                                                 const NnOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate_con_zonotope: the set has the dimension " + std::to_string(X.dim()) +
               ", the network takes " + std::to_string(input_dim());
    });
    CORA_CHECK(X.c().ndim() == 1, Error::not_supported, [] {
        return std::string("evaluate_con_zonotope: a constrained zonotope takes a single set");
    });
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string("evaluate_con_zonotope: a constrained zonotope takes a chain of layers");
    });
    // only the layers CORA converts
    for (const Activation a : activations_)
        CORA_CHECK(a == Activation::identity || a == Activation::relu, Error::not_supported, [] {
            return std::string(
                "evaluate_con_zonotope: a constrained zonotope takes identity and ReLU layers "
                "only (CORA's nnLayerNotSupported)");
        });
    // the star set through the layers, each ReLU neuron by neuron
    Star S;
    if (!aux_star(X, S)) return ConZonotope::empty(output_dim());
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        S = aux_affine(layers_[k], S);
        if (activations_[k] == Activation::relu)
            for (std::size_t j = 0; j < S.n; ++j) aux_relu(S, j, options.bound_approx);
    }
    return aux_con_zonotope(S, X.c());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
