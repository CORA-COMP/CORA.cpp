// nonlin_das_sys - the class of differential-algebraic systems: construction and evaluation
//
// The constructor calls f and g on symbolic states, algebraic variables and inputs, z = (x, y, u),
// and differentiates the result once; the second derivatives are derived on their first use,
// because a decomposition evaluates them on its subsystems only. The evaluators run those
// expressions on tensors, or over boxes, all entries of a derivative in one pass.
//
// See also: cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h, reach.cpp
//
// CORA: contDynamics/@nonlinDASys/nonlinDASys.m
// CORA: contDynamics/@contDynamics/derivatives.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

#include <mutex>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether x is a matrix (rows, ...) of `columns` columns (any if negative).
bool aux_is_matrix(const Tensor &x, int64_t rows, int64_t columns) {
    return x.defined() && x.shape().size() == 2 && x.shape()[0] == rows &&
           (columns < 0 || x.shape()[1] == columns);
}

/// The most Newton steps: CORA's loops are unbounded, and a Newton method that has not converged
/// by then does not.
constexpr int kMaxNewton = 100;

/// The columns of u (m, N) or (m, 1) as N columns wide, for the variables of N points.
Tensor aux_broadcast(const Tensor &u, int64_t points) {
    if (u.shape()[1] == 1 && points != 1) return u.matmul(u.ones_like({1, points}));
    return u;
}

/// The checks of the points of an evaluation: the columns x, y and the inputs u.
void aux_validate_points(const char *op, const Tensor &x, const Tensor &y, const Tensor &u,
                         int64_t n, int64_t p, int64_t m, int64_t columns) {
    detail::Validation entry;
    CORA_CHECK(aux_is_matrix(x, n, columns), Error::dimension_mismatch, [&] {
        return std::string(op) + ": the states must be columns (" + std::to_string(n) + ", " +
               (columns < 0 ? "N" : "1") + ")";
    });
    CORA_CHECK(aux_is_matrix(y, p, x.shape()[1]), Error::dimension_mismatch, [&] {
        return std::string(op) +
               ": the algebraic variables must have as many columns as the "
               "states, (" +
               std::to_string(p) + ", N)";
    });
    CORA_CHECK(m == 0 || aux_is_matrix(u, m, x.shape()[1]) || aux_is_matrix(u, m, 1),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the inputs must be columns (" + std::to_string(m) +
                          ", N) or one column (" + std::to_string(m) + ", 1)";
               });
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

/// The second derivatives (i, j, k) of the rows of f then g, derived by the first call that
/// needs them.
struct NonlinDASys::Second {
    std::once_flag once;
    std::vector<Expr> H;
};

NonlinDASys::NonlinDASys(const DaeDynamics &f, const DaeDynamics &g, int64_t n, int64_t m,
                         int64_t p)
    : n_(n), m_(m), p_(p), second_(std::make_shared<Second>()) {
    detail::Entry entry;
    CORA_CHECK(n >= 1 && m >= 0 && p >= 1, Error::wrong_input_in_constructor, [&] {
        return "NonlinDASys: needs at least one state and one algebraic variable and m >= 0 "
               "inputs, got n = " +
               std::to_string(n) + ", m = " + std::to_string(m) + ", p = " + std::to_string(p);
    });
    // the symbolic states, algebraic variables and inputs, and f and g on them
    std::vector<Expr> x, y, u;
    for (int64_t i = 0; i < n; ++i) x.push_back(Expr::var(int(i)));
    for (int64_t i = 0; i < p; ++i) y.push_back(Expr::var(int(n + i)));
    for (int64_t i = 0; i < m; ++i) u.push_back(Expr::var(int(n + p + i)));
    f_ = f(x, y, u);
    g_ = g(x, y, u);
    CORA_CHECK(static_cast<int64_t>(f_.size()) == n && static_cast<int64_t>(g_.size()) == p,
               Error::wrong_input_in_constructor, [&] {
                   return "NonlinDASys: f returned " + std::to_string(f_.size()) +
                          " components for " + std::to_string(n) + " states and g " +
                          std::to_string(g_.size()) + " for " + std::to_string(p) + " constraints";
               });
    // the Jacobian of (f, g) over z, row-major
    const int64_t d = n + p + m;
    for (const std::vector<Expr> *rows : {&f_, &g_})
        for (const Expr &row : *rows)
            for (int64_t j = 0; j < d; ++j) jacobian_.push_back(row.diff(int(j)));
}

// The variables -----------------------------------------------------------------------------------

std::vector<Tensor> NonlinDASys::variables(const Tensor &x, const Tensor &y,
                                           const Tensor &u) const {
    std::vector<Tensor> z = variable_columns(x);
    for (Tensor &column : variable_columns(y)) z.push_back(std::move(column));
    if (m_ == 0) return z;
    // a single input column serves every point
    for (Tensor &column : variable_columns(aux_broadcast(u, x.shape()[1])))
        z.push_back(std::move(column));
    return z;
}

std::vector<Range> NonlinDASys::variables(const Range &x, const Range &y, const Range &u) const {
    const std::vector<Tensor> lo = variables(x.lo, y.lo, u.lo), hi = variables(x.hi, y.hi, u.hi);
    std::vector<Range> out;
    out.reserve(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) out.emplace_back(lo[i], hi[i]);
    return out;
}

// Evaluation --------------------------------------------------------------------------------------

Tensor NonlinDASys::dynamics(const Tensor &x, const Tensor &y, const Tensor &u) const {
    detail::Entry entry;
    if (entry.check()) aux_validate_points("dynamics", x, y, u, n_, p_, m_, -1);
    return Tensor::concat(Expr::eval_tensor(f_, variables(x, y, u)), -1).matrix_transpose();
}

Tensor NonlinDASys::constraints(const Tensor &x, const Tensor &y, const Tensor &u) const {
    detail::Entry entry;
    if (entry.check()) aux_validate_points("constraints", x, y, u, n_, p_, m_, -1);
    return Tensor::concat(Expr::eval_tensor(g_, variables(x, y, u)), -1).matrix_transpose();
}

// The blocks of the Jacobian over z = (x, y, u): rows f then g, columns x, y, then u.
DaeJacobians NonlinDASys::jacobian(const Tensor &x, const Tensor &y, const Tensor &u) const {
    detail::Entry entry;
    if (entry.check()) aux_validate_points("jacobian", x, y, u, n_, p_, m_, 1);
    const int64_t d = n_ + p_ + m_;
    // the rows of f and of g, then the blocks of each
    const Tensor J = Expr::eval_matrix(jacobian_, 0, n_ + p_, d, variables(x, y, u), x);
    const Tensor Jf = J.slice(-2, 0, n_), Jg = J.slice(-2, n_, n_ + p_);
    return {Jf.slice(-1, 0, n_),
            m_ == 0 ? x.zeros_like({n_, 0}) : Jf.slice(-1, n_ + p_, d),
            Jf.slice(-1, n_, n_ + p_),
            Jg.slice(-1, 0, n_),
            m_ == 0 ? x.zeros_like({p_, 0}) : Jg.slice(-1, n_ + p_, d),
            Jg.slice(-1, n_, n_ + p_)};
}

Tensor NonlinDASys::newton_algebraic(const Tensor &x, const Tensor &y, const Tensor &u,
                                     bool add_last) const {
    Tensor out = y;
    for (int i = 0; i < kMaxNewton; ++i) {
        // the step solves F delta_y = -g; the loop ends at the first step below the threshold
        const Tensor delta_y = jacobian(x, out, u).F.solve(-constraints(x, out, u));
        const bool converged =
            delta_y.vector_norm(Tensor::Norm::two, -2).item() < numerics::kNewtonStep;
        if (!converged || add_last) out = out + delta_y;
        if (converged) return out;
    }
    CORA_FAIL(Error::not_converged, "Newton's method for the algebraic variables did not converge "
                                    "in " +
                                        std::to_string(kMaxNewton) + " steps");
}

void NonlinDASys::validate_state(const char *op, const Tensor &x, const Tensor &y) const {
    detail::Validation entry;
    CORA_CHECK(x.defined() && y.defined() && x.shape() == std::vector<int64_t>({n_, 1}) &&
                   y.shape() == std::vector<int64_t>({p_, 1}),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the state must be a column (" + std::to_string(n_) +
                          ", 1) and the algebraic variables (" + std::to_string(p_) + ", 1)";
               });
}

// Second derivatives ------------------------------------------------------------------------------

// The symmetric second derivatives of the rows of f then g, each pair of variables derived once.
DaeHessians NonlinDASys::hessian_enclosure(const Range &x, const Range &y, const Range &u) const {
    detail::Entry entry;
    if (entry.check()) aux_validate_points("hessian_enclosure", x.lo, y.lo, u.lo, n_, p_, m_, 1);
    const int64_t d = n_ + p_ + m_, rows = n_ + p_;
    // derived on first use: only the systems whose remainder is evaluated pay for them
    std::call_once(second_->once, [&] {
        std::vector<Expr> &H = second_->H;
        H.assign(std::size_t(rows * d * d), Expr(0.0));
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < d; ++j) {
                const Expr &first = jacobian_[std::size_t(i * d + j)];
                for (int64_t k = j; k < d; ++k) {
                    const Expr h = first.diff(int(k));
                    H[std::size_t((i * d + j) * d + k)] = h;
                    H[std::size_t((i * d + k) * d + j)] = h;
                }
            }
    });
    const std::vector<std::optional<Range>> all =
        Expr::enclose_nonzero(second_->H, variables(x, y, u));
    // the first n rows are those of f, the rest those of g
    const auto split = all.begin() + std::ptrdiff_t(n_ * d * d);
    return {std::vector<std::optional<Range>>(all.begin(), split),
            std::vector<std::optional<Range>>(split, all.end())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
