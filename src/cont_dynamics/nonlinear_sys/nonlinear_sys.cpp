// nonlinear_sys - the class of nonlinear systems: construction and evaluation of the dynamics
//
// The constructor calls the dynamics on symbolic states and inputs, z = (x, u), and
// differentiates the result twice; the third derivatives are derived on their first use. The
// evaluators run those expressions on tensors, or over boxes, all entries of a derivative in one
// pass that shares their common subexpressions.
//
// See also: cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h, reach.cpp
//
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m
// CORA: contDynamics/@contDynamics/derivatives.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "global/format.h"

#include <array>
#include <mutex>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The message of a point that is not a column (n, 1) of the system's dimension.
std::string aux_column_message(const Tensor &x, int64_t n, const std::string &what) {
    return what + " must be a column (" + std::to_string(n) + ", 1), got the shape " +
           shape_text(x.shape());
}

/// Whether x is a matrix (n, ...) whose number of columns is `columns` (any if negative).
bool aux_is_matrix(const Tensor &x, int64_t n, int64_t columns) {
    return x.defined() && x.shape().size() == 2 && x.shape()[0] == n &&
           (columns < 0 || x.shape()[1] == columns);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

/// The third derivatives (i, j, k, l), derived by the first call that needs them.
struct NonlinearSys::ThirdOrder {
    std::once_flag once;
    std::vector<Expr> T;
};

// f(x) is f(x, u) without inputs.
NonlinearSys::NonlinearSys(const Dynamics &f, int64_t n)
    : NonlinearSys([&f](const std::vector<Expr> &x, const std::vector<Expr> &) { return f(x); }, n,
                   0) {}

NonlinearSys::NonlinearSys(const InputDynamics &f, int64_t n, int64_t m)
    : n_(n), m_(m), third_(std::make_shared<ThirdOrder>()) {
    detail::Entry entry;
    CORA_CHECK(n >= 1 && m >= 0, Error::wrong_input_in_constructor, [&] {
        return "NonlinearSys: needs at least one state and m >= 0 inputs, got n = " +
               std::to_string(n) + ", m = " + std::to_string(m);
    });
    // the symbolic states and inputs, and f on them
    std::vector<Expr> x, u;
    for (int64_t i = 0; i < n; ++i) x.push_back(Expr::var(int(i)));
    for (int64_t i = 0; i < m; ++i) u.push_back(Expr::var(int(n + i)));
    f_ = f(x, u);
    CORA_CHECK(static_cast<int64_t>(f_.size()) == n, Error::wrong_input_in_constructor, [&] {
        return "NonlinearSys: the dynamics returned " + std::to_string(f_.size()) +
               " components for a system of dimension " + std::to_string(n);
    });
    // Row-major over z = (x, u): the Jacobian (i, j) and its symmetric Hessian (i, j, k).
    const int64_t d = n + m;
    hessian_.assign(std::size_t(n * d * d), Expr(0.0));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < d; ++j) {
            jacobian_.push_back(f_[std::size_t(i)].diff(int(j)));
            for (int64_t k = j; k < d; ++k) {
                const Expr h = jacobian_.back().diff(int(k));
                hessian_[std::size_t((i * d + j) * d + k)] = h;
                hessian_[std::size_t((i * d + k) * d + j)] = h;
            }
        }
}

// The variables -----------------------------------------------------------------------------------

void NonlinearSys::validate_point(const char *op, const Tensor &x,
                                  const std::optional<Tensor> &u) const {
    detail::Validation entry;
    CORA_CHECK(aux_is_matrix(x, n_, 1), Error::dimension_mismatch,
               [&] { return aux_column_message(x, n_, std::string(op) + ": the point"); });
    CORA_CHECK(m_ == 0 || (u && aux_is_matrix(*u, m_, 1)), Error::dimension_mismatch, [&] {
        return std::string(op) + ": the input must be a column (" + std::to_string(m_) + ", 1)";
    });
}

std::vector<Tensor> NonlinearSys::variables(const Tensor &x, const std::optional<Tensor> &u) const {
    std::vector<Tensor> z = variable_columns(x);
    if (m_ == 0) return z;
    // a single input column serves every point
    Tensor U = *u;
    if (U.shape()[1] == 1 && x.shape()[1] != 1) U = U.matmul(x.ones_like({1, x.shape()[1]}));
    for (Tensor &column : variable_columns(U)) z.push_back(std::move(column));
    return z;
}

std::vector<Range> NonlinearSys::variables(const Range &x, const std::optional<Range> &u) const {
    const std::vector<Tensor> lo = variables(x.lo, u ? std::optional<Tensor>(u->lo) : std::nullopt);
    const std::vector<Tensor> hi = variables(x.hi, u ? std::optional<Tensor>(u->hi) : std::nullopt);
    std::vector<Range> out;
    out.reserve(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) out.emplace_back(lo[i], hi[i]);
    return out;
}

// Evaluation --------------------------------------------------------------------------------------

// f at the columns of x: every variable is a column of N numbers, evaluated elementwise.
Tensor NonlinearSys::dynamics(const Tensor &x, const std::optional<Tensor> &u) const {
    detail::Entry entry;
    CORA_CHECK(aux_is_matrix(x, n_, -1), Error::dimension_mismatch, [&] {
        return "dynamics: the points must be columns (" + std::to_string(n_) + ", N), got " +
               (x.defined() ? shape_text(x.shape()) : std::string("none"));
    });
    CORA_CHECK(m_ == 0 || (u && (aux_is_matrix(*u, m_, x.shape()[1]) || aux_is_matrix(*u, m_, 1))),
               Error::dimension_mismatch, [&] {
                   return "dynamics: the inputs must be columns (" + std::to_string(m_) +
                          ", N) or one column (" + std::to_string(m_) + ", 1)";
               });
    return Tensor::concat(Expr::eval_tensor(f_, variables(x, u)), -1).matrix_transpose();
}

// A and B are the first n and the last m columns of the Jacobian over z.
Jacobians NonlinearSys::jacobian(const Tensor &x, const std::optional<Tensor> &u) const {
    detail::Entry entry;
    if (entry.check()) validate_point("jacobian", x, u);
    const Tensor J = Expr::eval_matrix(jacobian_, 0, n_, n_ + m_, variables(x, u), x);
    return {J.slice(-1, 0, n_), m_ == 0 ? x.zeros_like({n_, 0}) : J.slice(-1, n_, n_ + m_)};
}

// One range per component, joined into a column.
Range NonlinearSys::enclosure(const Range &x, const std::optional<Range> &u) const {
    detail::Entry entry;
    if (entry.check())
        validate_point("enclosure", x.lo, u ? std::optional<Tensor>(u->lo) : std::nullopt);
    std::vector<Tensor> lo, hi;
    for (const Range &r : Expr::enclose(f_, variables(x, u))) {
        lo.push_back(r.lo);
        hi.push_back(r.hi);
    }
    return {Tensor::concat(lo, -2), Tensor::concat(hi, -2)};
}

// Second and third derivatives --------------------------------------------------------------------

std::vector<Tensor> NonlinearSys::hessian(const Tensor &x, const std::optional<Tensor> &u) const {
    detail::Entry entry;
    if (entry.check()) validate_point("hessian", x, u);
    const int64_t d = n_ + m_;
    const std::vector<Tensor> z = variables(x, u);
    std::vector<Tensor> out;
    out.reserve(std::size_t(n_));
    for (int64_t i = 0; i < n_; ++i)
        out.push_back(Expr::eval_matrix(hessian_, std::size_t(i * d * d), d, d, z, x));
    return out;
}

// The constant zeros of the Hessian are skipped: most entries of a typical system.
std::vector<std::optional<Range>>
NonlinearSys::hessian_enclosure(const Range &x, const std::optional<Range> &u) const {
    detail::Entry entry;
    if (entry.check())
        validate_point("hessian_enclosure", x.lo, u ? std::optional<Tensor>(u->lo) : std::nullopt);
    return Expr::enclose_nonzero(hessian_, variables(x, u));
}

// T(i, j, k, l) is symmetric in j, k and l: each set of three indices is derived once.
std::vector<std::optional<Range>>
NonlinearSys::third_order_enclosure(const Range &x, const std::optional<Range> &u) const {
    detail::Entry entry;
    if (entry.check())
        validate_point("third_order_enclosure", x.lo,
                       u ? std::optional<Tensor>(u->lo) : std::nullopt);
    const int64_t d = n_ + m_;
    std::call_once(third_->once, [&] {
        std::vector<Expr> &T = third_->T;
        T.assign(std::size_t(n_ * d * d * d), Expr(0.0));
        auto at = [d](int64_t i, const std::array<int64_t, 3> &p) {
            return std::size_t(((i * d + p[0]) * d + p[1]) * d + p[2]);
        };
        for (int64_t i = 0; i < n_; ++i)
            for (int64_t j = 0; j < d; ++j)
                for (int64_t k = j; k < d; ++k)
                    for (int64_t l = k; l < d; ++l) {
                        // the six orders of (j, k, l) share the expression
                        const Expr t = hessian_[std::size_t((i * d + j) * d + k)].diff(int(l));
                        for (const std::array<int64_t, 3> &p :
                             {std::array<int64_t, 3>{j, k, l}, std::array<int64_t, 3>{j, l, k},
                              std::array<int64_t, 3>{k, j, l}, std::array<int64_t, 3>{k, l, j},
                              std::array<int64_t, 3>{l, j, k}, std::array<int64_t, 3>{l, k, j}})
                            T[at(i, p)] = t;
                    }
    });
    return Expr::enclose_nonzero(third_->T, variables(x, u));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
