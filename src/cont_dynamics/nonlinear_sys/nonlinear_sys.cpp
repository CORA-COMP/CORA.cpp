// nonlinear_sys - the class of nonlinear systems: construction and evaluation of the dynamics
//
// The constructor calls the dynamics on symbolic states and differentiates the result twice;
// the evaluators run those expressions on tensors.
//
// See also: cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h, reach.cpp
//
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The values of the states of the columns of X (n, N): N x 1 columns, one per state.
std::vector<Tensor> aux_columns(const Tensor &X) {
    const Tensor rows = X.transpose();
    std::vector<Tensor> out;
    for (int64_t i = 0; i < X.shape()[0]; ++i) out.push_back(rows.select_cols({i}));
    return out;
}

/// The n states of a box (n, 1) as n ranges of shape (1, 1).
std::vector<Range> aux_scalars(const Range &box) {
    const std::vector<Tensor> lo = aux_columns(box.lo), hi = aux_columns(box.hi);
    std::vector<Range> out;
    for (std::size_t i = 0; i < lo.size(); ++i) out.emplace_back(lo[i], hi[i]);
    return out;
}

/// The message of a point that is not a column (n, 1) of the system's dimension.
std::string aux_column_message(const Tensor &x, int64_t n, const std::string &what) {
    return what + " must be a column (" + std::to_string(n) + ", 1), got the shape " +
           shape_text(x.shape());
}

} // namespace


// ===========================================  MAIN  =========================================== //

NonlinearSys::NonlinearSys(const Dynamics &f, int64_t n) : n_(n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_input_in_constructor, [&] {
        return "NonlinearSys: needs at least one state, got n = " + std::to_string(n);
    });
    std::vector<Expr> x;
    for (int i = 0; i < n; ++i) x.push_back(Expr::var(i));
    f_ = f(x);
    CORA_CHECK(static_cast<int64_t>(f_.size()) == n, Error::wrong_input_in_constructor, [&] {
        return "NonlinearSys: the dynamics returned " + std::to_string(f_.size()) +
               " components for a system of dimension " + std::to_string(n);
    });
    // Row-major: entry (i, j) of the Jacobian, and the n entries (i, j, k) of its Hessian row.
    for (const Expr &fi : f_)
        for (int j = 0; j < n; ++j) {
            jacobian_.push_back(fi.diff(j));
            for (int k = 0; k < n; ++k) hessian_.push_back(jacobian_.back().diff(k));
        }
}

// f at the columns of x: every state is a column of N numbers, and the expressions are elementwise.
Tensor NonlinearSys::dynamics(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(x.shape().size() == 2 && x.shape()[0] == n_, Error::dimension_mismatch, [&] {
        return "dynamics: the points must be columns (" + std::to_string(n_) + ", N), got " +
               shape_text(x.shape());
    });
    const std::vector<Tensor> states = aux_columns(x);
    std::vector<Tensor> f;
    for (const Expr &fi : f_) f.push_back(fi.eval_tensor(states));
    return Tensor::cat_last(f).transpose();
}

// The n x n entries at one point, joined row by row.
Tensor NonlinearSys::jacobian(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(is_column(x, n_) && x.shape().size() == 2, Error::dimension_mismatch,
               [&] { return aux_column_message(x, n_, "jacobian: the point"); });
    const std::vector<Tensor> states = aux_columns(x);
    std::vector<Tensor> rows;
    for (int64_t i = 0; i < n_; ++i) {
        std::vector<Tensor> row;
        for (int64_t j = 0; j < n_; ++j) row.push_back(jacobian_[i * n_ + j].eval_tensor(states));
        rows.push_back(Tensor::cat_last(row));
    }
    return Tensor::cat_rows(rows);
}

// One range per component, joined into columns.
Range NonlinearSys::enclosure(const Range &box) const {
    detail::Entry entry;
    CORA_CHECK(is_column(box.lo, n_) && box.lo.shape().size() == 2, Error::dimension_mismatch,
               [&] { return aux_column_message(box.lo, n_, "enclosure: the box"); });
    const std::vector<Range> states = aux_scalars(box);
    std::vector<Tensor> lo, hi;
    for (const Expr &fi : f_) {
        // Every component is enclosed over the whole box.
        const Range r = fi.enclose(states);
        lo.push_back(r.lo);
        hi.push_back(r.hi);
    }
    return {Tensor::cat_rows(lo), Tensor::cat_rows(hi)};
}

// The constant zeros of the Hessian are skipped: most entries of a typical system.
std::vector<std::optional<Range>> NonlinearSys::hessian_enclosure(const Range &box) const {
    detail::Entry entry;
    CORA_CHECK(is_column(box.lo, n_) && box.lo.shape().size() == 2, Error::dimension_mismatch,
               [&] { return aux_column_message(box.lo, n_, "hessian_enclosure: the box"); });
    const std::vector<Range> states = aux_scalars(box);
    std::vector<std::optional<Range>> out;
    for (const Expr &h : hessian_)
        out.push_back(h.is_zero() ? std::nullopt : std::optional<Range>(h.enclose(states)));
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
