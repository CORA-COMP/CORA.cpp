// nonlinearSys - the class of nonlinear systems: construction and evaluation of the dynamics
//
// The constructor calls the dynamics on symbolic states and differentiates the result twice;
// the evaluators run those expressions on tensors.
//
// See also: nonlinearSys.h, reach.cpp

#include "contDynamics/nonlinearSys/nonlinearSys.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The values of the states of the columns of X (n, N): N x 1 columns, one per state.
std::vector<Tensor> aux_columns(const Tensor &X) {
    const Tensor rows = X.transpose();
    std::vector<Tensor> out;
    for (int64_t i = 0; i < X.shape()[0]; ++i) out.push_back(rows.selectCols({i}));
    return out;
}

/// The n states of a box (n, 1) as n ranges of shape (1, 1).
std::vector<Range> aux_scalars(const Range &box) {
    const std::vector<Tensor> lo = aux_columns(box.lo), hi = aux_columns(box.hi);
    std::vector<Range> out;
    for (std::size_t i = 0; i < lo.size(); ++i) out.emplace_back(lo[i], hi[i]);
    return out;
}

/// Requires a column (n, 1) of the system's dimension.
void aux_requireColumn(const Tensor &x, int64_t n, const std::string &what) {
    if (x.shape() != std::vector<int64_t>({n, 1}))
        throw std::invalid_argument("cora: expected a column of the dimension " +
                                    std::to_string(n) + " of the system for " + what);
}

} // namespace


// ===========================================  MAIN  =========================================== //

NonlinearSys::NonlinearSys(const Dynamics &f, int64_t n) : n_(n) {
    if (n < 1) throw std::invalid_argument("cora: a nonlinear system has at least one state");
    std::vector<Expr> x;
    for (int i = 0; i < n; ++i) x.push_back(Expr::var(i));
    f_ = f(x);
    if (static_cast<int64_t>(f_.size()) != n)
        throw std::invalid_argument("cora: the dynamics returned " + std::to_string(f_.size()) +
                                    " components for a system of dimension " + std::to_string(n));
    // Row-major: entry (i, j) of the Jacobian, and the n entries (i, j, k) of its Hessian row.
    for (const Expr &fi : f_)
        for (int j = 0; j < n; ++j) {
            jacobian_.push_back(fi.diff(j));
            for (int k = 0; k < n; ++k) hessian_.push_back(jacobian_.back().diff(k));
        }
}

// f at the columns of x: every state is a column of N numbers, and the expressions are elementwise.
Tensor NonlinearSys::dynamics(const Tensor &x) const {
    const std::vector<int64_t> shape = x.shape();
    if (shape.size() != 2 || shape[0] != n_)
        throw std::invalid_argument("cora: the points of a nonlinear system are columns (" +
                                    std::to_string(n_) + ", N)");
    const std::vector<Tensor> states = aux_columns(x);
    std::vector<Tensor> f;
    for (const Expr &fi : f_) f.push_back(fi.evalTensor(states));
    return Tensor::catLast(f).transpose();
}

// The n x n entries at one point, joined row by row.
Tensor NonlinearSys::jacobian(const Tensor &x) const {
    aux_requireColumn(x, n_, "the point of the Jacobian");
    const std::vector<Tensor> states = aux_columns(x);
    std::vector<Tensor> rows;
    for (int64_t i = 0; i < n_; ++i) {
        std::vector<Tensor> row;
        for (int64_t j = 0; j < n_; ++j) row.push_back(jacobian_[i * n_ + j].evalTensor(states));
        rows.push_back(Tensor::catLast(row));
    }
    return Tensor::catRows(rows);
}

// One range per component, joined into columns.
Range NonlinearSys::enclosure(const Range &box) const {
    aux_requireColumn(box.lo, n_, "the box of an enclosure");
    const std::vector<Range> states = aux_scalars(box);
    std::vector<Tensor> lo, hi;
    for (const Expr &fi : f_) {
        // Every component is enclosed over the whole box.
        const Range r = fi.enclose(states);
        lo.push_back(r.lo);
        hi.push_back(r.hi);
    }
    return {Tensor::catRows(lo), Tensor::catRows(hi)};
}

// The constant zeros of the Hessian are skipped: most entries of a typical system.
std::vector<std::optional<Range>> NonlinearSys::hessianEnclosure(const Range &box) const {
    aux_requireColumn(box.lo, n_, "the box of an enclosure");
    const std::vector<Range> states = aux_scalars(box);
    std::vector<std::optional<Range>> out;
    for (const Expr &h : hessian_)
        out.push_back(h.isZero() ? std::nullopt : std::optional<Range>(h.enclose(states)));
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
