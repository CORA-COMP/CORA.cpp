// nonlinearSys - the class of nonlinear systems: construction and evaluation of the dynamics
//
// The constructor calls the dynamics on symbolic states and differentiates the result twice.
//
// See also: nonlinearSys.h, reach.cpp

#include "contDynamics/nonlinearSys/nonlinearSys.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The values of the expressions at a point or over a box.
std::vector<double> aux_values(const std::vector<Expr> &es, const std::vector<double> &x) {
    std::vector<double> out;
    for (const Expr &e : es) out.push_back(e.eval(x));
    return out;
}

std::vector<Range> aux_enclosures(const std::vector<Expr> &es, const std::vector<Range> &box) {
    std::vector<Range> out;
    for (const Expr &e : es) out.push_back(e.enclose(box));
    return out;
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

Tensor NonlinearSys::dynamics(const Tensor &x) const {
    return Tensor::like(x, values(x.data()), {n_, 1});
}

// The four evaluators forward to the expressions.
std::vector<double> NonlinearSys::values(const std::vector<double> &x) const {
    return aux_values(f_, x);
}

std::vector<double> NonlinearSys::jacobian(const std::vector<double> &x) const {
    return aux_values(jacobian_, x);
}

std::vector<Range> NonlinearSys::enclosure(const std::vector<Range> &box) const {
    return aux_enclosures(f_, box);
}

std::vector<Range> NonlinearSys::hessianEnclosure(const std::vector<Range> &box) const {
    return aux_enclosures(hessian_, box);
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
