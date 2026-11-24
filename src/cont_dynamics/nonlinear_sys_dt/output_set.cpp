// output_set - the set of the outputs y = g(x, u), as CORA's outputSet for nonlinearSysDT
//
// The output equation is linearized at the centers of the states R and the inputs U, and the
// remainder of tensor_order_output is added: at order 2 the Hessians over the boxes of R and U
// give the box 1/2 dz' max|H_i| dz; at order 3 the Hessian at the point maps the set, and the
// third derivatives over the boxes bound the rest. A linear output equation has no error.
//
// Syntax:   Y = sys.output_set(R, U, alg);
// Inputs:   R - the states, U - the inputs (none: the input 0), alg - tensor_order_output
// Outputs:  Y - the set of the outputs and the error set that was added
// See also: reach, post
//
// CORA: contDynamics/@contDynamics/outputSet.m
// CORA authors: Mark Wetzlinger, Laura Luetzow
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys_dt/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

DiscreteStep NonlinearSysDT::output_set(const Zonotope &R, const std::optional<Zonotope> &U,
                                        const alg::DiscreteLinearization &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("output_set", R, U);
        alg.validate();
    }
    const NonlinearSys &g = output_;
    const int64_t n = dim(), m = input_dim(), r = output_dim();
    // the expansion point, and g and its Jacobians there
    const Tensor p_x = R.center();
    const std::optional<Zonotope> Uset = priv_input_set(*this, p_x, U);
    const Tensor p_u = Uset ? Uset->center() : p_x.zeros_like({0});
    const std::optional<Tensor> u_point = m > 0 ? std::optional<Tensor>(p_u) : std::nullopt;
    const Tensor y0 = g.dynamics(p_x, u_point);
    const Jacobians J = g.jacobian(p_x, u_point);
    Zonotope first = (R - p_x).linear_map(J.A);
    if (Uset) first = first + (*Uset - p_u).linear_map(J.B);
    // the error: none for a linear output equation
    Zonotope error = Zonotope(y0.zeros_like({r}), y0.zeros_like({0, r}));
    if (!g.is_affine()) {
        const Interval Ix = R.interval();
        std::optional<Interval> Iu;
        if (Uset) Iu = Uset->interval();
        if (alg.tensor_order_output == 2) {
            // the Hessians over the boxes, the largest |z| of the boxes
            const Tensor x_col = Ix.inf().expand_dims(-1), x_hi = Ix.sup().expand_dims(-1);
            std::optional<Range> u_box;
            if (Iu) u_box = Range(Iu->inf().expand_dims(-1), Iu->sup().expand_dims(-1));
            std::vector<Tensor> dz = priv_column_entries(x_col.abs().maximum(x_hi.abs()));
            if (Iu)
                for (Tensor &e : priv_column_entries(
                         Iu->inf().expand_dims(-1).abs().maximum(Iu->sup().expand_dims(-1).abs())))
                    dz.push_back(std::move(e));
            const Tensor l =
                priv_second_order_remainder(g.hessian_enclosure(Range(x_col, x_hi), u_box), dz, r);
            error = Zonotope(y0.zeros_like({r}), l.diag());
        } else {
            // the Hessian at the point maps the set around it; the third derivatives over the boxes
            Zonotope Z = Uset ? R.cart_prod(*Iu) : R;
            const Tensor p = Uset ? Tensor::concat({p_x, p_u}, -1) : p_x;
            error = (Z - p).quad_map(g.hessian(p_x, u_point)) * 0.5;
            std::optional<Interval> IUr;
            if (Iu) IUr = Interval(Iu->inf() - p_u, Iu->sup() - p_u);
            const std::optional<Interval> third =
                priv_third_order_box(g, Interval(Ix.inf() - p_x, Ix.sup() - p_x), IUr, p_x, p_u);
            if (third) error = error + *third;
        }
    }
    (void)n;
    return {(first + y0) + error, error};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
