// trajectory_error - the error of a linearized trajectory, as CORA's linearSys.trajectoryError
//
// The trajectory x(t) = e^{A t} x0 + (integral) u is expanded around t_c; the terms of the Taylor
// series past the linear one, in the time powers [t_min^i, t_max^i], and the remainder of the
// series give a zonotope E_err. The coefficients c_tilde = A_tilde' n and d_tilde = -n' u_tilde,
// which the error contributes to a halfspace with the normal n, are returned as boxes.
//
// Syntax:   TrajectoryError e = sys.trajectory_error({f1, f2}, u, R0, t_min, t_max, t_c, order, n);
// Inputs:   f - flow at the start and the end of the interval (one value: both)
//           u - constant input
//           R0 - initial set
//           t_min - start of the time interval
//           t_max - end of the time interval
//           t_c - expansion point
//           order - number of terms of the series, at least 1
//           n - normal vector (dim, 1)
// Outputs:  e - error (zonotope), c_tilde and d_tilde (boxes)
// See also: homogeneous_solution, expm_remainder
//
// CORA: contDynamics/@linearSys/trajectoryError.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_taylor_lin_sys.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The scalar interval [mid - rad, mid + rad] times a zonotope, as CORA's matZonotope times
/// zonotope: the center mid c, the generators mid G, rad c and rad G.
Zonotope aux_scale(const Zonotope &Z, double mid, double rad) {
    const Tensor &c = Z.center(), &G = Z.generators();
    return {c * mid, Tensor::concat({G * mid, c * rad, G * rad}, -1)};
}

// Time powers -------------------------------------------------------------------------------------

/// The time powers t_min^i and t_max^i as midpoint and radius, for i = 1..order.
struct TimePowers {
    std::vector<double> mid, rad;
};

TimePowers aux_time_powers(double t_min, double t_max, int order) {
    TimePowers t;
    double low = t_min, high = t_max;
    for (int i = 1; i <= order; ++i) {
        t.mid.push_back(0.5 * (high + low));
        t.rad.push_back(0.5 * (high - low));
        low *= t_min;
        high *= t_max;
    }
    return t;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Error of the trajectory -------------------------------------------------------------------------

TrajectoryError LinearSys::trajectory_error(const std::vector<Tensor> &f, const Tensor &u,
                                            const Zonotope &R0, double t_min, double t_max,
                                            double t_c, int order, const Tensor &n) const {
    detail::Entry entry;
    CORA_CHECK(A().ndim() == 2 && order >= 1 && t_min <= t_max, Error::wrong_value, [] {
        return std::string("trajectory_error: needs a single system, order >= 1 and "
                           "t_min <= t_max");
    });
    CORA_CHECK(f.size() == 1 || f.size() == 2, Error::wrong_value,
               [] { return std::string("trajectory_error: f holds one or two flow values"); });
    const std::vector<int64_t> column{dim(), 1};
    CORA_CHECK(R0.dim() == dim() && u.shape() == column && n.shape() == column &&
                   f[0].shape() == column && f.back().shape() == column,
               Error::dimension_mismatch, [&] {
                   return "trajectory_error: R0, f, u and n must match the " +
                          std::to_string(dim()) + " states, with u, n and f of shape (" +
                          std::to_string(dim()) + ", 1)";
               });
    // the sets of the expansion: the centered set, the start point and the input as a point
    TaylorLinSys taylor(A());
    const TimePowers t = aux_time_powers(t_min, t_max, order);
    const Zonotope x0(R0.center(), R0.center().zeros_like({dim(), 0}));
    const Zonotope Y(R0.center().zeros_like(), R0.generators());
    const Zonotope origin(R0.center().zeros_like(), R0.center().zeros_like({dim(), 0}));
    const Zonotope u_set(u, u.zeros_like({dim(), 0}));
    // powers of the expansion point, t_c^i
    std::vector<double> tc(order + 1, 1.0);
    for (int i = 1; i <= order; ++i) tc[i] = tc[i - 1] * t_c;

    // the series past the linear term: for the set, the input and the offset
    Zonotope E_err = origin, E_err_u = origin, u_tilde = origin;
    double factorial = 1.0;
    Tensor c_center = A().zeros_like({dim(), dim()});
    std::vector<Tensor> c_generators;
    for (int i = 1; i <= order; ++i) {
        factorial *= i;
        const Tensor Ai = taylor.Apower(i);
        // the terms of the linearization error start at the square of A
        if (i >= 2) {
            const Zonotope moved = aux_scale(Y, t.mid[i - 2], t.rad[i - 2])
                                       .plus(aux_scale(x0, t.mid[i - 2] - tc[i - 1], t.rad[i - 2]));
            E_err = E_err.plus(moved.linear_map(Ai / factorial));
            u_tilde = u_tilde.plus(aux_scale(x0, t.mid[0] * -tc[i - 1], t.rad[0] * -tc[i - 1])
                                       .linear_map(Ai / factorial));
            c_center = c_center + Ai * (t.mid[i - 1] / factorial);
            c_generators.push_back(Ai * (t.rad[i - 1] / factorial));
        }
        // the input enters from the first power on
        E_err_u = E_err_u.plus(aux_scale(u_set, t.mid[i - 1] - tc[i], t.rad[i - 1])
                                   .linear_map(Ai / (factorial * (i + 1))));
    }
    E_err = aux_scale(E_err.plus(E_err_u), t.mid[0], t.rad[0]);
    u_tilde = u_tilde.plus(aux_scale(E_err_u, t.mid[0], t.rad[0]));

    // the remainder of the series of e^{|A| t_max}
    const Interval E = expm_remainder(t_max, order);
    // the flow gap f1 - f2 enters scaled by the time interval
    const Zonotope flow_gap(f[0] - f.back(), f[0].zeros_like({dim(), 0}));
    const Zonotope gap = aux_scale(flow_gap, t.mid[0], t.rad[0]);
    const Zonotope remainder_u = u_set.linear_map(t_max * E);
    E_err = E_err.plus(R0.linear_map(E)).plus(remainder_u).plus(gap);
    u_tilde = u_tilde.plus(remainder_u).plus(gap);

    // c_tilde = A_tilde' n: the center maps, each generator keeps its sign, the rest adds up
    const Tensor center = c_center.matrix_transpose().matmul(n);
    Tensor radius = E.sup().matrix_transpose().matmul(n.abs());
    for (const Tensor &G : c_generators) radius = radius + G.matrix_transpose().matmul(n).abs();
    const Interval c_tilde{center - radius, center + radius};
    const Interval d_tilde = (u_tilde.linear_map(n.matrix_transpose()) * -1.0).interval();
    return {E_err, c_tilde, d_tilde};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
