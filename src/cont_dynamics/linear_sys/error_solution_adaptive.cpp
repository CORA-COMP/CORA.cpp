// error_solution_adaptive - the adaptive error of the input solution, as CORA's
//
// Sums the series of A^i dt^(i+1)/(i+1)! applied to Vdyn term by term and cuts it where a further
// term gains less than zeta_tabs of the largest radius (or at a given order); the remainder of
// the series is added as a box. With Vstat, the static input adds the integral of e^{A t}.
//
// Syntax:   Zonotope R = sys.error_solution_adaptive(Vdyn, Vstat, time_step, zeta_tabs);
//           sys.error_solution_adaptive(Vdyn, std::nullopt, time_step, zeta_tabs, 4);
// Inputs:   Vdyn - the time-varying part (zonotope of the dimension of A)
//           Vstat - the constant part, or none
//           time_step - step size, positive
//           zeta_tabs - the smallest relative gain that keeps the series going, in [0, 1)
//           taylor_terms - stop at this order (at least 2) instead of by the gain
// Outputs:  R - the error set
// See also: particular_solution_time_varying, expm_remainder
//
// CORA: contDynamics/@linearSys/errorSolution_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum of the lengths of the generators per coordinate (n, 1).
Tensor aux_radius(const Zonotope &Z) { return Z.generators().abs().sum(-1, true); }

/// The value of r (n, 1) at the coordinate where `at` (1, 1) points.
double aux_entry(const Tensor &r, const Tensor &at) { return r.take_along_axis(at, -2).item(); }

} // namespace


// ===========================================  MAIN  =========================================== //

// The checked entry -------------------------------------------------------------------------------

Zonotope LinearSys::error_solution_adaptive(const Zonotope &Vdyn,
                                            const std::optional<Zonotope> &Vstat, double time_step,
                                            double zeta_tabs,
                                            std::optional<int> taylor_terms) const {
    detail::Entry entry;
    // the step and the input set, then the dimension of Vstat and the options
    if (entry.check()) validate_step("error_solution_adaptive", Vdyn, time_step, 1);
    CORA_CHECK(!Vstat || Vstat->dim() == dim(), Error::dimension_mismatch, [&] {
        return std::string("error_solution_adaptive: Vstat must have the dimension of A");
    });
    CORA_CHECK(zeta_tabs >= 0 && zeta_tabs < 1 && (!taylor_terms || *taylor_terms >= 2),
               Error::wrong_value, [] {
                   return std::string("error_solution_adaptive: needs zeta_tabs in [0, 1) and "
                                      "taylor_terms of at least 2");
               });
    CORA_CHECK(A().ndim() == 2, Error::not_supported, [] {
        return std::string("error_solution_adaptive: it supports a single system, not a batch of "
                           "matrices A");
    });
    // the series itself, which also counts its terms
    int terms = 0;
    return priv_error_solution_adaptive(A(), Vdyn, Vstat, time_step, zeta_tabs, taylor_terms,
                                        terms);
}

// The series --------------------------------------------------------------------------------------

// The series itself; CORA returns the number of terms it used in options.tt_err.
Zonotope priv_error_solution_adaptive(const Tensor &A, const Zonotope &Vdyn,
                                      const std::optional<Zonotope> &Vstat, double time_step,
                                      double zeta_tabs, std::optional<int> taylor_terms,
                                      int &terms_used) {
    detail::Entry entry;
    // the powers of A, and the series built term by term
    TaylorLinSys taylor(A);
    const Tensor Aabs = A.abs();
    // the series so far: the sum of the terms, the powers of |A| of the remainder, the radii
    Zonotope AVsum = Vdyn * time_step;
    Tensor Asum = A.eye_like() * time_step, M = A.eye_like(), Aabs_power = A.eye_like();
    Tensor radius = aux_radius(AVsum), critical;
    double former = 0;
    for (int eta = 1;; ++eta) {
        Aabs_power = Aabs_power.matmul(Aabs);
        const double dt_over_fac = taylor.dtoverfac(time_step, eta);
        M = M + Aabs_power * dt_over_fac;
        const Tensor term = taylor.Apower(eta) * taylor.dtoverfac(time_step, eta + 1);
        const Zonotope AV = Vdyn.linear_map(term);
        AVsum = AVsum.plus(AV);
        Asum = Asum + term;
        radius = radius + aux_radius(AV);
        if (eta == 1) {
            // the coordinate that the gain is measured at is the largest after the first term
            critical = radius.argmax(-2, true);
            former = aux_entry(radius, critical);
            continue;
        }
        const double gain = 1 - former / aux_entry(radius, critical);
        const bool stop = taylor_terms ? eta == *taylor_terms : (gain < zeta_tabs || former == 0);
        if (!stop) {
            former = aux_entry(radius, critical);
            continue;
        }
        // the remainder of the series of e^{|A| dt} past the last term
        const Tensor W = ((Aabs * time_step).expm() - M).abs();
        const Interval E{W * -1.0, W};
        const Zonotope F = Vdyn.linear_map(E) * time_step;
        const Zonotope dynamic = AVsum.plus(F);
        terms_used = eta;
        if (!Vstat) return dynamic;
        const Interval integral{Asum + E.inf() * time_step, Asum + E.sup() * time_step};
        return dynamic.plus(Vstat->linear_map(integral));
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
