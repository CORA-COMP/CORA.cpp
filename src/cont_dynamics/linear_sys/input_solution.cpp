// input_solution - what the input adds over one step, as CORA's particular solutions
//
// With U = c + U0 and U0 centered, B c is held constant over the step and B U0 varies in time:
//    PU = sum_j A^j dt^(j+1)/(j+1)! B U0 + E dt B U0     (every term kept: u(t) varies)
//    Pu = sum_{j=0}^inf A^j dt^(j+1)/(j+1)! B c          (e^{[A Bc; 0 0] dt}, no truncation)
//    C  = G B c                                           (G = correction_matrix_input)
// The series of PU stops at taylor_terms; E bounds what it leaves.
//
// Syntax:   InputSolution in = sys.input_solution(U, time_step, taylor_terms);
// Inputs:   U - input set (one zonotope of the dimension of the columns of B; needs B)
//           time_step - step size, positive;  taylor_terms - order of the series, at least 1
// Outputs:  in - PU, Pu and Cinput (see linear_sys.h), a building block of reach
// See also: reach, correction_matrix_input, expm_remainder
//
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
// CORA: contDynamics/@linearSys/particularSolution_constant.m

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

InputSolution LinearSys::input_solution(const Zonotope &U, double time_step,
                                        int taylor_terms) const {
    detail::Entry entry;
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("input_solution: needs B; use LinearSys(A, B)"); });
    CORA_CHECK(U.center().shape() == std::vector<int64_t>({B_->shape().back(), 1}),
               Error::dimension_mismatch, [&] {
                   return "input_solution: U must be one zonotope of the dimension " +
                          std::to_string(B_->shape().back()) +
                          " of the columns of B, got the center " + shape_text(U.center().shape());
               });
    CORA_CHECK(time_step > 0 && taylor_terms >= 1, Error::wrong_value, [&] {
        return "input_solution: needs time_step > 0 and taylor_terms >= 1, got " +
               std::to_string(time_step) + " and " + std::to_string(taylor_terms);
    });
    const Tensor &A = A_;
    const Tensor BG = B_->matmul(U.generators()); // generators of B U0
    const Tensor u = B_->matmul(U.center());      // the constant input B c

    // M_j = A^j dt^(j+1)/(j+1)!; the time-varying part keeps each M_j B U0 as generators.
    Tensor Apower = A.eye_like();
    double dt_over_fac = time_step;
    std::vector<Tensor> generators{BG * time_step};
    for (int j = 1; j <= taylor_terms; ++j) {
        Apower = Apower.matmul(A);
        dt_over_fac *= time_step / (j + 1);
        generators.push_back((Apower * dt_over_fac).matmul(BG));
    }

    const Interval Edt = time_step * expm_remainder(time_step, taylor_terms);
    const Zonotope U0{u.zeros_like(), Tensor::cat_last(generators)};
    const Zonotope PU = U0.plus(Zonotope{u.zeros_like(), BG}.linear_map(Edt));

    // The constant part is the last column of e^{M dt} for M = [A Bc; 0 0], the input as a state.
    const int64_t n = A.shape().back();
    const Tensor top = Tensor::cat_last({A, u});
    const Tensor M =
        Tensor::cat_rows({top, Tensor::like(A, std::vector<double>(n + 1, 0.0), {1, n + 1})});
    std::vector<int64_t> state_rows(n);
    for (int64_t i = 0; i < n; ++i) state_rows[i] = i;
    const Tensor input = (M * time_step).expm().select_cols({n});
    const Tensor Pu = input.transpose().select_cols(state_rows).transpose();

    // G u for the interval matrix G: the center maps u, the radius maps |u|.
    const Interval G = correction_matrix_input(time_step, taylor_terms);
    const Zonotope Cinput{G.center().matmul(u), G.rad().matmul(u.abs()).diag()};
    return {PU, Pu, Cinput};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
