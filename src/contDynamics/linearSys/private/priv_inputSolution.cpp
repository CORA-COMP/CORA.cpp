// priv_inputSolution - what the input adds over one step, as CORA's particular solutions
//
// With U = c + U0 and U0 centered, B c is held constant over the step and B U0 varies in time:
//    PU = sum_j A^j dt^(j+1)/(j+1)! B U0 + E dt B U0     (every term kept: u(t) varies)
//    Pu = sum_{j=0}^inf A^j dt^(j+1)/(j+1)! B c          (e^{[A Bc; 0 0] dt}, no truncation)
//    C  = G B c                                           (G = correctionMatrixInput)
// The series of PU stops at taylorTerms; E bounds what it leaves.
//
// Syntax:   InputSolution in = priv_inputSolution(sys, U, timeStep, taylorTerms);
// Inputs:   sys - system with B;  U - input set;  timeStep, taylorTerms - step and series order
// Outputs:  in - PU, Pu and Cinput, see priv.h
// See also: priv_reach_standard, LinearSys::correctionMatrixInput

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

InputSolution priv_inputSolution(const LinearSys &sys, const Zonotope &U, double timeStep,
                                 int taylorTerms) {
    const Tensor &A = sys.A();
    const Tensor BG = sys.B()->matmul(U.G);  // generators of B U0
    const Tensor u = sys.B()->matmul(U.c);   // the constant input B c

    // M_j = A^j dt^(j+1)/(j+1)!; the time-varying part keeps each M_j B U0 as generators.
    Tensor Apower = A.eyeLike();
    double dtOverFac = timeStep;
    std::vector<Tensor> generators{BG * timeStep};
    for (int j = 1; j <= taylorTerms; ++j) {
        Apower = Apower.matmul(A);
        dtOverFac *= timeStep / (j + 1);
        generators.push_back((Apower * dtOverFac).matmul(BG));
    }

    const Interval Edt = timeStep * priv_expmRemainder(A, timeStep, taylorTerms);
    const Zonotope U0{u.zerosLike(), Tensor::catLast(generators)};
    const Zonotope PU = U0.plus(Zonotope{u.zerosLike(), BG}.mtimes(Edt));

    // The constant part is the last column of e^{M dt} for M = [A Bc; 0 0], the input as a state.
    const int64_t n = A.shape().back();
    const Tensor top = Tensor::catLast({A, u});
    const Tensor M = Tensor::catRows({top, Tensor::zeros({1, n + 1}, A.device())});
    std::vector<int64_t> stateRows(n);
    for (int64_t i = 0; i < n; ++i) stateRows[i] = i;
    const Tensor input = (M * timeStep).expm().selectCols({n});
    const Tensor Pu = input.transpose().selectCols(stateRows).transpose();

    // G u for the interval matrix G: the center maps u, the radius maps |u|.
    const Interval G = sys.correctionMatrixInput(timeStep, taylorTerms);
    const Zonotope Cinput{G.center().matmul(u), G.rad().matmul(u.abs()).diag()};
    return {PU, Pu, Cinput};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
