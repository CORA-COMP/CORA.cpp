// linear_sys - lean::LinearSys, the reachability of x' = A x + B u computed by CORALean
//
// See also: lean/zonotope.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#include "cora/global/oracle/linear_sys.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

LinearSys::LinearSys(const cora::Tensor &A) : A_(Tensor::from(A)) {
    detail::Entry entry;
    CORA_CHECK(A_.rows() == A_.cols(), Error::wrong_input_in_constructor,
               [] { return std::string("lean::LinearSys: the dynamics matrix must be square"); });
}

LinearSys::LinearSys(const cora::Tensor &A, const std::optional<cora::Tensor> &B,
                     const std::optional<cora::Tensor> &C)
    : LinearSys(A) {
    if (B) B_ = Tensor::from(*B);
    if (C) C_ = Tensor::from(*C);
}

// the system's matrices go with every request
Reply LinearSys::ask_system(const char *op, std::initializer_list<Operand> more) const {
    std::vector<Operand> operands{{"A", A_}};
    if (B_) operands.push_back({"B", *B_});
    if (C_) operands.push_back({"C", *C_});
    for (const Operand &o : more) operands.push_back(o);
    return ask(op, operands);
}

ReachSet LinearSys::reach(const Zonotope &X0, double t_final, const alg::Standard &alg) const {
    detail::Entry entry;
    if (entry.check()) alg.validate();
    const double time_step = alg.time_step;
    const int taylor_terms = alg.taylor_terms, zonotope_order = alg.zonotope_order;
    CORA_CHECK(X0.center().dtype() == A_.dtype(), Error::wrong_value, [] {
        return std::string("lean reach: the system and the initial set must share one dtype");
    });
    const double steps = t_final / time_step;
    CORA_CHECK(
        t_final > 0 && std::abs(steps - std::round(steps)) <= numerics::kStepCount * steps,
        Error::wrong_value,
        [] { return std::string("lean reach: t_final must be a positive multiple of time_step"); });
    // the times are exact doubles, not values of the dtype
    const Reply r = ask("linearSysReach", {{"A", A_},
                                           {"X0", X0},
                                           {"timeStep", time_step},
                                           {"tFinal", t_final},
                                           {"taylorTerms", taylor_terms},
                                           {"zonotopeOrder", zonotope_order}});
    return {r.get<std::vector<Zonotope>>("timePoint"),
            r.get<std::vector<Zonotope>>("timeInterval")};
}

ReachSet LinearSys::reach(const Zonotope &X0, const Zonotope &U, double t_final,
                          const alg::Standard &alg) const {
    detail::Entry entry;
    if (entry.check()) alg.validate();
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("lean reach: an input set needs the matrix B"); });
    const double steps = t_final / alg.time_step;
    CORA_CHECK(
        t_final > 0 && std::abs(steps - std::round(steps)) <= numerics::kStepCount * steps,
        Error::wrong_value,
        [] { return std::string("lean reach: t_final must be a positive multiple of time_step"); });
    // the times are exact doubles; the start time is 0
    const Reply r = ask_system("linearSys.reach", {{"X0", X0},
                                                   {"U", U},
                                                   {"tStart", 0.0},
                                                   {"tFinal", t_final},
                                                   {"timeStep", alg.time_step},
                                                   {"taylorTerms", alg.taylor_terms},
                                                   {"zonotopeOrder", alg.zonotope_order},
                                                   {"linAlg", std::string("standard")}});
    return {r.get<std::vector<Zonotope>>("timePoint"),
            r.get<std::vector<Zonotope>>("timeInterval")};
}

// Operations --------------------------------------------------------------------------------------

CanonicalForm LinearSys::canonical_form(const Zonotope &U, const cora::Tensor &u_vec) const {
    // CORA.cpp's systems have no disturbance and no noise: their sets have no dimension
    const cora::Tensor none = u_vec.zeros_like({0, 1});
    const Zonotope empty(none, u_vec.zeros_like({0, 0}));
    const Reply r = ask_system("linearSys.canonicalForm", {{"U", U},
                                                           {"uVec", Tensor::from(u_vec)},
                                                           {"W", empty},
                                                           {"V", empty},
                                                           {"vVec", Tensor::from(none)}});
    return {r.get<Zonotope>("U"), r.get<Interval>("u")};
}

// one request per op of the oracle from here on
bool LinearSys::is_canonical_form() const {
    return ask_system("linearSys.isCanonicalForm", {}).get<bool>("value");
}

Zonotope LinearSys::output_set(const cora::Zonotope &R) const {
    // no input and no noise: their sets have no dimension
    const cora::Tensor none = R.center().zeros_like({0, 1});
    const Zonotope empty(none, R.center().zeros_like({0, 0}));
    return ask_system("linearSys.outputSet", {{"R", Zonotope(R)}, {"U", empty}}).get<Zonotope>("Z");
}

// the solutions of a step take the set, the step size and the order
Zonotope LinearSys::homogeneous_solution(const Zonotope &X, double time_step,
                                         int taylor_terms) const {
    return ask_system("linearSys.homogeneousSolution",
                      {{"X", X}, {"timeStep", time_step}, {"taylorTerms", taylor_terms}})
        .get<Zonotope>("Z");
}

// the input solutions of a step, with the step size r as the oracle names it
Zonotope LinearSys::particular_solution_time_varying(const Zonotope &V, double time_step,
                                                     int taylor_terms) const {
    return ask_system("linearSys.particularSolution_timeVarying",
                      {{"V", V}, {"r", time_step}, {"taylorTerms", taylor_terms}})
        .get<Zonotope>("Z");
}

Zonotope LinearSys::particular_solution_constant(const Zonotope &V, double time_step,
                                                 int taylor_terms) const {
    return ask_system("linearSys.particularSolution_constant",
                      {{"V", V}, {"r", time_step}, {"taylorTerms", taylor_terms}})
        .get<Zonotope>("Z");
}

Zonotope LinearSys::affine_solution(const Zonotope &V, double time_step, int taylor_terms) const {
    return ask_system("linearSys.affineSolution",
                      {{"V", V}, {"r", time_step}, {"taylorTerms", taylor_terms}})
        .get<Zonotope>("Z");
}

// The steps ---------------------------------------------------------------------------------------

// whole steps: a step of reach, and the homogeneous step the oracle calls post
Step LinearSys::one_step(const Zonotope &X, const Zonotope &U, const cora::Tensor &u,
                         double time_step, int taylor_terms) const {
    // the oracle names the constant input u and the step size timeStep
    const Reply r = ask_system("linearSys.oneStep", {{"X", X},
                                                     {"U", U},
                                                     {"u", Tensor::from(u)},
                                                     {"timeStep", time_step},
                                                     {"taylorTerms", taylor_terms}});
    return {r.get<Zonotope>("timePoint"), r.get<Zonotope>("timeInterval")};
}

std::optional<Step> LinearSys::one_step_closed_form(const Zonotope &X, const Zonotope &U,
                                                    const cora::Tensor &u, double time_step,
                                                    int taylor_terms) const {
    // the same operands as one_step; the answer is null if no order encloses the exponential
    const Reply r = ask_system("linearSys.oneStepClosedForm", {{"X", X},
                                                               {"U", U},
                                                               {"u", Tensor::from(u)},
                                                               {"timeStep", time_step},
                                                               {"taylorTerms", taylor_terms}});
    // both sets or none
    const auto point = r.get<std::optional<Zonotope>>("timePoint");
    if (!point) return std::nullopt;
    return Step{*point, r.get<Zonotope>("timeInterval")};
}

std::optional<TaylorMatrices> LinearSys::taylor_matrices(double time_step, int taylor_terms) const {
    const Reply r = ask_system("linearSys.taylorMatrices",
                               {{"timeStep", time_step}, {"taylorTerms", taylor_terms}});
    // all four fields or none
    const auto order = r.get<std::optional<int64_t>>("order");
    if (!order) return std::nullopt;
    return TaylorMatrices{*order, r.get<Tensor>("W"), r.get<Interval>("FEntrywise"),
                          r.get<Interval>("GEntrywise")};
}

std::optional<TaylorMatrices> LinearSys::taylor_matrices_cora(double time_step,
                                                              int taylor_terms) const {
    const Reply r = ask_system("linearSys.taylorMatrices",
                               {{"timeStep", time_step}, {"taylorTerms", taylor_terms}});
    // all three fields or none
    const auto W = r.get<std::optional<Tensor>>("WCora");
    if (!W) return std::nullopt;
    return TaylorMatrices{taylor_terms, *W, r.get<Interval>("FCora"), r.get<Interval>("GCora")};
}

VerifyCorrection LinearSys::verify_correction(double time_step) const {
    const Reply r = ask_system("linearSys.verifyCorrection", {{"timeStep", time_step}});
    VerifyCorrection out{r.get<Tensor>("scales"), std::nullopt, std::nullopt};
    // an order and a matrix, or null for both
    if (const auto order = r.get<std::optional<int64_t>>("FOrder"))
        out.F = std::make_pair(*order, r.get<Interval>("F"));
    if (const auto order = r.get<std::optional<int64_t>>("GOrder"))
        out.G = std::make_pair(*order, r.get<Interval>("G"));
    return out;
}

std::optional<VerifyExpBall> LinearSys::exp_ball(double time_step) const {
    // D and r, or nothing where no halving count or order is found
    const Reply r = ask("linearSys.expBall", {{"A", A_}, {"timeStep", time_step}});
    const auto D = r.get<std::optional<Tensor>>("D");
    if (!D) return std::nullopt;
    return VerifyExpBall{*D, r.get<double>("r")};
}

VerifyEnergyGrowth LinearSys::energy_growth(double t_final) const {
    // the weight and the factor, each null where there is none
    const Reply r = ask("linearSys.energyGrowth", {{"A", A_}, {"tFinal", t_final}});
    return VerifyEnergyGrowth{r.get<std::optional<Tensor>>("P"),
                              r.get<std::optional<double>>("kappa")};
}

VerifySupportRun LinearSys::verify_support_func(const cora::Tensor &x0, const cora::Tensor &G0,
                                                const cora::Tensor &Cs, const cora::Tensor &d,
                                                double t_final, double min_step, int fuel) const {
    // the output matrix, the identity without one
    const int64_t n = A_.rows();
    std::vector<double> eye(std::size_t(n) * n, 0.0);
    for (int64_t i = 0; i < n; ++i) eye[std::size_t(i) * n + i] = 1.0;
    const Tensor C = C_ ? *C_ : Tensor(cora::Tensor::like(x0, eye, {n, n}));
    // the verdict, the statistics, and the falsifying step and halfspace if falsified
    const Reply r = ask("linearSys.verifySupportFunc", {{"A", A_},
                                                        {"x0", Tensor(x0)},
                                                        {"G0", Tensor(G0)},
                                                        {"Cs", Tensor(Cs)},
                                                        {"C", C},
                                                        {"d", Tensor(d)},
                                                        {"tFinal", t_final},
                                                        {"minStep", min_step},
                                                        {"fuel", fuel}});
    // the reply's fields in VerifySupportRun's order
    return VerifySupportRun{r.get<bool>("verified"),
                            r.get<bool>("falsified"),
                            r.get<int64_t>("iterations"),
                            r.get<int64_t>("nrSteps"),
                            r.get<double>("timeStep"),
                            r.get<std::optional<int64_t>>("falsStep"),
                            r.get<std::optional<int64_t>>("falsSpec")};
}

Step LinearSys::post(const Zonotope &X, double time_step, int taylor_terms) const {
    // the time-point and the time-interval set of one step
    const Reply r = ask_system("linearSys.post",
                               {{"X", X}, {"timeStep", time_step}, {"taylorTerms", taylor_terms}});
    return {r.get<Zonotope>("timePoint"), r.get<Zonotope>("timeInterval")};
}

std::optional<Zonotope> LinearSys::init_reach(const Zonotope &R_tau, double time_step,
                                              int taylor_terms, int step) const {
    return ask_system("linearSys.initReach", {{"Rtau", R_tau},
                                              {"timeStep", time_step},
                                              {"taylorTerms", taylor_terms},
                                              {"step", step}})
        .get<std::optional<Zonotope>>("Z");
}

// the static part is left out of the request when there is none
Zonotope LinearSys::error_solution_adaptive(const Zonotope &Vdyn,
                                            const std::optional<Zonotope> &Vstat, double time_step,
                                            int taylor_terms) const {
    const Operand times{"timeStep", time_step}, order{"taylorTerms", taylor_terms};
    if (!Vstat)
        return ask_system("linearSys.errorSolution_adaptive", {{"Vdyn", Vdyn}, times, order})
            .get<Zonotope>("Z");
    return ask_system("linearSys.errorSolution_adaptive",
                      {{"Vdyn", Vdyn}, {"Vstat", *Vstat}, times, order})
        .get<Zonotope>("Z");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
