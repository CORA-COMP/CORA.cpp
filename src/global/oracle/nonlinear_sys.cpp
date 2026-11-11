// nonlinear_sys - lean::NonlinearSys, lin-adaptive reachability of x' = f(x, u) by CORALean
//
// See also: lean/linear_sys.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/nonlinear_sys.h"
#include "cora/global/error.h"

#include <stdexcept>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// the precision of the elementary functions: unused at nearest, where the oracle answers these ops
constexpr int kPrecision = 53;

// a list of booleans, as the oracle reads the masks
Wire aux_flags(const std::vector<bool> &flags) {
    Json j = Json::array();
    for (const bool b : flags) j.push(Json(b));
    return Wire(j, "");
}

// the three gains of the algorithm, a column of exact scalars
Wire aux_gains(const alg::LinearizationAdaptive &alg) {
    Json data = Json::array(), column = Json::object();
    for (const double g : alg.zeta_phi) data.push(Tensor::encode(g));
    column.set("rows", Json(int64_t(3))).set("cols", Json(int64_t(1))).set("data", data);
    return Wire(column, "");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

NonlinearSys::NonlinearSys(const cora::NonlinearSys &sys) : n_(sys.dim()), m_(sys.input_dim()) {
    f_ = Json::array();
    for (const Expr &e : sys.expressions())
        f_.push(Json::parse(e.to_json(int(n_), [](double x) { return Tensor::encode(x); })));
}

std::vector<Operand> NonlinearSys::system() const {
    return {{"n", n_}, {"m", m_}, {"f", Wire(f_, "")}, {"p", kPrecision}};
}

// The run -----------------------------------------------------------------------------------------

AdaptiveRun NonlinearSys::reach_adaptive(const Zonotope &R0, const std::optional<Zonotope> &U,
                                         double t_final,
                                         const alg::LinearizationAdaptive &alg) const {
    // the start set, the input set and the options of the algorithm
    std::vector<Operand> request = system();
    request.push_back({"R0", R0});
    if (U) request.push_back({"U", *U});
    for (const Operand &o : std::vector<Operand>{{"sets", true},
                                                 {"tFinal", t_final},
                                                 {"taylorTerms", alg.taylor_terms},
                                                 {"zonotopeOrder", alg.zonotope_order},
                                                 {"redFactor", alg.zeta_z},
                                                 {"decrFactor", alg.decr_factor},
                                                 {"zetaphi", aux_gains(alg)},
                                                 {"zetaK", alg.zeta_k},
                                                 {"fuel", alg.fuel}})
        request.push_back(o);
    // the steps with their tensor orders, the end and the sets
    const Reply r = ask("nonlinearSys.reachAdaptive", request);
    return {r.get<std::vector<double>>("steps"), r.get<std::vector<int64_t>>("tensorOrders"),
            r.get<double>("t"), r.get<Zonotope>("timePoint"),
            r.get<std::vector<Zonotope>>("timeInterval")};
}

std::optional<AdaptivePassResult> NonlinearSys::lin_reach_adaptive_pass(
    const std::optional<Zonotope> &U, const Zonotope &R, double time_step, int tensor_order,
    const cora::Tensor &error_adm, const alg::LinearizationAdaptive &alg) const {
    // the window, the input set and the state of the pass
    std::vector<Operand> request = system();
    request.push_back({"R", R});
    if (U) request.push_back({"U", *U});
    for (const Operand &o : std::vector<Operand>{{"timeStep", time_step},
                                                 {"taylorTerms", alg.taylor_terms},
                                                 {"zonotopeOrder", alg.zonotope_order},
                                                 {"tensorOrder", tensor_order},
                                                 {"errorAdm", Tensor::from(error_adm)},
                                                 {"redFactor", alg.zeta_z},
                                                 {"fuel", alg.fuel}})
        request.push_back(o);
    // the loop's failure is an error of the oracle: no pass
    try {
        const Reply r = ask("nonlinearSys.linReachAdaptivePass", request);
        return AdaptivePassResult{r.get<Zonotope>("timeInterval"),
                                  r.get<Zonotope>("timePoint"),
                                  r.get<Zonotope>("Rmax"),
                                  r.get<Tensor>("A").gather(),
                                  r.get<Tensor>("xs").gather(),
                                  r.get<Tensor>("trueError").gather(),
                                  r.get<Tensor>("abstrerr").gather(),
                                  r.get<std::vector<bool>>("Gzero")};
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

AbstrErrResult NonlinearSys::abstr_err_adaptive(const std::optional<Zonotope> &U,
                                                const cora::Tensor &xs, const Zonotope &Rmax,
                                                int tensor_order, double red_factor) const {
    // the linearization point, the set around it and the order of the error
    std::vector<Operand> request = system();
    request.push_back({"xs", Tensor::from(xs)});
    request.push_back({"Rmax", Rmax});
    if (U) request.push_back({"U", *U});
    request.push_back({"tensorOrder", tensor_order});
    request.push_back({"redFactor", red_factor});
    const Reply r = ask("nonlinearSys.abstrErrAdaptive", request);
    return {r.get<Zonotope>("VerrorDyn"), r.get<Tensor>("trueError").gather()};
}

std::vector<std::optional<int64_t>>
NonlinearSys::orders_adaptive(const std::optional<Zonotope> &U, const cora::Tensor &A,
                              const cora::Tensor &xs, const std::vector<bool> &zero_width,
                              const std::vector<bool> &g_zero) const {
    // the input set, the matrix of the linearization and the flags of the dimensions
    std::vector<Operand> request = system();
    if (U) request.push_back({"U", *U});
    request.push_back({"A", Tensor::from(A)});
    request.push_back({"xs", Tensor::from(xs)});
    request.push_back({"zeroWidthDim", aux_flags(zero_width)});
    request.push_back({"Gzero", aux_flags(g_zero)});
    return ask("nonlinearSys.ordersAdaptive", request)
        .get<std::vector<std::optional<int64_t>>>("orders");
}

// The tuning formulas -----------------------------------------------------------------------------

// the formulas read the constants of the algorithm and nothing of the dynamics
double NonlinearSys::optimal_delta_t_adaptive(const alg::LinearizationAdaptive &alg,
                                              double red_factor, double r_radius, double rerr1,
                                              double deltat, double varphimin, double zeta_p) {
    return ask("nonlinearSys.optimalDeltatAdaptive", {{"p", kPrecision},
                                                      {"decrFactor", alg.decr_factor},
                                                      {"redFactor", red_factor},
                                                      {"rR", r_radius},
                                                      {"rerr1", rerr1},
                                                      {"deltat", deltat},
                                                      {"varphimin", varphimin},
                                                      {"zetaP", zeta_p}})
        .get<double>("value");
}

// the gain over a change of the step size from prev to curr
double NonlinearSys::varphi_total_adaptive(const alg::LinearizationAdaptive &alg, double varphi,
                                           double varphi_lim, double prev, double curr) {
    return ask("nonlinearSys.varphiTotalAdaptive", {{"p", kPrecision},
                                                    {"varphi", varphi},
                                                    {"varphiLim", varphi_lim},
                                                    {"prev", prev},
                                                    {"curr", curr},
                                                    {"decrFactor", alg.decr_factor}})
        .get<double>("value");
}

// the gain that explains the errors of two step sizes
double NonlinearSys::varphi_est_adaptive(const alg::LinearizationAdaptive &alg,
                                         const cora::Tensor &abstrerr_h,
                                         const cora::Tensor &abstrerr, double horizon,
                                         double deltat, const std::vector<bool> &mask,
                                         int minorder) {
    return ask("nonlinearSys.varphiEstAdaptive", {{"n", abstrerr.numel()},
                                                  {"p", kPrecision},
                                                  {"horizon", horizon},
                                                  {"deltat", deltat},
                                                  {"abstrerrH", Tensor::from(abstrerr_h)},
                                                  {"abstrerr", Tensor::from(abstrerr)},
                                                  {"decrFactor", alg.decr_factor},
                                                  {"mask", aux_flags(mask)},
                                                  {"minorder", minorder},
                                                  {"fuel", alg.fuel}})
        .get<double>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
