// linear_sys - lean::LinearSys, the reachability of x' = A x + B u computed by CORALean
//
// See also: lean/zonotope.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/global/oracle/linear_sys.h"
#include "cora/global/error.h"

#include "cora/global/oracle/request.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The fields of a zonotope verification: x0, G0, the input (with B), C (the identity without one)
/// and the specifications {safe, A, b}.
std::vector<Operand> aux_zonotope_fields(const Tensor &A, const std::optional<Tensor> &B,
                                         const std::optional<Tensor> &C, const cora::Tensor &x0,
                                         const cora::Tensor &G0, const cora::Tensor &uc,
                                         const cora::Tensor &GU,
                                         const std::vector<VerifyZonotopeSpec> &specs) {
    const int64_t n = A.rows();
    std::vector<double> eye(std::size_t(n) * n, 0.0);
    for (int64_t i = 0; i < n; ++i) eye[std::size_t(i) * n + i] = 1.0;
    // the system, the initial set and the input
    std::vector<Operand> fields{{"A", A}, {"x0", Tensor(x0)}, {"G0", Tensor(G0)}};
    if (B) {
        fields.push_back({"B", *B});
        fields.push_back({"uc", Tensor(uc)});
        fields.push_back({"GU", Tensor(GU)});
    }
    fields.push_back({"C", C ? *C : Tensor(cora::Tensor::like(x0, eye, {n, n}))});
    // the specifications, each {safe, A, b}
    Json js = Json::array();
    for (const VerifyZonotopeSpec &S : specs)
        js.push(Json::object()
                    .set("safe", Json(S.safe))
                    .set("A", Tensor(S.A).to_json())
                    .set("b", Tensor(S.b).to_json()));
    fields.push_back({"specs", Wire(js, "")});
    return fields;
}

/// A grid as the wire's array of {timeStep, full, budget}.
Json aux_grid_json(const std::vector<VerifyZonotopeGridStep> &grid) {
    Json jg = Json::array();
    for (const VerifyZonotopeGridStep &g : grid)
        jg.push(Json::object()
                    .set("timeStep", Wire(g.time_step).json)
                    .set("full", Json(g.full))
                    .set("budget", Wire(g.budget).json));
    return jg;
}

/// The optional weight as the fields P, L and a (none: no fields).
void aux_weight_fields(std::vector<Operand> &fields, const std::optional<VerifyWeight> &weight) {
    if (!weight) return;
    fields.push_back({"P", Tensor(weight->P)});
    fields.push_back({"L", Tensor(weight->L)});
    fields.push_back({"a", weight->a});
}

} // namespace


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
                                                double t_final, double min_step, int fuel,
                                                const std::optional<Zonotope> &U) const {
    // the output matrix, the identity without one
    const int64_t n = A_.rows();
    std::vector<double> eye(std::size_t(n) * n, 0.0);
    for (int64_t i = 0; i < n; ++i) eye[std::size_t(i) * n + i] = 1.0;
    const Tensor C = C_ ? *C_ : Tensor(cora::Tensor::like(x0, eye, {n, n}));
    // the verdict, the statistics, and the falsifying step and halfspace if falsified
    std::vector<Operand> fields{
        {"A", A_},        {"x0", Tensor(x0)},  {"G0", Tensor(G0)},    {"Cs", Tensor(Cs)}, {"C", C},
        {"d", Tensor(d)}, {"tFinal", t_final}, {"minStep", min_step}, {"fuel", fuel}};
    // the input: B, the center and the generators of U
    if (B_ && U) {
        fields.push_back({"B", *B_});
        fields.push_back({"uc", Tensor(U->center())});
        fields.push_back({"GU", Tensor(U->generators())});
    }
    const Reply r = ask("linearSys.verifySupportFunc", fields);
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

// The zonotope verification ----------------------------------------------------------------------

std::optional<std::vector<std::optional<VerifyZonotopeStep>>> LinearSys::verify_zonotope_pass(
    const cora::Tensor &x0, const cora::Tensor &G0, const cora::Tensor &uc, const cora::Tensor &GU,
    const std::vector<VerifyZonotopeSpec> &specs, const std::vector<VerifyZonotopeGridStep> &grid,
    const std::optional<VerifyWeight> &weight) const {
    std::vector<Operand> fields = aux_zonotope_fields(A_, B_, C_, x0, G0, uc, GU, specs);
    fields.push_back({"grid", Wire(aux_grid_json(grid), "")});
    aux_weight_fields(fields, weight);
    const Reply r = ask("linearSys.verifyZonotopePass", fields);
    // null steps: a step's enclosures are not computed (an empty grid has no steps either way)
    const std::vector<Reply> steps = r.list("steps");
    if (steps.empty() && !grid.empty()) return std::nullopt;
    // per step: none (propagated only), or its output set and flags
    std::vector<std::optional<VerifyZonotopeStep>> out;
    for (const Reply &s : steps) {
        if (s.is_null()) {
            out.emplace_back(std::nullopt);
            continue;
        }
        const Reply set = s.member("set");
        out.emplace_back(VerifyZonotopeStep{set.get<Tensor>("c"), set.get<Tensor>("r"),
                                            set.get<std::vector<Tensor>>("G"),
                                            s.get<std::vector<bool>>("flags")});
    }
    return out;
}

// The zonotope verification run -------------------------------------------------------------------

std::optional<bool>
LinearSys::verify_zonotope_run(const cora::Tensor &x0, const cora::Tensor &G0,
                               const cora::Tensor &uc, const cora::Tensor &GU,
                               const std::vector<VerifyZonotopeSpec> &specs,
                               const std::vector<std::vector<VerifyZonotopeGridStep>> &passes,
                               const std::vector<VerifyZonotopeWitness> &witnesses, double t_final,
                               const std::optional<VerifyWeight> &weight) const {
    std::vector<Operand> fields = aux_zonotope_fields(A_, B_, C_, x0, G0, uc, GU, specs);
    // the passes' grids and the witnesses
    Json jp = Json::array();
    for (const std::vector<VerifyZonotopeGridStep> &grid : passes) jp.push(aux_grid_json(grid));
    Json jw = Json::array();
    for (const VerifyZonotopeWitness &w : witnesses)
        jw.push(Json::object()
                    .set("spec", Json(w.spec))
                    .set("t", Wire(w.t).json)
                    .set("b", Tensor(w.b).to_json())
                    .set("beta", Tensor(w.beta).to_json()));
    fields.push_back({"passes", Wire(jp, "")});
    fields.push_back({"witnesses", Wire(jw, "")});
    fields.push_back({"tFinal", t_final});
    aux_weight_fields(fields, weight);
    // true, false, or null for undecided
    const Reply r = ask("linearSys.verifyZonotopeRun", fields);
    const Reply v = r.member("verdict");
    if (v.is_null()) return std::nullopt;
    return r.get<bool>("verdict");
}

// The Krylov verification ------------------------------------------------------------------------

VerifyKrylovPass LinearSys::verify_krylov_pass(const cora::Tensor &x0, const cora::Tensor &G0,
                                               const cora::Tensor &uc, const cora::Tensor &GU,
                                               const std::vector<VerifyKrylovDirection> &dirs,
                                               int64_t groups, double dt0, int64_t halvings,
                                               int64_t steps, double eps,
                                               const std::optional<VerifyWeight> &cert) const {
    // the system, the initial set, the input in the state space
    std::vector<Operand> fields{
        {"A", A_}, {"x0", Tensor(x0)}, {"G0", Tensor(G0)}, {"uc", Tensor(uc)}, {"GU", Tensor(GU)}};
    // the directions, each {w, plan, limit, strict, group}
    Json jd = Json::array();
    for (const VerifyKrylovDirection &d : dirs) {
        Json plan = Json::array();
        for (const auto &[k, m] : d.plan) plan.push(Json::array().push(Json(k)).push(Json(m)));
        jd.push(Json::object()
                    .set("w", Tensor(d.w).to_json())
                    .set("plan", plan)
                    .set("limit", Wire(d.limit).json)
                    .set("strict", Json(d.strict))
                    .set("group", Json(d.group)));
    }
    fields.push_back({"dirs", Wire(jd, "")});
    fields.push_back({"groups", groups});
    fields.push_back({"timeStep", dt0});
    fields.push_back({"halvings", halvings});
    fields.push_back({"steps", steps});
    fields.push_back({"eps", eps});
    aux_weight_fields(fields, cert);
    const Reply r = ask("linearSys.verifyKrylovPass", fields);
    // the verdict, then per direction its windows and bounds (null: not computed)
    VerifyKrylovPass out;
    out.verified = r.get<bool>("verified");
    for (const Reply &d : r.list("dirs")) {
        VerifyKrylovDirectionResult res;
        if (!d.member("windows").is_null()) {
            std::vector<VerifyKrylovDirectionResult::Window> ws;
            for (const Reply &w : d.list("windows"))
                ws.push_back({w.get<int64_t>("order"), w.get<double>("nu"), w.get<double>("err")});
            res.windows = std::move(ws);
        }
        if (!d.member("bounds").is_null()) res.bounds = d.get<std::vector<double>>("bounds");
        out.dirs.push_back(std::move(res));
    }
    return out;
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
