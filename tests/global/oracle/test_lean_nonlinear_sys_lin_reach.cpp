// test_lean_nonlinear_sys_lin_reach - one step of alg lin against the CORALean oracle, bit for bit
//
// The step of NonlinearSys::init_reach with error_bound = region (the default at tensor order 2) is
// CORALean's linReachFloat: the region the step stays in (regionFloat), the Hessian remainder over
// it joined with the expansion point (hessRemainder), one linear step with f(x*, u*) + B (U - u*)
// + [-l, l] as input (oneStepFloat), with the linear step of CORALean's reachFloatInput (coralean)
// and of the closed form (closed_form). Both sets of the step are the oracle's bit for bit at
// nearest on van der Pol with an input, the Robertson reaction, and the recorded steps of TORA,
// ACC and TRAF22's car (data/nonlinear_lin_steps.txt: the sets that init_reach and post received);
// a step whose region check fails fails in both. Several steps in a row (post) are the oracle's
// reach windows. error_bound = loop is CORALean's linReachLoopFloat, bit for bit in the sets and
// the error the next step starts from (also from an error of the step before, and on x' = x^2).
// Short horizon: the oracle's closed form takes minutes on three states, so it runs on van der Pol
// only; of the recorded steps three of TORA, two of ACC and one of TRAF22 (loop: 20 s) run.
//
// ORACLE: contDynamics/@contDynamics/linReach.m
// ORACLE: contDynamics/@contDynamics/reach.m
// ORACLE: contDynamics/@nonlinearSys/linearize.m
// ORACLE: contDynamics/@nonlinearSys/nonlinearSys.m
// ORACLE: contDynamics/@nonlinearSys/initReach.m
// ORACLE: contDynamics/@nonlinearSys/post.m
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// the systems, then the helpers that compare a step with the oracle's
using Vars = std::vector<Expr>;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const Vars &x, const Vars &u) {
            return Vars{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

NonlinearSys square() {
    return NonlinearSys([](const Vars &x, const Vars &) { return Vars{pow(x[0], 2)}; }, 1, 0);
}

NonlinearSys robertson() {
    return NonlinearSys(
        [](const Vars &x, const Vars &) {
            return Vars{-0.4 * x[0] + 100 * x[1] * x[2],
                        0.4 * x[0] - 100 * x[1] * x[2] - 1000 * pow(x[1], 2), 1000 * pow(x[1], 2)};
        },
        3, 1);
}

/// The closed loop of TORA (the benchmark's plant, the controlled input a state of derivative 0).
NonlinearSys tora() {
    return NonlinearSys(
        [](const Vars &x, const Vars &) {
            return Vars{x[1], -1.0 * x[0] + 0.1 * sin(x[2]), x[3], 1.0 * x[4] - 10.0, Expr(0.0)};
        },
        5, 0);
}

/// The closed loop of ACC (the benchmark's plant with the controlled acceleration a state).
NonlinearSys acc() {
    constexpr double u_f = 0.0001, a_lead = -2;
    return NonlinearSys(
        [=](const Vars &x, const Vars &) {
            return Vars{x[1],     x[2], -2.0 * x[2] + 2.0 * a_lead - u_f * pow(x[1], 2),
                        x[4],     x[5], -2.0 * x[5] + 2.0 * x[6] - u_f * pow(x[3], 2),
                        Expr(0.0)};
        },
        7, 0);
}

/// The open-loop car of TRAF22: steering rate and acceleration u plus the disturbance w.
Vars traf_car(const Vars &x, const std::array<Expr, 2> &w, const std::array<Expr, 2> &u) {
    const double l_wb = 2.578;
    return {u[0] + w[0], (x[2] / l_wb) * tan(x[0]), u[1] + w[1], x[2] * cos(x[1]),
            x[2] * sin(x[1])};
}

/// The car of a TRAF22 segment at its reference input (the last two parameters); w are the inputs.
NonlinearSys traf22_car(const std::vector<double> &p) {
    return NonlinearSys(
        [p](const Vars &x, const Vars &w) {
            return traf_car(x, {w[0], w[1]}, {Expr(p[10]), Expr(p[11])});
        },
        5, 2);
}

alg::Linearization aux_alg(double dt, int eta, int rho, alg::LinearStep step,
                           alg::ErrorBound bound) {
    alg::Linearization alg;
    alg.time_step = dt;
    alg.taylor_terms = eta;
    alg.zonotope_order = rho;
    alg.linear_step = step;
    alg.error_bound = bound;
    return alg;
}

const char *aux_name(alg::LinearStep step) {
    return step == alg::LinearStep::closed_form ? "closed form" : "coralean";
}

/// The input set for the oracle.
std::optional<lean::Zonotope> aux_lean(const std::optional<Zonotope> &U) {
    if (!U) return std::nullopt;
    return lean::Zonotope(*U);
}

/// One step of both from R under U, bit for bit: the region step, then CORA's loop from the error 0
/// and from 'start' (the error of the step before, through post; rho large so that post reduces
/// nothing). A step that fails in the oracle fails here too.
void aux_compare(const std::string &name, const NonlinearSys &sys, const Zonotope &R,
                 const std::optional<Zonotope> &U, double dt, int eta = 4, int rho = 50,
                 bool with_loop = true, bool with_closed_form = true,
                 const std::optional<Tensor> &start = std::nullopt) {
    for (const alg::LinearStep step : {alg::LinearStep::coralean, alg::LinearStep::closed_form}) {
        // the oracle's closed form takes a minute on five states and much more on seven
        if (step == alg::LinearStep::closed_form && !with_closed_form) continue;
        const std::string what = name + " " + aux_name(step);
        for (const alg::ErrorBound bound : {alg::ErrorBound::region, alg::ErrorBound::loop}) {
            const bool loop = bound == alg::ErrorBound::loop;
            // the oracle's closed form takes minutes for the passes of a loop on more than 2 states
            if (loop && (!with_loop || (step == alg::LinearStep::closed_form && sys.dim() > 2)))
                continue;
            const std::string kind = what + (loop ? " loop" : "");
            const alg::Linearization alg = aux_alg(dt, eta, rho, step, bound);
            // the loop from the error of the step before: post from R with that error
            const auto mine_step = [&](const std::optional<Tensor> &from) {
                if (!from) return sys.init_reach(R, U, alg);
                return sys.post(NonlinearStep{R, R, *from}, U, alg);
            };
            std::vector<std::optional<Tensor>> starts{std::nullopt};
            if (loop && start) starts.push_back(start);
            for (const std::optional<Tensor> &from : starts) {
                const std::string tag = kind + (from ? " from an error" : "");
                const std::optional<lean::LinReachResult> theirs =
                    lean::NonlinearSys(sys).lin_reach(aux_lean(U), lean::Zonotope(R), alg, from,
                                                      loop);
                // a step the oracle refuses fails here too
                if (!theirs) {
                    test::check(test::throws([&] { mine_step(from); }),
                                tag + ": the step fails as the oracle's does");
                    continue;
                }
                const NonlinearStep mine = mine_step(from);
                oracle::check(tag + " time interval", mine.time_interval, theirs->time_interval);
                oracle::check(tag + " time point", mine.time_point, theirs->time_point);
                if (loop) oracle::check(tag + " error", mine.error, *theirs->abstrerr);
            }
        }
    }
}

/// Several steps in a row, init_reach then post: the windows of the oracle's reach (the order is
/// large: post reduces the windows too, which the oracle does not); the loop carries its error.
void aux_chain(const std::string &name, const NonlinearSys &sys, const Zonotope &R,
               const std::optional<Zonotope> &U, double dt, int steps, alg::ErrorBound bound) {
    const alg::Linearization alg = aux_alg(dt, 4, 1000, alg::LinearStep::coralean, bound);
    const std::optional<std::vector<lean::Zonotope>> theirs =
        lean::NonlinearSys(sys).reach(lean::Zonotope(R), aux_lean(U), dt * steps, alg);
    test::check(theirs.has_value() && int(theirs->size()) == steps, name + ": the oracle's run");
    if (!theirs || int(theirs->size()) != steps) return;
    NonlinearStep step = sys.init_reach(R, U, alg);
    for (int k = 0; k < steps; ++k) {
        oracle::check(name + " window " + std::to_string(k), step.time_interval,
                      (*theirs)[std::size_t(k)]);
        if (k + 1 < steps) step = sys.post(step, U, alg);
    }
    // and the whole run of reach: the same windows, step by step
    const ReachSet run = sys.reach({.initial_set = R, .input_set = U, .t_final = dt * steps}, alg);
    test::check(int(run.time_interval.size()) == steps, name + ": the steps of reach");
    for (int k = 0; k < steps && k < int(run.time_interval.size()); ++k)
        oracle::check(name + " reach window " + std::to_string(k),
                      run.time_interval[std::size_t(k)], (*theirs)[std::size_t(k)]);
}

/// What a recorded step carries besides its sets: the dimensions, the options, the parameters.
struct Record {
    int64_t n = 0, m = 0;
    double dt = 0;
    int eta = 0, rho = 0;
    std::vector<double> params;
};

/// The recorded steps of the section of the file: "S n m N M dt eta rho p" and then the p
/// parameters, the center and the generators of R, the center and the generators of U.
void aux_recorded(const std::string &name, int count, bool car, bool closed_form,
                  const std::function<NonlinearSys(const Record &)> &make) {
    const std::filesystem::path file =
        std::filesystem::path(__FILE__).parent_path() / "data" / "nonlinear_lin_steps.txt";
    std::ifstream in(file);
    test::check(bool(in), "cannot read " + file.string());
    std::string line, section;
    int index = 0;
    const auto read = [&](int64_t how_many) {
        std::vector<double> v(static_cast<std::size_t>(how_many));
        for (double &x : v) in >> x;
        return v;
    };
    // each record of the section: its line, then its numbers
    while (std::getline(in, line)) {
        if (line.rfind("# ", 0) == 0) section = line.substr(2);
        if (line.rfind("S ", 0) != 0 || section != name || index >= count) continue;
        Record r;
        int64_t N = 0, M = 0, p = 0;
        char tag = 0;
        std::istringstream(line) >> tag >> r.n >> r.m >> N >> M >> r.dt >> r.eta >> r.rho >> p;
        r.params = read(p);
        const std::vector<double> c = read(r.n), G = read(r.n * N);
        Zonotope R(Tensor::from_data(c, {r.n, 1}), Tensor::from_data(G, {r.n, N}));
        const std::vector<double> u_center = read(r.m), u_gens = read(r.m * M);
        std::optional<Zonotope> U;
        if (r.m > 0)
            U = Zonotope(Tensor::from_data(u_center, {r.m, 1}),
                         Tensor::from_data(u_gens, {r.m, M}));
        in >> std::ws;
        // the car: the first five states, the first two inputs
        if (car) {
            R = Zonotope(R.center().slice(-2, 0, 5), R.generators().slice(-2, 0, 5));
            U = Zonotope(U->center().slice(-2, 0, 2), U->generators().slice(-2, 0, 2));
        }
        aux_compare(name + " step " + std::to_string(index), make(r), R, U, r.dt, r.eta, r.rho,
                    index == 0, closed_form && index == 0);
        ++index;
    }
    test::check(index > 0, "no recorded step of " + name);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// linearize and the constructor are exercised too: the oracle runs the dynamics as stored.
int main() {
    if (!oracle::available()) return 0;
    // the sets are made inside a pass, on its backend
    const auto vdp_R = [] {
        return Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.01, 0.002}, {0.0, 0.05, 0.003}}));
    };
    const auto vdp_U = [] { return Zonotope(Tensor({0.0}), Tensor({{0.1}})); };
    oracle::run("nonlinearSys.linReach", [&](oracle::Random &) {
        aux_compare("van der Pol", van_der_pol(), vdp_R(), vdp_U(), 0.02);
        const Zonotope rob_R(Tensor({1.0, 0.0, 0.0}),
                             Tensor({{1e-3, 0.0, 0.0}, {0.0, 1e-4, 0.0}, {0.0, 0.0, 1e-4}}));
        aux_compare("Robertson", robertson(), rob_R, Zonotope(Tensor({0.0}), Tensor::zeros({1, 0})),
                    0.001, 4, 50, true, false);
        // a time step too large for the region: the step fails in both
        aux_compare("van der Pol, large step", van_der_pol(), vdp_R(), vdp_U(), 3.0, 4, 50, false);
        // CORA's loop from the error of the step before (rho large: post reduces nothing)
        aux_compare("van der Pol, from an error", van_der_pol(), vdp_R(), vdp_U(), 0.03125, 4, 1000,
                    true, false, Tensor::from_data({0.0, 0x1p-20}, {2, 1}));
        // x' = x^2 at a time step of 1/8: the region check fails, the loop succeeds
        aux_compare("x' = x^2", square(), Zonotope(Tensor({1.5}), Tensor({{0.5}})), std::nullopt,
                    0.125, 6, 10);
        // the recorded steps
        aux_recorded("TORA", 3, false, false, [](const Record &) { return tora(); });
        aux_recorded("ACC", 2, false, false, [](const Record &) { return acc(); });
        aux_recorded("TRAF22", 1, true, false,
                     [](const Record &r) { return traf22_car(r.params); });
    });
    oracle::run("nonlinearSys.reach", [&](oracle::Random &) {
        for (const alg::ErrorBound bound : {alg::ErrorBound::region, alg::ErrorBound::loop})
            aux_chain(bound == alg::ErrorBound::loop ? "van der Pol loop" : "van der Pol",
                      van_der_pol(), vdp_R(), vdp_U(), 0.015625, 5, bound);
    });
    return test::finish("lean nonlinearSys lin reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
