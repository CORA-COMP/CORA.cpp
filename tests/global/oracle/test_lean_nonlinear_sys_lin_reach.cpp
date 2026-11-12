// test_lean_nonlinear_sys_lin_reach - one step of alg lin against the CORALean oracle, bit for bit
//
// The step of NonlinearSys::init_reach with error_bound = region (the default at tensor order 2) is
// CORALean's linReachFloat: the region the step stays in (regionFloat), the Hessian remainder over
// it joined with the expansion point (hessRemainder), one linear step with f(x*, u*) + B (U - u*)
// + [-l, l] as input (oneStepFloat), with the linear step of CORALean's reachFloatInput (coralean)
// and of the closed form (closed_form). Both sets of the step are the oracle's bit for bit at
// nearest on van der Pol with an input, the Robertson reaction, and the recorded steps of TORA,
// ACC and TRAF22 (data/nonlinear_lin_steps.txt, dumped from CORA.cpp: the sets and inputs that
// init_reach and post received); a step whose region check fails fails in both. Several steps in
// a row (post) are the oracle's reach windows. error_bound = loop is CORA's own loop, which the
// oracle does not have: its boxes are of the oracle's width. TRAF22 is its car (five states, the
// disturbance as input): the closed loop of ten states is too slow for the oracle's trees.
//
// ORACLE: contDynamics/@contDynamics/linReach.m
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

constexpr double kSlack = 1e-9;

using Vars = std::vector<Expr>;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const Vars &x, const Vars &u) {
            return Vars{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
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

/// The box of CORA's loop and the oracle's agree to a factor 2 in every width: the two bound the
/// error differently (neither is always the tighter), both enclose the true set.
void aux_close(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    // the widths of both, coordinate by coordinate
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double wm = c[i] - a[i], wt = d[i] - b[i];
        ok = wm <= 2 * wt + kSlack && wt <= 2 * wm + kSlack;
    }
    test::check(ok, what + ": a box of the width of the oracle's");
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

/// One step of both from R under U: the region step bit for bit, the loop's box inside.
void aux_compare(const std::string &name, const NonlinearSys &sys, const Zonotope &R,
                 const std::optional<Zonotope> &U, double dt, int eta = 4, int rho = 50,
                 bool with_loop = true, bool with_closed_form = true) {
    for (const alg::LinearStep step : {alg::LinearStep::coralean, alg::LinearStep::closed_form}) {
        // the oracle's closed form takes a minute on five states and much more on seven
        if (step == alg::LinearStep::closed_form && !with_closed_form) continue;
        const std::string what = name + " " + aux_name(step);
        const alg::Linearization region = aux_alg(dt, eta, rho, step, alg::ErrorBound::region);
        const std::optional<lean::LinReachResult> theirs =
            lean::NonlinearSys(sys).lin_reach(aux_lean(U), lean::Zonotope(R), region);
        // the region step: exactly the oracle's, or a failure in both
        if (!theirs) {
            test::check(test::throws([&] { sys.init_reach(R, U, region); }),
                        what + ": the step fails as the oracle's does");
            continue;
        }
        const NonlinearStep mine = sys.init_reach(R, U, region);
        oracle::check(what + " time interval", mine.time_interval, theirs->time_interval);
        oracle::check(what + " time point", mine.time_point, theirs->time_point);
        // CORA's loop, which the oracle does not have (not for the recorded steps: slow on lean)
        if (!with_loop) continue;
        const NonlinearStep loop =
            sys.init_reach(R, U, aux_alg(dt, eta, rho, step, alg::ErrorBound::loop));
        aux_close(what + " loop time interval", loop.time_interval, theirs->time_interval);
        aux_close(what + " loop time point", loop.time_point, theirs->time_point);
    }
}

/// Several steps in a row, init_reach then post: the windows of the oracle's reach (the order is
/// large: post reduces the windows too, which the oracle does not).
void aux_chain(const std::string &name, const NonlinearSys &sys, const Zonotope &R,
               const std::optional<Zonotope> &U, double dt, int steps) {
    const alg::Linearization alg =
        aux_alg(dt, 4, 1000, alg::LinearStep::coralean, alg::ErrorBound::region);
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
                    false, closed_form && index == 0);
        ++index;
    }
    test::check(index > 0, "no recorded step of " + name);
}

} // namespace


// ===========================================  MAIN  =========================================== //

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
                    0.001);
        // a time step too large for the region: the step fails in both
        aux_compare("van der Pol, large step", van_der_pol(), vdp_R(), vdp_U(), 3.0);
        // the recorded steps
        aux_recorded("TORA", 6, false, true, [](const Record &) { return tora(); });
        aux_recorded("ACC", 4, false, false, [](const Record &) { return acc(); });
        aux_recorded("TRAF22", 6, true, false,
                     [](const Record &r) { return traf22_car(r.params); });
    });
    oracle::run("nonlinearSys.reach", [&](oracle::Random &) {
        aux_chain("van der Pol", van_der_pol(), vdp_R(), vdp_U(), 0.015625, 5);
    });
    return test::finish("lean nonlinearSys lin reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
