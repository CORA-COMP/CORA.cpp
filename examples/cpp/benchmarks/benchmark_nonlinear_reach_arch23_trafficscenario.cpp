// benchmark_nonlinear_reach_arch23_trafficscenario - the traffic scenario (TRAF22) of ARCH 2023
//
// MATLAB CORA's benchmark_nonlinear_reach_ARCH23_trafficscenario.m on the controls of
// BEL_Putte-4_2_T-1_controls.csv (copied to examples/benchmarks/data): a kinematic car (steering
// angle, heading, velocity, x, y) tracks a reference car under the feedback u = u_ref + K (x -
// x_ref + v), with disturbances w and measurement errors v as inputs. Each row of the file holds
// u_ref and K of one segment of 0.1 s, reached by alg "lin" in 20 steps from the end of the
// segment before (CORA's nonlinParamSys with numeric parameters, so a NonlinearSys per segment).
// The verdict: the applied inputs stay in [-0.7, 0.7] x [-11, 11]. The time includes the
// occupancy sets of the script (the car's body over each set), as CORA's tComp does.
//
// Syntax:   build/core/benchmarks/benchmark_nonlinear_reach_arch23_trafficscenario [TRAF22]
// Outputs:  one line 'benchmark,instance,result,time', as the MATLAB script
// See also: benchmark_run.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <chrono>
#include <fstream>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::matrix;

/// The solution file: the start time and state, then per segment its end time, u_ref and K (2, 5).
struct Controls {
    std::vector<double> x0, t;
    std::vector<std::array<double, 2>> u;
    std::vector<std::vector<double>> K; // row by row
};

Controls aux_read_controls(const std::string &path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("trafficscenario: cannot read " + path);
    Controls c;
    std::string line;
    // the fields of a line, separated by ';'
    auto fields = [](const std::string &l) {
        std::vector<double> out;
        std::stringstream in(l);
        for (std::string f; std::getline(in, f, ';');) out.push_back(std::stod(f));
        return out;
    };
    std::getline(file, line);
    const std::vector<double> first = fields(line);
    c.t.push_back(first[0]);
    c.x0.assign(first.begin() + 1, first.end());
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const std::vector<double> f = fields(line);
        c.t.push_back(f[0]);
        c.u.push_back({f[1], f[2]});
        c.K.emplace_back(f.begin() + 3, f.begin() + 13);
    }
    return c;
}

/// The open-loop car of the script: steering rate and acceleration u plus the disturbance w.
std::vector<Expr> aux_car(const std::vector<Expr> &x, const std::array<Expr, 2> &w,
                          const std::array<Expr, 2> &u) {
    const double l_wb = 2.578;
    return {u[0] + w[0], (x[2] / l_wb) * tan(x[0]), u[1] + w[1], x[2] * cos(x[1]),
            x[2] * sin(x[1])};
}

/// The closed loop of a segment: x (10) = (car, reference), inputs (7) = (w, v).
NonlinearSys aux_controlled_car(const std::vector<double> &K, const std::array<double, 2> &u_ref) {
    return NonlinearSys(
        [K, u_ref](const std::vector<Expr> &x, const std::vector<Expr> &w) {
            std::array<Expr, 2> u{u_ref[0], u_ref[1]};
            for (std::size_t i = 0; i < 2; ++i)
                for (std::size_t j = 0; j < 5; ++j)
                    u[i] = u[i] + K[i * 5 + j] * (x[j] - x[j + 5] + w[j + 2]);
            std::vector<Expr> dx = aux_car({x[0], x[1], x[2], x[3], x[4]}, {w[0], w[1]}, u);
            const std::vector<Expr> dr =
                aux_car({x[5], x[6], x[7], x[8], x[9]}, {Expr(0.0), Expr(0.0)},
                        {Expr(u_ref[0]), Expr(u_ref[1])});
            dx.insert(dx.end(), dr.begin(), dr.end());
            return dx;
        },
        10, 7);
}

// The occupancy sets ------------------------------------------------------------------------------

/// The map (x, y, phi, z) -> the point z of the car's body in the plane, and its derivatives.
struct Body {
    std::vector<Expr> f, A, Q, T; // f (2), A (2, 5), Q (2, 5, 5), T (2, 5, 5, 5), row-major

    Body() {
        std::vector<Expr> v;
        v.reserve(5);
        for (int i = 0; i < 5; ++i) v.push_back(Expr::var(i));
        f = {v[0] + cos(v[2]) * v[3] - sin(v[2]) * v[4],
             v[1] + cos(v[2]) * v[4] + sin(v[2]) * v[3]};
        for (const Expr &fi : f)
            for (int j = 0; j < 5; ++j) {
                A.push_back(fi.diff(j));
                for (int k = 0; k < 5; ++k) {
                    Q.push_back(A.back().diff(k));
                    for (int l = 0; l < 5; ++l) T.push_back(Q.back().diff(l));
                }
            }
    }
};

/// The values of the expressions at the point p (5 numbers).
std::vector<double> aux_values(const std::vector<Expr> &e, const std::vector<double> &p) {
    std::vector<Tensor> z;
    z.reserve(p.size());
    for (double v : p) z.push_back(column({v}));
    std::vector<double> out;
    for (const Tensor &t : Expr::eval_tensor(e, z)) out.push_back(t.item());
    return out;
}

/// The occupancy of the set S of the car (x, y, phi): f(p) + A (X - p) + 1/2 quadMap(X - p, Q) +
/// the cubic remainder, with X the set times the body; its vertices, as the script computes.
std::vector<Polygon> aux_occupancy(const Body &body, const Zonotope &S, const Zonotope &vehicle) {
    const Zonotope X = S.project({3, 4, 1}).as<Zonotope>().reduce(8).cart_prod(vehicle);
    const std::vector<double> p = X.center().data();
    const Zonotope D = X - X.center();
    const std::vector<double> f = aux_values(body.f, p), A = aux_values(body.A, p);
    const std::vector<double> q = aux_values(body.Q, p);
    std::vector<Tensor> Q = {matrix(5, 5, {q.begin(), q.begin() + 25}),
                             matrix(5, 5, {q.begin() + 25, q.end()})};
    // the remainder 1/6 sum_j d_j (d' T_ij d) over the box d of X - p
    const Interval box = X.interval();
    std::vector<Range> z, d;
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    for (std::size_t j = 0; j < 5; ++j) {
        z.emplace_back(column({lo[j]}), column({hi[j]}));
        d.emplace_back(column({lo[j] - p[j]}), column({hi[j] - p[j]}));
    }
    // the third derivatives over the box, summed against d three times
    const std::vector<Range> T = Expr::enclose(body.T, z);
    std::vector<double> rlo, rhi;
    for (std::size_t i = 0; i < 2; ++i) {
        Range rem(column({0.0}));
        for (std::size_t j = 0; j < 5; ++j) {
            Range quad(column({0.0}));
            for (std::size_t k = 0; k < 5; ++k)
                for (std::size_t l = 0; l < 5; ++l)
                    if (!body.T[((i * 5 + j) * 5 + k) * 5 + l].is_zero())
                        quad = quad + d[k] * T[((i * 5 + j) * 5 + k) * 5 + l] * d[l];
            rem = rem + d[j] * quad;
        }
        rlo.push_back(rem.lo.item() / 6);
        rhi.push_back(rem.hi.item() / 6);
    }
    const Zonotope res = (D.linear_map(matrix(2, 5, A)) + column(f) + D.quad_map(Q) * 0.5)
                             .plus(Interval(column(rlo), column(rhi)));
    return res.vertices();
}

// The instance ------------------------------------------------------------------------------------

/// The hull (lo0, hi0, lo1, hi1) of the inputs u_ref + K (x - x_ref) over every set.
std::array<double, 4> aux_input_hull(const std::vector<std::vector<Zonotope>> &R,
                                     const Controls &c) {
    // K (x - x_ref) over the difference of the two halves of each set
    std::array<double, 2> lo{1e300, 1e300}, hi{-1e300, -1e300};
    for (std::size_t i = 0; i < R.size(); ++i)
        for (const Zonotope &S : R[i]) {
            const Tensor &C = S.center(), &G = S.generators();
            const Zonotope diff(C.slice(-2, 0, 5) - C.slice(-2, 5, 10),
                                G.slice(-2, 0, 5) - G.slice(-2, 5, 10));
            const Interval I =
                (diff.linear_map(matrix(2, 5, c.K[i])) + column({c.u[i][0], c.u[i][1]})).interval();
            const std::vector<double> l = I.inf().data(), h = I.sup().data();
            for (std::size_t k = 0; k < 2; ++k) {
                lo[k] = std::min(lo[k], l[k]);
                hi[k] = std::max(hi[k], h[k]);
            }
        }
    return {lo[0], hi[0], lo[1], hi[1]};
}

/// Reaches, computes the occupancy and decides TRAF22; prints the MATLAB line.
bool aux_run(const std::string &name) {
    // the controls of the scenario
    if (name != "TRAF22")
        CORA_FAIL(Error::wrong_value, "trafficscenario: unknown instance " + name);
    const std::string folder = std::getenv("CORACPP_BENCHMARK_DATA")
                                   ? std::getenv("CORACPP_BENCHMARK_DATA")
                                   : CORACPP_BENCHMARK_DATA;
    const Controls c = aux_read_controls(folder + "/BEL_Putte-4_2_T-1_controls.csv");
    const auto start = std::chrono::steady_clock::now();
    // the start set (car with measurement errors, reference exact) and the inputs (w, v)
    const std::vector<double> v = {0.0004, 0.0004, 0.006, 0.002, 0.002};
    std::vector<double> G0(50, 0.0), x0 = c.x0;
    for (std::size_t i = 0; i < 5; ++i) G0[i * 5 + i] = v[i];
    x0.insert(x0.end(), c.x0.begin(), c.x0.end());
    Zonotope X(column(x0), matrix(10, 5, G0));
    const std::vector<double> u_rad = {0.02, 0.3, 0.0004, 0.0004, 0.006, 0.002, 0.002};
    std::vector<double> GU(49, 0.0);
    for (std::size_t i = 0; i < 7; ++i) GU[i * 7 + i] = u_rad[i];
    const Zonotope U(column(std::vector<double>(7, 0.0)), matrix(7, 7, GU));
    // each segment from the end of the one before
    std::vector<std::vector<Zonotope>> R;
    for (std::size_t i = 0; i < c.u.size(); ++i) {
        const double T = c.t[i + 1] - c.t[i];
        const ReachSet Ri =
            aux_controlled_car(c.K[i], c.u[i])
                .reach({.initial_set = X, .input_set = U, .t_final = T},
                       alg::Linearization{.time_step = T / 20,
                                          .taylor_terms = 10,
                                          .zonotope_order = 20,
                                          .tensor_order = 2,
                                          .linear_step = cora::bench::linear_step()});
        R.push_back(Ri.time_interval);
        X = Ri.time_point.back();
    }
    // the occupancy sets, then the verdict on the inputs
    const Body body;
    const Zonotope vehicle = Interval(column({-4.508 / 2 + 2.578 / 2, -1.610 / 2}),
                                      column({4.508 / 2 + 2.578 / 2, 1.610 / 2}))
                                 .zonotope();
    std::size_t polygons = 0;
    for (const std::vector<Zonotope> &segment : R)
        for (const Zonotope &S : segment) polygons += aux_occupancy(body, S, vehicle).size();
    // the script's contains(U, U_) with U = [-0.7, 0.7] x [-11, 11]
    const std::array<double, 4> h = aux_input_hull(R, c);
    const bool verified = h[0] >= -0.7 && h[1] <= 0.7 && h[2] >= -11 && h[3] <= 11;
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    cora::bench::print_line("Traffic", "TRAF22", verified, wall.count());
    std::cerr << "  " << R.size() << " segments, " << polygons << " occupancy polygons, inputs in ["
              << h[0] << ", " << h[1] << "] x [" << h[2] << ", " << h[3] << "]  (MATLAB CORA: 1)"
              << '\n';
    // with CORACPP_BENCHMARK_VERBOSE: the first time-interval set and the last set, to compare
    if (std::getenv("CORACPP_BENCHMARK_VERBOSE"))
        for (const Zonotope *Z : {&R.front().front(), &X}) {
            const Interval I = Z->interval();
            std::cerr << "    center " << Z->center().matrix_transpose().data().size() << ":";
            for (double x : Z->center().data()) std::cerr << " " << x;
            std::cerr << "\n    radius:";
            for (double x : I.rad().data()) std::cerr << " " << x;
            std::cerr << "\n    generators: " << Z->generators().shape()[1] << '\n';
        }
    return verified;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_family(argc, argv, {"TRAF22"}, aux_run); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
