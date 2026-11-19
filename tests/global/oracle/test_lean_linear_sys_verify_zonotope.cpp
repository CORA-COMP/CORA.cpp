// test_lean_linear_sys_verify_zonotope - verify's sound zonotope algorithm against CORALean
//
// The zonotope algorithm runs CORA's tuning (priv_reach_adaptive) for the grids and CORALean's
// LinearSys.verifyZonotopePassFloat on each: e^{A dt} as a ball, the carried zonotope and e^{A t}
// with scaled errors, the input's orders reduced by CORA's budget, a full step's output set
// checked by Farkas certificates; a violation is a witness trajectory. Here against the oracle,
// bit for bit at nearest: the passes on fixed grids (every step's output set and flags; steps
// propagated only; budgets that keep and that box the input's generators; a step too long for
// the input's series), the witnesses of violations and of none, and verify on systems of two to
// six states (verified, falsified, a safe and an unsafe set, inputs, an output matrix): the
// verdict of LinearSys.verifyZonotopeRunFloat on the grids and witnesses CORA.cpp used, and the
// steps of each of its passes.
//
// ORACLE: contDynamics/@linearSys/verify.m
// ORACLE: contDynamics/@linearSys/private/priv_verifyRA_zonotope.m
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cont_dynamics/linear_sys/private/priv_verify_zonotope_pass.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"
#include "spec_testing.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A problem: x' = A x + B u, y = C x, R0, U, and the specifications as rows.
struct Problem {
    Tensor A, B, C;
    Zonotope R0, U;
    std::vector<VerifySpecSet> specs;
};

/// The host system of the sound pass.
VerifyZonoSystem aux_system(const Problem &P) {
    VerifyZonoSystem S;
    S.A = P.A;
    S.n = P.A.shape()[0];
    S.x0 = P.R0.center().data();
    S.m = P.R0.nr_generators();
    S.G0 = P.R0.generators().matrix_transpose().data();
    // the input and the outputs
    S.k = P.B.shape()[1];
    S.B = P.B.data();
    S.uc = P.U.center().data();
    S.p = P.U.nr_generators();
    S.GU = P.U.generators().matrix_transpose().data();
    S.q = P.C.shape()[0];
    S.C = P.C.data();
    return S;
}

/// The oracle's system and the operands of a request.
struct Lean {
    lean::LinearSys sys;
    Tensor x0, G0, uc, GU;
    std::vector<lean::VerifyZonotopeSpec> specs;
};
Lean aux_lean(const Problem &P) {
    const VerifyZonoSystem S = aux_system(P);
    Lean L{lean::LinearSys(P.A, P.B, P.C),      Tensor::from_data(S.x0, {S.n, 1}),
           Tensor::from_data(S.G0, {S.n, S.m}), Tensor::from_data(S.uc, {S.k, 1}),
           Tensor::from_data(S.GU, {S.k, S.p}), {}};
    // the specifications as row matrices
    for (const VerifySpecSet &spec : P.specs) {
        std::vector<double> a;
        for (const std::vector<double> &row : spec.a) a.insert(a.end(), row.begin(), row.end());
        const auto r = int64_t(spec.b.size());
        L.specs.push_back(
            {spec.safe, Tensor::from_data(a, {r, S.q}), Tensor::from_data(spec.b, {r, 1})});
    }
    return L;
}

std::vector<lean::VerifyZonotopeGridStep> aux_lean_grid(const std::vector<VerifyGridStep> &grid) {
    std::vector<lean::VerifyZonotopeGridStep> out;
    out.reserve(grid.size());
    for (const VerifyGridStep &g : grid) out.push_back({g.dt, g.full, g.budget});
    return out;
}

/// The pass on the grid, CORA.cpp's against the oracle's, step by step.
void aux_pass(const std::string &name, const Problem &P, const std::vector<VerifyGridStep> &grid) {
    VerifyZonotopePass pass(aux_system(P));
    const auto mine = pass.run(P.specs, grid);
    // the oracle on the same grid
    const Lean L = aux_lean(P);
    const auto theirs =
        L.sys.verify_zonotope_pass(L.x0, L.G0, L.uc, L.GU, L.specs, aux_lean_grid(grid));
    test::check(mine.has_value() == theirs.has_value(), name + ": computed by both or neither");
    if (!mine || !theirs) return;
    test::check(mine->size() == theirs->size(), name + ": the same steps");
    // each full step: its output set, bit for bit, and its flags
    for (std::size_t j = 0; j < mine->size() && j < theirs->size(); ++j) {
        const std::string at = name + " step " + std::to_string(j);
        const auto &a = (*mine)[j];
        const auto &b = (*theirs)[j];
        test::check(a.has_value() == b.has_value(), at + ": full in both or neither");
        if (!a || !b) continue;
        const auto q = int64_t(a->Y.c.size());
        test::oracle::check(at + " c", Tensor::from_data(a->Y.c, {q, 1}), b->c);
        test::oracle::check(at + " r", Tensor::from_data(a->Y.r, {q, 1}), b->r);
        test::check(a->Y.G.size() == b->G.size(), at + ": the same generators");
        for (std::size_t g = 0; g < a->Y.G.size() && g < b->G.size(); ++g)
            test::oracle::check(at + " G" + std::to_string(g), Tensor::from_data(a->Y.G[g], {q, 1}),
                                b->G[g]);
        for (std::size_t s = 0; s < a->flags.size() && s < b->flags.size(); ++s)
            test::oracle::check(at + " flag " + std::to_string(s), bool(a->flags[s]),
                                bool(b->flags[s]));
    }
}

/// A witness, CORA.cpp's check against the oracle's (a run without passes falsifies iff it holds).
void aux_witness(const std::string &name, const Problem &P, std::size_t spec, double t,
                 const std::vector<double> &b, const std::vector<double> &beta, double t_final) {
    VerifyZonotopePass pass(aux_system(P));
    const bool mine = pass.witness(P.specs[spec], t, b, beta);
    const Lean L = aux_lean(P);
    const lean::VerifyZonotopeWitness w{int64_t(spec), t,
                                        Tensor::from_data(b, {int64_t(b.size()), 1}),
                                        Tensor::from_data(beta, {int64_t(beta.size()), 1})};
    const std::optional<bool> theirs =
        L.sys.verify_zonotope_run(L.x0, L.G0, L.uc, L.GU, L.specs, {}, {w}, t_final);
    test::oracle::check(name + " witness", mine, theirs.has_value() && !*theirs);
}

/// The specification of the rows, as verify takes it.
Specification aux_spec(const VerifySpecSet &S) {
    std::vector<std::pair<Tensor, double>> rows;
    rows.reserve(S.b.size());
    for (std::size_t r = 0; r < S.b.size(); ++r)
        rows.emplace_back(Tensor::from_data(S.a[r], {int64_t(S.a[r].size()), 1}), S.b[r]);
    const Polytope H = test::polytope(rows);
    return S.safe ? Specification::safe_set(H) : Specification::unsafe_set(H);
}

/// verify by the zonotope algorithm, its verdict against the oracle's run on the grids and the
/// witnesses it used, and each of its passes step by step.
void aux_verify(const std::string &name, const Problem &P, double t_final) {
    VerifyZonotopeTrace trace;
    // CORA's tuning takes expm, which lean has not: the run on eigen, its passes compared below
    VerifyResult mine;
    {
        const BackendScope tuning("eigen");
        auto copy = [](const Tensor &x) { return Tensor::from_data(x.data(), x.shape()); };
        const LinearSys sys(copy(P.A), copy(P.B), copy(P.C));
        const Zonotope R0(copy(P.R0.center()), copy(P.R0.generators()));
        const Zonotope U(copy(P.U.center()), copy(P.U.generators()));
        std::vector<Specification> eigen_specs;
        eigen_specs.reserve(P.specs.size());
        for (const VerifySpecSet &S : P.specs) eigen_specs.push_back(aux_spec(S));
        mine = priv_verify_ra_zonotope(sys, R0, U, t_final, eigen_specs, &trace);
    }
    // the oracle's run on the same grids and witnesses
    const Lean L = aux_lean(P);
    std::vector<std::vector<lean::VerifyZonotopeGridStep>> passes;
    passes.reserve(trace.passes.size());
    for (const std::vector<VerifyGridStep> &grid : trace.passes)
        passes.push_back(aux_lean_grid(grid));
    std::vector<lean::VerifyZonotopeWitness> witnesses;
    witnesses.reserve(trace.witnesses.size());
    for (const VerifyZonoWitness &w : trace.witnesses)
        witnesses.push_back({int64_t(w.spec), w.t, Tensor::from_data(w.b, {int64_t(w.b.size()), 1}),
                             Tensor::from_data(w.beta, {int64_t(w.beta.size()), 1})});
    const std::optional<bool> theirs =
        L.sys.verify_zonotope_run(L.x0, L.G0, L.uc, L.GU, L.specs, passes, witnesses, t_final);
    test::oracle::check(name + " verified", mine.verified, theirs.value_or(false));
    test::oracle::check(name + " falsified", mine.falsification.has_value(),
                        theirs.has_value() && !*theirs);
    for (std::size_t i = 0; i < trace.passes.size(); ++i)
        aux_pass(name + " pass " + std::to_string(i), P, trace.passes[i]);
}

/// A box about c of radius r in every coordinate.
Zonotope aux_box(const std::vector<double> &c, double r) {
    const auto n = int64_t(c.size());
    std::vector<double> G(std::size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) G[std::size_t(i * n + i)] = r;
    return {Tensor::from_data(c, {n}), Tensor::from_data(G, {n, n})};
}

/// The rows of the box [lo, hi] of the outputs (an unsafe set) or of one halfspace a'y <= b.
VerifySpecSet aux_unsafe_box(const std::vector<double> &lo, const std::vector<double> &hi) {
    VerifySpecSet S;
    S.safe = false;
    for (std::size_t i = 0; i < lo.size(); ++i) {
        std::vector<double> e(lo.size(), 0.0);
        e[i] = 1.0;
        S.a.push_back(e);
        S.b.push_back(hi[i]);
        e[i] = -1.0;
        S.a.push_back(e);
        S.b.push_back(-lo[i]);
    }
    return S;
}

/// A damped rotation of two states pushed by one input (RAND01's form), y = x.
Problem aux_rotation(const std::vector<VerifySpecSet> &specs) {
    return {Tensor({{-0.5, 2.0}, {-2.0, -0.5}}),
            Tensor({{1.0}, {0.5}}),
            Tensor({{1.0, 0.0}, {0.0, 1.0}}),
            aux_box({1.0, 0.0}, 0.1),
            Zonotope(Tensor::from_data({0.2}, {1}), Tensor::from_data({0.1}, {1, 1})),
            specs};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.verifyZonotopePass", [](test::oracle::Random &rnd) {
        // the rotation against an unsafe box it misses, a safe halfspace, and one it enters
        const VerifySpecSet missed = aux_unsafe_box({1.5, 1.5}, {2.0, 2.0});
        VerifySpecSet half;
        half.a = {{1.0, 0.0}};
        half.b = {1.2};
        const VerifySpecSet entered = aux_unsafe_box({0.0, -0.85}, {0.35, -0.6});
        const Problem P = aux_rotation({missed, half, entered});
        std::vector<VerifyGridStep> grid(20, {0.05, true, 0.0});
        aux_pass("uniform", P, grid);
        // propagated steps, a large budget that boxes the input's generators, varying sizes
        for (std::size_t j = 0; j < grid.size(); ++j) {
            grid[j].full = j % 3 != 1;
            grid[j].budget = j < 10 ? 1.0 : 0.0;
            grid[j].dt = 0.02 + 0.01 * double(j % 4);
        }
        aux_pass("mixed", P, grid);
        // a step too long for the input's series: no enclosure
        aux_pass("too long", P, {{0.05, true, 0.0}, {200.0, true, 0.0}});
        // six states, five inputs, two outputs, random stable dynamics
        const Tensor A6 = rnd.matrix(6, 6) - Tensor::eye(6) * 3.0;
        const Tensor B6 = rnd.matrix(6, 5);
        const Problem Q{
            A6,
            B6,
            rnd.matrix(2, 6),
            aux_box({1.0, 0.5, 0.0, -0.5, 0.2, 0.1}, 0.05),
            Zonotope(Tensor::from_data({0.1, 0.0, -0.1, 0.0, 0.05}, {5}), Tensor::eye(5) * 0.02),
            {aux_unsafe_box({2.0, 2.0}, {3.0, 3.0}), half}};
        std::vector<VerifyGridStep> g6(12, {0.04, true, 0.05});
        g6[3].full = false;
        aux_pass("six states", Q, g6);
    });
    test::oracle::run("linearSys.verifyZonotopeRun", [](test::oracle::Random &) {
        const VerifySpecSet entered = aux_unsafe_box({0.0, -0.85}, {0.35, -0.6});
        VerifySpecSet half;
        half.a = {{1.0, 0.0}};
        half.b = {0.9};
        const Problem P = aux_rotation({entered, half});
        // the center's trajectory crosses the box near t = 0.75, the start leaves the halfspace
        aux_witness("inside", P, 0, 0.75, {0.0, 0.0}, {0.0}, 3.0);
        aux_witness("outside", P, 0, 0.3, {1.0, 1.0}, {1.0}, 3.0);
        aux_witness("safe violated", P, 1, 0.01, {1.0, 1.0}, {1.0}, 3.0);
        aux_witness("safe kept", P, 1, 0.01, {-1.0, -1.0}, {-1.0}, 3.0);
        aux_witness("coefficient above 1", P, 1, 0.01, {1.5, 1.0}, {1.0}, 3.0);
    });
    test::oracle::run("linearSys.verifyZonotopeRun", [](test::oracle::Random &rnd) {
        // the rotation: an unsafe box it misses (verified), one it enters (falsified)
        const VerifySpecSet missed = aux_unsafe_box({1.5, 1.5}, {2.0, 2.0});
        const VerifySpecSet entered = aux_unsafe_box({0.0, -0.85}, {0.35, -0.6});
        VerifySpecSet half;
        half.a = {{1.0, 0.0}};
        half.b = {1.25};
        aux_verify("verified", aux_rotation({missed, half}), 3.0);
        aux_verify("falsified", aux_rotation({entered}), 3.0);
        VerifySpecSet tight = half;
        tight.b = {1.05};
        aux_verify("safe falsified", aux_rotation({tight}), 3.0);
        // four states, two inputs, one output row
        const VerifySpecSet far = aux_unsafe_box({3.0, 3.0}, {4.0, 4.0});
        const Tensor A4 = rnd.matrix(4, 4) - Tensor::eye(4) * 2.5;
        const Tensor B4 = rnd.matrix(4, 2);
        const Problem Q{A4,
                        B4,
                        rnd.matrix(2, 4),
                        aux_box({1.0, 0.5, 0.0, -0.5}, 0.05),
                        Zonotope(Tensor::from_data({0.1, -0.1}, {2}), Tensor::eye(2) * 0.05),
                        {far}};
        aux_verify("four states", Q, 2.0);
    });
    return test::finish("lean linearSys verify zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
