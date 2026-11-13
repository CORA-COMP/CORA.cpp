// test_linprog - the simplex linear program solver: known optima, statuses, random programs
//
// Authors:       Claude Sonnet 5.5
// Written:       30-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/linprog.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using Status = LinProgResult::Status;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The minimum of f'x over {Ax <= b} in the plane by checking every vertex (a bounded program).
double brute_force(const std::vector<double> &f, const std::vector<double> &A,
                   const std::vector<double> &b) {
    const int m = static_cast<int>(b.size());
    double best = 1e300;
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j) {
            const double det = A[2 * i] * A[2 * j + 1] - A[2 * i + 1] * A[2 * j];
            if (std::abs(det) < 1e-12) continue;
            const double x = (b[i] * A[2 * j + 1] - A[2 * i + 1] * b[j]) / det;
            const double y = (A[2 * i] * b[j] - b[i] * A[2 * j]) / det;
            bool ok = true;
            for (int k = 0; k < m; ++k) ok = ok && A[2 * k] * x + A[2 * k + 1] * y <= b[k] + 1e-9;
            if (ok) best = std::min(best, f[0] * x + f[1] * y);
        }
    return best;
}

void known_programs() {
    // max x + y (min -x - y) s.t. x + 2y <= 4, 3x + y <= 6, x, y >= 0: optimum at (1.6, 1.2).
    LinProg lp;
    lp.n = 2;
    lp.f = {-1, -1};
    lp.Aineq = {1, 2, 3, 1};
    lp.bineq = {4, 6};
    lp.lb = {0, 0};
    LinProgResult r = linprog(lp);
    check(r.status == Status::optimal && test::close(r.fval, -2.8, 1e-9), "textbook maximum");
    check(test::close(r.x[0], 1.6) && test::close(r.x[1], 1.2), "textbook maximizer");

    // equality and free variables: min x s.t. x - y = 1, y = 3 (x free) -> x = 4.
    LinProg eq;
    eq.n = 2;
    eq.f = {1, 0};
    eq.Aeq = {1, -1, 0, 1};
    eq.beq = {1, 3};
    r = linprog(eq);
    check(r.status == Status::optimal && test::close(r.fval, 4.0), "equalities, free variables");

    // upper bounds only: min -x s.t. x <= 2 (as a bound), x free below -> -2.
    LinProg ub;
    ub.n = 1;
    ub.f = {-1};
    ub.ub = {2};
    r = linprog(ub);
    check(r.status == Status::optimal && test::close(r.fval, -2.0), "upper bound only");

    // infeasible: x >= 1 and x <= 0.
    LinProg inf;
    inf.n = 1;
    inf.f = {1};
    inf.Aineq = {-1, 1};
    inf.bineq = {-1, 0};
    check(linprog(inf).status == Status::infeasible, "infeasible program");

    // unbounded: min -x, x >= 0.
    LinProg unb;
    unb.n = 1;
    unb.f = {-1};
    unb.lb = {0};
    check(linprog(unb).status == Status::unbounded, "unbounded program");

    // a redundant equality and a degenerate vertex.
    LinProg deg;
    deg.n = 2;
    deg.f = {-1, -1};
    deg.Aineq = {1, 0, 0, 1, 1, 1};
    deg.bineq = {1, 1, 2};
    deg.Aeq = {1, 1, 2, 2};
    deg.beq = {2, 4};
    deg.lb = {0, 0};
    r = linprog(deg);
    check(r.status == Status::optimal && test::close(r.fval, -2.0), "redundant equality");

    check(test::throws([] {
              LinProg bad;
              bad.n = 1;
              bad.f = {1};
              bad.lb = {2};
              bad.ub = {1};
              linprog(bad);
          }),
          "a lower bound above its upper bound is refused");
}

// random planar programs
void random_planar_programs() {
    Rng rng(3);
    for (int trial = 0; trial < 200; ++trial) {
        // A box around the origin plus random cuts: bounded, feasible (the origin is inside).
        const int cuts = 1 + trial % 6;
        std::vector<double> A{1, 0, -1, 0, 0, 1, 0, -1}, b{3, 3, 3, 3};
        for (int c = 0; c < cuts; ++c) {
            std::vector<double> d = test::random_direction(rng, 2);
            A.push_back(d[0]);
            A.push_back(d[1]);
            b.push_back(0.5 + std::abs(d[0]));
        }
        std::vector<double> f = test::random_direction(rng, 2);
        LinProg lp;
        lp.n = 2;
        lp.f = f;
        lp.Aineq = A;
        lp.bineq = b;
        const LinProgResult r = linprog(lp);
        check(r.status == Status::optimal && std::abs(r.fval - brute_force(f, A, b)) < 1e-7,
              "random planar program " + std::to_string(trial));
    }
}

// The multipliers of the optimal basis: known ones, and strong duality on random programs.
void multipliers() {
    // x >= 1 as -x <= -1 (a negative right-hand side: the row is flipped inside): lambda = 1
    LinProg low;
    low.n = 1;
    low.f = {1};
    low.Aineq = {-1};
    low.bineq = {-1};
    LinProgResult r = linprog(low);
    check(r.status == Status::optimal && r.dual_ineq.size() == 1 && r.dual_eq.empty() &&
              test::close(r.dual_ineq[0], 1.0),
          "the multiplier of a row with a negative right-hand side");
    // min x s.t. x - y = 1, y = 3: stationarity in x gives mu_1 = -1, in y mu_2 = mu_1
    LinProg eq;
    eq.n = 2;
    eq.f = {1, 0};
    eq.Aeq = {1, -1, 0, 1};
    eq.beq = {1, 3};
    r = linprog(eq);
    check(r.status == Status::optimal && r.dual_eq.size() == 2 && test::close(r.dual_eq[0], -1.0) &&
              test::close(r.dual_eq[1], -1.0),
          "the multipliers of equalities");
    // a row that does not bind has multiplier 0: min x s.t. x <= 5, x >= 0
    LinProg slack;
    slack.n = 1;
    slack.f = {1};
    slack.Aineq = {1};
    slack.bineq = {5};
    slack.lb = {0};
    r = linprog(slack);
    check(r.status == Status::optimal && test::close(r.dual_ineq[0], 0.0), "a slack row");

    // random programs: the dual value -l'b - m'beq - |f + A'l + Aeq'm|_1 is the optimum
    Rng rng(11);
    const auto uniform = [&](double lo, double hi) {
        double x = 0.0;
        rng.uniform(&x, 1, lo, hi);
        return x;
    };
    for (int trial = 0; trial < 100; ++trial) {
        // three variables, some rows and one equality
        const int n = 3, rows = 2 + trial % 3;
        LinProg lp;
        lp.n = n;
        lp.f = test::random_direction(rng, n);
        for (int i = 0; i < rows; ++i) {
            const std::vector<double> a = test::random_direction(rng, n);
            lp.Aineq.insert(lp.Aineq.end(), a.begin(), a.end());
            lp.bineq.push_back(uniform(-0.2, 0.8));
        }
        const std::vector<double> e = test::random_direction(rng, n);
        lp.Aeq = e;
        lp.beq = {uniform(-0.3, 0.3)};
        lp.lb.assign(std::size_t(n), -1.0);
        lp.ub.assign(std::size_t(n), 1.0);
        r = linprog(lp);
        if (r.status != Status::optimal) continue;
        // the dual value over the box [-1, 1]^n
        double dual = 0.0;
        bool signs = true;
        for (int i = 0; i < rows; ++i) {
            dual -= r.dual_ineq[std::size_t(i)] * lp.bineq[std::size_t(i)];
            signs = signs && r.dual_ineq[std::size_t(i)] > -1e-9;
        }
        dual -= r.dual_eq[0] * lp.beq[0];
        // the residual of the stationarity condition, paid by the box
        for (int j = 0; j < n; ++j) {
            double g = lp.f[std::size_t(j)] + r.dual_eq[0] * e[std::size_t(j)];
            for (int i = 0; i < rows; ++i)
                g += r.dual_ineq[std::size_t(i)] *
                     lp.Aineq[std::size_t(i) * std::size_t(n) + std::size_t(j)];
            dual -= std::abs(g);
        }
        check(signs && std::abs(dual - r.fval) < 1e-7, "strong duality " + std::to_string(trial));
    }
}

/// linprog_costs gives, for each cost, the very result of linprog with that cost (bit for bit,
/// the multipliers too), on programs with inequalities, equalities and bounds.
void shared_phase_one() {
    Rng rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const int n = 4 + trial % 5, rows = 3 + trial % 4;
        LinProg lp;
        lp.n = n;
        // random rows, equalities and the unit box
        lp.Aineq.assign(std::size_t(rows) * std::size_t(n), 0.0);
        lp.bineq.assign(std::size_t(rows), 0.0);
        for (double &x : lp.Aineq) rng.uniform(&x, 1, -0.5, 0.5);
        for (double &x : lp.bineq) rng.uniform(&x, 1, 0.5, 1.5);
        lp.Aeq.assign(2 * std::size_t(n), 0.0);
        for (double &x : lp.Aeq) rng.uniform(&x, 1, -0.5, 0.5);
        lp.beq = {0.1, -0.05};
        lp.lb.assign(std::size_t(n), -1.0);
        lp.ub.assign(std::size_t(n), 1.0);
        std::vector<std::vector<double>> costs(4, std::vector<double>(std::size_t(n)));
        for (auto &f : costs)
            for (double &x : f) rng.uniform(&x, 1, -0.5, 0.5);
        // together and one by one
        const std::vector<LinProgResult> together = linprog_costs(lp, costs);
        bool same = together.size() == costs.size();
        for (std::size_t k = 0; same && k < costs.size(); ++k) {
            lp.f = costs[k];
            const LinProgResult one = linprog(lp);
            same = one.status == together[k].status && one.x == together[k].x &&
                   one.dual_ineq == together[k].dual_ineq && one.dual_eq == together[k].dual_eq &&
                   (one.fval == together[k].fval ||
                    (std::isnan(one.fval) && std::isnan(together[k].fval)));
        }
        check(same, "costs together equal one by one " + std::to_string(trial));
    }
    // an infeasible program is infeasible for every cost
    LinProg none;
    none.n = 1;
    none.Aeq = {1.0};
    none.beq = {5.0};
    none.lb = {0.0};
    none.ub = {1.0};
    for (const LinProgResult &r : linprog_costs(none, {{1.0}, {-1.0}}))
        check(r.status == Status::infeasible, "infeasible for every cost");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    known_programs();
    random_planar_programs();
    multipliers();
    shared_phase_one();
    return test::finish("linprog");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
