// test_linprog - the simplex linear program solver: known optima, statuses, random programs

#include "global/linprog.h"
#include "global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using Status = LinProgResult::Status;

namespace {

/// The minimum of f'x over {Ax <= b} in the plane by checking every vertex (a bounded program).
double bruteForce(const std::vector<double> &f, const std::vector<double> &A,
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
    check(r.status == Status::Optimal && test::close(r.fval, -2.8, 1e-9), "textbook maximum");
    check(test::close(r.x[0], 1.6) && test::close(r.x[1], 1.2), "textbook maximizer");

    // equality and free variables: min x s.t. x - y = 1, y = 3 (x free) -> x = 4.
    LinProg eq;
    eq.n = 2;
    eq.f = {1, 0};
    eq.Aeq = {1, -1, 0, 1};
    eq.beq = {1, 3};
    r = linprog(eq);
    check(r.status == Status::Optimal && test::close(r.fval, 4.0), "equalities, free variables");

    // upper bounds only: min -x s.t. x <= 2 (as a bound), x free below -> -2.
    LinProg ub;
    ub.n = 1;
    ub.f = {-1};
    ub.ub = {2};
    r = linprog(ub);
    check(r.status == Status::Optimal && test::close(r.fval, -2.0), "upper bound only");

    // infeasible: x >= 1 and x <= 0.
    LinProg inf;
    inf.n = 1;
    inf.f = {1};
    inf.Aineq = {-1, 1};
    inf.bineq = {-1, 0};
    check(linprog(inf).status == Status::Infeasible, "infeasible program");

    // unbounded: min -x, x >= 0.
    LinProg unb;
    unb.n = 1;
    unb.f = {-1};
    unb.lb = {0};
    check(linprog(unb).status == Status::Unbounded, "unbounded program");

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
    check(r.status == Status::Optimal && test::close(r.fval, -2.0), "redundant equality");

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
        check(r.status == Status::Optimal &&
                  std::abs(r.fval - bruteForce(f, A, b)) < 1e-7,
              "random planar program " + std::to_string(trial));
    }
}

} // namespace

int main() {
    known_programs();
    random_planar_programs();
    return test::finish("linprog");
}
