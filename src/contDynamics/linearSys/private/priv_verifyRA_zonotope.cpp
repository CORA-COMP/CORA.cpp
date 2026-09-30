// priv_verifyRA_zonotope - reach-avoid verification of a linear system with zonotopes [1]
//
// An outer approximation of the reachable output sets with a bound on its error to the exact sets
// (priv_reach_adaptive) is checked against the specifications: if it lies inside every safe set
// and misses every unsafe set, the specifications hold; the inner approximation (the outer one
// minus the ball of the error bound) proves a violation if it leaves a safe set or hits an
// unsafe one. Otherwise the error bound shrinks by the distances and the set is computed again.
// Supported as in the ARCH benchmarks: x' = A x + B u, y = C x, a zonotope R0 and a constant
// zonotope U, no offsets; every specification applies over the whole horizon [0, tFinal].
//
// Syntax:   res = priv_verifyRA_zonotope(sys, params, specs);
// Inputs:   sys - linear system;  params - R0, U, tFinal;  specs - safe and unsafe sets
// Outputs:  res - verified, the refinements (iterations), the smallest step size and the
//           number of steps of the last run, tComp
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner- and
//        outer-approximations of reachable sets", TAC, 2023.
// See also: LinearSys::verify, priv_reach_adaptive

#include "contDynamics/linearSys/private/priv_reach_adaptive.h"
#include "contDynamics/linearSys/private/priv_verify.h"
#include "global/linprog.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf =
    std::numeric_limits<double>::infinity();
constexpr double kNaN =
    std::numeric_limits<double>::quiet_NaN();
constexpr double kIntersectTol = 1e-9;  // a linear program distance below this counts as 0

// Sets on the host --------------------------------------------------------------------------------

/// A zonotope c + G [-1,1]^m on the host (G row-major, n x m).
struct ZHost {
    int n = 0, m = 0;
    std::vector<double> c, G;
};

/// A polytope {x | A x <= b} with unit row normals (A row-major, rows x dim), and what is
/// precomputed for specifications: the time it still needs to be verified, whether it is bounded
/// and its interval hull, and whether it is one halfspace perpendicular to an axis.
struct Poly {
    int rows = 0, dim = 0;
    std::vector<double> A, b;
    VerifyTime time;
    bool isBounded = false, fastInner = false;
    std::vector<double> lo, hi;
};

/// An inner approximation: the Minkowski difference of the zonotope Z and the cross polytope of
/// radius rho (the ball of the 1-norm), {x | x + v in Z for every vertex v = +-rho e_i}.
struct InnerSet {
    ZHost Z;
    double rho = 0;
};

ZHost aux_zhost(const Zonotope &Z) {
    ZHost h;
    h.n = static_cast<int>(Z.c.shape()[0]);
    h.m = static_cast<int>(Z.G.shape()[1]);
    h.c = Z.c.data();
    h.G = Z.G.data();
    return h;
}

/// The vertices of the cross polytope of radius rho in q dimensions, flattened (2q x q).
std::vector<std::vector<double>> aux_crossVertices(int q, double rho) {
    std::vector<std::vector<double>> v;
    for (int i = 0; i < q; ++i)
        for (double s : {1.0, -1.0}) {
            std::vector<double> x(q, 0.0);
            x[i] = s * rho;
            v.push_back(x);
        }
    return v;
}

/// Row i of G times the vector of generator factors: sum_j |C_i G_j| is the support value.
double aux_rowAbsSum(const Poly &P, int row, const ZHost &Z) {
    double s = 0;
    for (int j = 0; j < Z.m; ++j) {
        double v = 0;
        for (int k = 0; k < Z.n; ++k) v += P.A[row * P.dim + k] * Z.G[k * Z.m + j];
        s += std::abs(v);
    }
    return s;
}

double aux_rowDot(const Poly &P, int row, const std::vector<double> &x) {
    double v = 0;
    for (int k = 0; k < P.dim; ++k) v += P.A[row * P.dim + k] * x[k];
    return v;
}


// Containment and intersection --------------------------------------------------------------------

/// The distance of a zonotope from being inside the polytope: max(-d + C c + sum |C G|), which
/// is <= 0 iff it is contained.
double aux_containmentCheckZono(const ZHost &Z, const Poly &P) {
    double dist = -kInf;
    for (int i = 0; i < P.rows; ++i)
        dist = std::max(dist, -P.b[i] + aux_rowDot(P, i, Z.c) + aux_rowAbsSum(P, i, Z));
    return dist;
}

/// The constraints that tie the generator factors of the vertex copies of an inner set together:
/// the same point x = c - v_k + G a_k for every vertex v_k. Variable block k of the LP starts at
/// offset(k) and has m entries; the rows are appended to Aeq/beq (nVars wide).
void aux_innerEqualities(
    const InnerSet &R, const std::vector<std::vector<double>> &V, int nVars, const
    std::function<int(int)> &offset, LinProg &lp) {
    const int n = R.Z.n, m = R.Z.m;
    for (std::size_t k = 1; k < V.size(); ++k)
        for (int r = 0; r < n; ++r) {
            std::vector<double> row(nVars, 0.0);
            for (int j = 0; j < m; ++j) {
                row[offset(static_cast<int>(k)) + j] = R.Z.G[r * m + j];
                row[offset(0) + j] = -R.Z.G[r * m + j];
            }
            lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
            lp.beq.push_back(V[k][r] - V[0][r]);
        }
}

/// The distance of the inner set from being inside the polytope: max over the rows of
/// max_{x in R} (C_i x - d_i); NaN if the inner set is empty.
double aux_containmentCheckInner(const InnerSet &R, const Poly &P) {
    std::vector<std::vector<double>> V = aux_crossVertices(R.Z.n, R.rho);
    // Without a ball there is one copy of the zonotope.
    if (R.rho <= 0) V.resize(1);
    const int n = R.Z.n, m = R.Z.m, nV = static_cast<int>(V.size());
    // The point is c - v_0 + G a_0.
    std::vector<double> c0(n);
    for (int i = 0; i < n; ++i) c0[i] = R.Z.c[i] - V[0][i];
    double dist = -kInf;
    for (int i = 0; i < P.rows; ++i) {
        LinProg lp;
        lp.n = nV * m;
        lp.f.assign(lp.n, 0.0);
        for (int j = 0; j < m; ++j) {
            double v = 0;
            for (int k = 0; k < n; ++k) v += P.A[i * P.dim + k] * R.Z.G[k * m + j];
            lp.f[j] = -v;
        }
        lp.lb.assign(lp.n, -1.0);
        lp.ub.assign(lp.n, 1.0);
        aux_innerEqualities(R, V, lp.n, [&](int k) { return k * m; }, lp);
        const LinProgResult r = linprog(lp);
        if (r.status != LinProgResult::Status::Optimal) return kNaN;
        dist = std::max(dist, aux_rowDot(P, i, c0) - P.b[i] - r.fval);
    }
    return dist;
}

/// The 1-norm distance between the inner set and the polytope (0 iff they intersect), by the
/// linear program over the slack a >= |z - x|, the generator factors and the polytope point x;
/// NaN if the inner set is empty.
double aux_intersectionCheckInner(const InnerSet &R, const Poly &P) {
    const std::vector<std::vector<double>> V = aux_crossVertices(R.Z.n, R.rho);
    const int n = R.Z.n, m = R.Z.m;
    const int nV = R.rho > 0 ? static_cast<int>(V.size()) : 1;
    // Variables: slack a (n), generator factors of each vertex copy (nV m), polytope point x (n).
    const int nVars = n + nV * m + n;
    const int xOff = n + nV * m;
    LinProg lp;
    lp.n = nVars;
    lp.f.assign(nVars, 0.0);
    for (int i = 0; i < n; ++i) lp.f[i] = 1.0;
    lp.lb.assign(nVars, -kInf);
    lp.ub.assign(nVars, kInf);
    for (int i = 0; i < n; ++i) lp.lb[i] = 0.0;
    for (int i = n; i < xOff; ++i) lp.lb[i] = -1.0, lp.ub[i] = 1.0;
    // The polytope contains x.
    for (int i = 0; i < P.rows; ++i) {
        std::vector<double> row(nVars, 0.0);
        for (int k = 0; k < n; ++k) row[xOff + k] = P.A[i * P.dim + k];
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
        lp.bineq.push_back(P.b[i]);
    }
    // |c_0 + G a_0 - x| <= a with the point c_0 = c - v_0 of the inner set.
    for (int r = 0; r < n; ++r)
        for (double s : {1.0, -1.0}) {
            std::vector<double> row(nVars, 0.0);
            row[r] = -1.0;
            for (int j = 0; j < m; ++j) row[n + j] = s * R.Z.G[r * m + j];
            row[xOff + r] = -s;
            lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
            lp.bineq.push_back(-s * (R.Z.c[r] - V[0][r]));
        }
    if (nV > 1) {
        const std::vector<std::vector<double>> all = V;
        aux_innerEqualities(R, all, nVars, [&](int k) { return n + k * m; }, lp);
    }
    const LinProgResult r = linprog(lp);
    return r.status == LinProgResult::Status::Optimal ? r.fval : kNaN;
}

/// The interval hull of the intersection of a zonotope and a polytope by 2n linear programs
/// over the generator factors; false if they do not intersect.
bool aux_intersectionHull(
    const ZHost &Z, const Poly &P, std::vector<double> &lo, std::vector<double> &hi) {
    const int n = Z.n, m = Z.m;
    LinProg base;
    base.n = m;
    base.lb.assign(m, -1.0);
    base.ub.assign(m, 1.0);
    for (int i = 0; i < P.rows; ++i) {
        for (int j = 0; j < m; ++j) {
            double v = 0;
            for (int k = 0; k < n; ++k) v += P.A[i * P.dim + k] * Z.G[k * m + j];
            base.Aineq.push_back(v);
        }
        base.bineq.push_back(P.b[i] - aux_rowDot(P, i, Z.c));
    }
    lo.assign(n, 0.0);
    hi.assign(n, 0.0);
    for (int d = 0; d < n; ++d)
        for (double s : {1.0, -1.0}) {
            LinProg lp = base;
            lp.f.assign(m, 0.0);
            for (int j = 0; j < m; ++j) lp.f[j] = s * Z.G[d * m + j];
            const LinProgResult r = linprog(lp);
            if (r.status != LinProgResult::Status::Optimal) return false;
            // min of s x_d: the lower bound for s = 1, minus the upper bound for s = -1.
            (s > 0 ? lo[d] : hi[d]) = s > 0 ? Z.c[d] + r.fval : Z.c[d] - r.fval;
        }
    return true;
}

/// The maximum distance between a zonotope and a polytope that intersect: twice the 2-norm of
/// the radius of the interval hull of their intersection.
double aux_distanceIntersectionZono(const ZHost &Z, const Poly &P) {
    std::vector<double> lo, hi;
    if (!aux_intersectionHull(Z, P, lo, hi)) return kNaN;
    double s = 0;
    for (int i = 0; i < Z.n; ++i) s += 0.25 * (hi[i] - lo[i]) * (hi[i] - lo[i]);
    return 2 * std::sqrt(s);
}

/// A quick hierarchical test whether a zonotope intersects the unsafe set: a separating
/// halfspace, then the interval hulls, then the linear program.
bool aux_intersectionCheckFast(const ZHost &Z, const Poly &F) {
    for (int i = 0; i < F.rows; ++i)
        if (aux_rowDot(F, i, Z.c) - aux_rowAbsSum(F, i, Z) - F.b[i] > 0) return false;
    if (F.isBounded) {
        // The hull of the zonotope must meet the hull of the polytope.
        for (int d = 0; d < Z.n; ++d) {
            double rad = 0;
            for (int j = 0; j < Z.m; ++j) rad += std::abs(Z.G[d * Z.m + j]);
            if (Z.c[d] - rad > F.hi[d] || Z.c[d] + rad < F.lo[d]) return false;
        }
    }
    // The LP distance is zero iff they intersect (up to the solver accuracy).
    const InnerSet Z0{Z, 0.0};
    const double d = aux_intersectionCheckInner(Z0, F);
    return d <= kIntersectTol;
}


// Specifications ----------------------------------------------------------------------------------

/// The polytope of the halfspaces of a specification with unit normals (scaling does not change
/// the set), for outputs of dimension p.
Poly aux_polytope(const std::vector<Halfspace> &halfspaces, int64_t p, double sign) {
    Poly P;
    P.dim = static_cast<int>(p);
    for (const Halfspace &h : halfspaces) {
        const std::vector<int64_t> shape = h.a.shape();
        if (shape.size() != 2 || shape[0] != p || shape[1] != 1)
            throw std::invalid_argument(
                "LinearSys::verify: a halfspace normal must be a column of the output dimension");
        const std::vector<double> a = h.a.data();
        double norm = 0;
        for (double v : a) norm += v * v;
        norm = std::sqrt(norm);
        if (!(norm > 0))
            throw std::invalid_argument("LinearSys::verify: a halfspace normal must not be zero");
        for (double v : a) P.A.push_back(sign * v / norm);
        P.b.push_back(sign * h.b / norm);
        ++P.rows;
    }
    return P;
}

/// The interval hull of the polytope by 2p linear programs; false if it is unbounded.
bool aux_polytopeHull(Poly &P) {
    LinProg lp;
    lp.n = P.dim;
    lp.Aineq = P.A;
    lp.bineq = P.b;
    lp.lb.assign(P.dim, -kInf);
    lp.ub.assign(P.dim, kInf);
    P.lo.assign(P.dim, 0.0);
    P.hi.assign(P.dim, 0.0);
    for (int d = 0; d < P.dim; ++d)
        for (double s : {1.0, -1.0}) {
            lp.f.assign(P.dim, 0.0);
            lp.f[d] = s;
            const LinProgResult r = linprog(lp);
            if (r.status == LinProgResult::Status::Unbounded) return false;
            if (r.status == LinProgResult::Status::Infeasible)
                throw std::invalid_argument("LinearSys::verify: an unsafe set is empty");
            (s > 0 ? P.lo[d] : P.hi[d]) = s * r.fval;
        }
    return true;
}

/// The safe sets and the unsafe sets of the specifications. An unsafe set of one halfspace is
/// turned into the safe set of its complement; the others stay polytopes that must not be
/// intersected. All are to be verified over [0, tFinal].
void aux_getSetsFromSpec(
    const std::vector<Specification> &specs, int64_t p, double tFinal, std::vector<Poly> &safeSets,
    std::vector<Poly> &unsafeSets) {
    if (specs.empty()) throw std::invalid_argument("LinearSys::verify: no specification given");
    for (const Specification &spec : specs) {
        switch (spec.type()) {
        case SpecType::SafeSet:
            safeSets.push_back(aux_polytope(spec.halfspaces(), p, 1.0));
            break;
        case SpecType::UnsafeSet:
            if (spec.halfspaces().size() == 1) {
                safeSets.push_back(aux_polytope(spec.halfspaces(), p, -1.0));
            } else {
                unsafeSets.push_back(aux_polytope(spec.halfspaces(), p, 1.0));
                unsafeSets.back().isBounded = aux_polytopeHull(unsafeSets.back());
            }
            break;
        default:
            throw std::invalid_argument("LinearSys::verify: unknown specification type");
        }
    }
    // Every set is to be verified over the whole horizon; one that is a single halfspace
    // perpendicular to an axis needs no inner approximation (fastInner).
    for (std::vector<Poly> *sets : {&safeSets, &unsafeSets})
        for (Poly &P : *sets) {
            int nnz = 0;
            for (double v : P.A) nnz += v != 0;
            P.fastInner = P.rows == 1 && nnz == 1;
            P.time = VerifyTime({{0.0, tFinal}});
        }
}


// Distances of points -----------------------------------------------------------------------------

/// Whether the point is in the polytope (up to 1e-12).
bool aux_contains(const Poly &P, const std::vector<double> &y) {
    for (int i = 0; i < P.rows; ++i)
        if (aux_rowDot(P, i, y) > P.b[i] + 1e-12) return false;
    return true;
}

/// The 1-norm distance of the point p to the interval [lo, hi], counted negative (as CORA does).
double aux_distanceIntervalPoint(const Poly &P, const std::vector<double> &p) {
    double d = 0;
    for (int i = 0; i < P.dim; ++i) {
        if (p[i] > P.hi[i]) d += P.hi[i] - p[i];
        else if (p[i] < P.lo[i]) d += P.lo[i] - p[i];
    }
    return d;
}

/// The distance of the point p to the polytope: to the halfspace if only one is violated, to
/// the vertex of the violated halfspaces if there are dim of them, else the 1-norm by a program.
double aux_distancePolyPoint(const Poly &P, const std::vector<double> &p) {
    const int n = P.dim;
    std::vector<int> violated;
    std::vector<double> offset(P.rows);
    for (int i = 0; i < P.rows; ++i) {
        offset[i] = aux_rowDot(P, i, p) - P.b[i];
        if (offset[i] > 0) violated.push_back(i);
    }
    if (violated.size() == 1) return offset[violated[0]];
    if (static_cast<int>(violated.size()) == n) {
        // The vertex of the violated halfspaces: solve C_ind v = d_ind by Gauss elimination.
        std::vector<std::vector<double>> M(n, std::vector<double>(n + 1));
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) M[r][c] = P.A[violated[r] * n + c];
            M[r][n] = P.b[violated[r]];
        }
        for (int c = 0; c < n; ++c) {
            int piv = c;
            for (int r = c + 1; r < n; ++r)
                if (std::abs(M[r][c]) > std::abs(M[piv][c])) piv = r;
            std::swap(M[c], M[piv]);
            for (int r = c + 1; r < n; ++r) {
                const double f = M[r][c] / M[c][c];
                for (int k = c; k <= n; ++k) M[r][k] -= f * M[c][k];
            }
        }
        std::vector<double> v(n);
        for (int r = n - 1; r >= 0; --r) {
            double s = M[r][n];
            for (int c = r + 1; c < n; ++c) s -= M[r][c] * v[c];
            v[r] = s / M[r][r];
        }
        double s = 0;
        for (int i = 0; i < n; ++i) s += (v[i] - p[i]) * (v[i] - p[i]);
        return std::sqrt(s);
    }
    // min |p - x|_1 s.t. C x <= d, with x + u - w = p and u, w >= 0.
    LinProg lp;
    lp.n = 3 * n;
    lp.f.assign(lp.n, 0.0);
    for (int i = n; i < 3 * n; ++i) lp.f[i] = 1.0;
    lp.lb.assign(lp.n, 0.0);
    for (int i = 0; i < n; ++i) lp.lb[i] = -kInf;
    lp.ub.assign(lp.n, kInf);
    for (int i = 0; i < P.rows; ++i) {
        std::vector<double> row(lp.n, 0.0);
        for (int k = 0; k < n; ++k) row[k] = P.A[i * n + k];
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
        lp.bineq.push_back(P.b[i]);
    }
    for (int i = 0; i < n; ++i) {
        std::vector<double> row(lp.n, 0.0);
        row[i] = 1.0, row[n + i] = 1.0, row[2 * n + i] = -1.0;
        lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
        lp.beq.push_back(p[i]);
    }
    const LinProgResult r = linprog(lp);
    return r.status == LinProgResult::Status::Optimal ? r.fval : kNaN;
}


// Initial error -----------------------------------------------------------------------------------

// ode45 -------------------------------------------------------------------------------------------

/// The time points and states MATLAB's ode45 returns for x' = A x + b from x0 on [0, tFinal] with
/// its default options (tolerances 1e-3 and 1e-6, refinement 4): the error control of the
/// Dormand-Prince pair decides the steps; every accepted step adds three interpolated points
/// (its dense output) and its end.
void aux_ode45(
    const std::vector<double> &A, int n, const std::vector<double> &b, std::vector<double> y,
    double tFinal, std::vector<double> &times, std::vector<std::vector<double>> &states) {
    static const double c[7] = {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1};
    static const double a[7][6] = {{0, 0, 0, 0, 0, 0},
                                   {1.0 / 5, 0, 0, 0, 0, 0},
                                   {3.0 / 40, 9.0 / 40, 0, 0, 0, 0},
                                   {44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0, 0},
                                   {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561,
                                    -212.0 / 729, 0, 0},
                                   {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176,
                                    -5103.0 / 18656, 0},
                                   {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784,
                                    11.0 / 84}};
    static const double E[7] = {71.0 / 57600, 0, -71.0 / 16695, 71.0 / 1920, -17253.0 / 339200,
                                22.0 / 525, -1.0 / 40};
    const double rtol = 1e-3, threshold = 1e-6 / rtol, pow5 = 1.0 / 5;
    auto f = [&](const std::vector<double> &x) {
        std::vector<double> dx(b);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) dx[i] += A[i * n + j] * x[j];
        return dx;
    };
    // The largest step is a tenth of the horizon.
    const double hmax = 0.1 * tFinal;
    double t = 0, absh = std::min(hmax, tFinal);
    std::vector<std::vector<double>> k(7);
    k[0] = f(y);
    // The initial step size from the size of the derivative.
    double rh = 0;
    for (int i = 0; i < n; ++i)
        rh = std::max(rh, std::abs(k[0][i]) / std::max(std::abs(y[i]), threshold));
    rh /= 0.8 * std::pow(rtol, pow5);
    if (absh * rh > 1) absh = 1 / rh;

    // The coefficients of the dense output (MATLAB's ntrp45).
    static const double BI[7][4] = {{1, -183.0 / 64, 37.0 / 12, -145.0 / 128},
                                    {0, 0, 0, 0},
                                    {0, 1500.0 / 371, -1000.0 / 159, 1000.0 / 371},
                                    {0, -125.0 / 32, 125.0 / 12, -375.0 / 64},
                                    {0, 9477.0 / 3392, -729.0 / 106, 25515.0 / 6784},
                                    {0, -11.0 / 7, 11.0 / 3, -55.0 / 28},
                                    {0, 3.0 / 2, -4, 5.0 / 2}};
    times = {0.0};
    states = {y};
    bool done = false;
    while (!done) {
        const double hmin = 16 * std::nextafter(t, kInf) - 16 * t;
        absh = std::min(hmax, std::max(hmin, absh));
        double h = absh;
        // A step that gets within 10% of the end is stretched to it.
        if (1.1 * absh >= tFinal - t) {
            h = tFinal - t;
            absh = h;
            done = true;
        }
        bool nofailed = true, accept = false;
        double err = 0, tnew = 0;
        std::vector<double> ynew(n);
        while (true) {
            for (int s = 1; s < 7; ++s) {
                std::vector<double> ys(y);
                for (int i = 0; i < n; ++i)
                    for (int q = 0; q < s; ++q) ys[i] += h * a[s][q] * k[q][i];
                if (s == 6) ynew = ys;
                k[s] = f(ys);
            }
            tnew = done ? tFinal : t + h * c[5];
            h = tnew - t;
            // The error estimate: the difference of the fourth and fifth order solutions.
            err = 0;
            for (int i = 0; i < n; ++i) {
                double fe = 0;
                for (int s = 0; s < 7; ++s) fe += k[s][i] * E[s];
                const double wt = std::max(std::max(std::abs(y[i]), std::abs(ynew[i])), threshold);
                err = std::max(err, absh * std::abs(fe) / wt);
            }
            accept = err <= rtol;
            if (accept) break;
            if (absh <= hmin) break;
            if (nofailed) {
                nofailed = false;
                absh = std::max(hmin, absh * std::max(0.1, 0.8 * std::pow(rtol / err, pow5)));
            } else {
                absh = std::max(hmin, 0.5 * absh);
            }
            h = absh;
            done = false;
        }
        // Output: three interpolation points and the end of the step.
        for (int r = 1; r <= 3; ++r) {
            const double s = r / 4.0;
            std::vector<double> yi(y);
            for (int i = 0; i < n; ++i)
                for (int q = 0; q < 7; ++q)
                    yi[i] += h * k[q][i] * (BI[q][0] * s + BI[q][1] * s * s + BI[q][2] * s * s * s +
                                            BI[q][3] * s * s * s * s);
            times.push_back(t + (tnew - t) * r / 4.0);
            states.push_back(yi);
        }
        times.push_back(tnew);
        states.push_back(ynew);
        if (nofailed) {
            const double temp = 1.25 * std::pow(err / rtol, pow5);
            absh = temp > 0.2 ? absh / temp : 5.0 * absh;
        }
        t = tnew;
        y = ynew;
        k[0] = k[6];
    }
}

/// The smallest value of the error scale 10 - 9 t / tEnd over the time points.
double aux_scaleMin(const std::vector<double> &t) {
    double lo = kInf;
    for (double v : t) lo = std::min(lo, 10 - 9 / t.back() * v);
    return lo;
}

/// An estimate of the maximum allowed error from trajectories: from points on the facets of the
/// interval hull of R0, the distance of the output trajectory to the specifications, weighted
/// to be strictest at the end of the horizon. The trajectories are exact solutions sampled at
/// the time points of ode45 (first point) and on an equidistant grid of twice that step size.
double aux_initError(
    const Tensor &A, const std::optional<Tensor> &C, const Tensor &uOffset, const Zonotope &R0,
    const std::vector<Poly> &safeSets, const std::vector<Poly> &unsafeSets, double tFinal) {
    const int n = static_cast<int>(A.shape()[0]);
    const std::vector<double> c = R0.c.data(), Gd = R0.G.data();
    const int m0 = static_cast<int>(R0.G.shape()[1]);
    std::vector<double> rad(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m0; ++j) rad[i] += std::abs(Gd[i * m0 + j]);

    // The points: centers of the facets of the interval hull (the 10 longest edges at most).
    std::vector<int> dims;
    for (int i = 0; i < n; ++i)
        if (rad[i] != 0) dims.push_back(i);
    if (dims.size() > 10) {
        std::stable_sort(dims.begin(), dims.end(), [&](int p, int q) { return rad[p] > rad[q]; });
        dims.resize(10);
    }
    std::vector<std::vector<double>> points;
    if (dims.empty()) points.push_back(c);
    for (double s : {1.0, -1.0})
        for (int d : dims) {
            std::vector<double> x(c);
            x[d] += s * rad[d];
            points.push_back(x);
        }

    // The augmented system [x; 1]' = [A b; 0 0] [x; 1] gives the exact solution by expm.
    const int N = n + 1;
    std::vector<double> Aaug(N * N, 0.0);
    const std::vector<double> Ad = A.data(), bd = uOffset.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) Aaug[i * N + j] = Ad[i * n + j];
        Aaug[i * N + n] = bd[i];
    }
    const Tensor Maug = Tensor::like(A, Aaug, {N, N});
    auto toOutput = [&](const std::vector<double> &x) {
        if (!C) return x;
        const std::vector<int64_t> cs = C->shape();
        const std::vector<double> Cd = C->data();
        std::vector<double> y(cs[0], 0.0);
        for (int64_t i = 0; i < cs[0]; ++i)
            for (int64_t j = 0; j < cs[1]; ++j) y[i] += Cd[i * cs[1] + j] * x[j];
        return y;
    };
    // The exact trajectory through x0, at time t, as an output.
    auto outputAt = [&](const std::vector<double> &x0, double t) {
        const std::vector<double> E = (Maug * t).expm().data();
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = E[i * N + n];
            for (int j = 0; j < n; ++j) x[i] += E[i * N + j] * x0[j];
        }
        return toOutput(x);
    };

    double err = kInf, dt = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::vector<double> t;
        std::vector<std::vector<double>> x0states;
        if (i == 0) {
            aux_ode45(Ad, n, bd, points[0], tFinal, t, x0states);
            // The step of the discrete-time simulation: twice the mean step of the ode solver.
            const double meanStep = (t.back() - t.front()) / double(t.size() - 1);
            dt = tFinal / std::ceil(tFinal / (2 * meanStep));
        } else {
            const int steps = static_cast<int>(std::round(tFinal / dt));
            for (int j = 0; j <= steps; ++j) t.push_back(j * dt);
        }
        const double scaleMin = aux_scaleMin(t);
        std::vector<std::vector<double>> y;
        for (std::size_t j = 0; j < t.size(); ++j)
            y.push_back(i == 0 ? toOutput(x0states[j]) : outputAt(points[i], t[j]));

        // The distance of the output trajectory to the unsafe sets.
        for (const Poly &P : unsafeSets)
            for (std::size_t j = 0; j < t.size(); ++j) {
                if (!P.time.contains(t[j])) continue;
                double est;
                if (aux_contains(P, y[j])) {
                    est = kInf;
                    for (int r = 0; r < P.rows; ++r)
                        est = std::min(est, std::abs(aux_rowDot(P, r, y[j]) - P.b[r]));
                } else if (P.isBounded) {
                    // The distance to the interval hull is counted negative: always smaller.
                    if (!(aux_distanceIntervalPoint(P, y[j]) < err)) continue;
                    est = aux_distancePolyPoint(P, y[j]);
                } else {
                    est = P.rows == 1 ? aux_rowDot(P, 0, y[j]) - P.b[0]
                                      : aux_distancePolyPoint(P, y[j]);
                }
                if (!std::isnan(est)) err = std::min(err, est * scaleMin);
            }

        // The distance to the safe sets.
        for (const Poly &P : safeSets)
            for (std::size_t j = 0; j < t.size(); ++j) {
                if (!P.time.contains(t[j])) continue;
                double est;
                if (aux_contains(P, y[j])) {
                    est = kInf;
                    for (int r = 0; r < P.rows; ++r)
                        est = std::min(est, std::abs(aux_rowDot(P, r, y[j]) - P.b[r]));
                } else if (P.rows == 1) {
                    est = aux_rowDot(P, 0, y[j]) - P.b[0];
                } else {
                    est = aux_distancePolyPoint(P, y[j]);
                }
                // Each point is weighted by the scale at its own time (unlike the unsafe sets).
                if (!std::isnan(est)) err = std::min(err, est * (10 - 9 / t.back() * t[j]));
            }
    }
    return err;
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verifyRA_zonotope(const LinearSys &sys, const VerifyParams &params,
                                    const std::vector<Specification> &specs) {
    const auto start = std::chrono::steady_clock::now();
    VerifyResult res;
    auto finish = [&]() {
        res.tComp = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return res;
    };

    // Validation ----------------------------------------------------------------------------------
    const Tensor &A = sys.A();
    const std::vector<int64_t> shapeA = A.shape();
    if (shapeA.size() != 2 || shapeA[0] != shapeA[1])
        throw std::invalid_argument("LinearSys::verify: A must be a single square matrix");
    const int64_t n = shapeA[0];
    const Zonotope &R0 = params.R0;
    if (R0.c.shape() != std::vector<int64_t>{n, 1} || R0.G.shape().size() != 2 ||
        R0.G.shape()[0] != n)
        throw std::invalid_argument("LinearSys::verify: R0 must be a single zonotope of dim(A)");
    if (!(params.tFinal > 0) || std::isinf(params.tFinal))
        throw std::invalid_argument("LinearSys::verify: tFinal must be positive and finite");
    if (sys.C() && (sys.C()->shape().size() != 2 || sys.C()->shape()[1] != n))
        throw std::invalid_argument("LinearSys::verify: C must be a matrix with dim(A) columns");
    const int64_t p = sys.C() ? sys.C()->shape()[0] : n;

    // The input in canonical form: u in B U = uTrans + B (U - center), U - center by generators.
    Tensor uTrans = Tensor::like(A, std::vector<double>(n, 0.0), {n, 1});
    Tensor GU = Tensor::like(A, {}, {n, 0});
    if (sys.B()) {
        const Tensor &B = *sys.B();
        if (B.shape().size() != 2 || B.shape()[0] != n ||
            params.U.c.shape() != std::vector<int64_t>{B.shape()[1], 1} ||
            params.U.G.shape().size() != 2 || params.U.G.shape()[0] != B.shape()[1])
            throw std::invalid_argument("LinearSys::verify: U must be a zonotope of dim(B.cols)");
        uTrans = B.matmul(params.U.c);
        if (params.U.G.shape()[1] > 0) GU = B.matmul(params.U.G);
    } else {
        for (double v : params.U.c.data())
            if (v != 0) throw std::invalid_argument("LinearSys::verify: U given without a B");
        for (double v : params.U.G.data())
            if (v != 0) throw std::invalid_argument("LinearSys::verify: U given without a B");
    }
    bool anyGen = false;
    for (double v : GU.data()) anyGen = anyGen || v != 0;
    if (!anyGen) GU = Tensor::like(A, {}, {n, 0});

    std::vector<Poly> safeSets, unsafeSets;
    aux_getSetsFromSpec(specs, p, params.tFinal, safeSets, unsafeSets);

    // Adaptive loop ------------------------------------------------------------------------------
    // The maximum allowed error: a first estimate from simulations, refined by the distances.
    // The simulations use no input: CORA centers U before and drops its center from them.
    const Tensor noInput = Tensor::like(A, std::vector<double>(n, 0.0), {n, 1});
    double emax = aux_initError(A, sys.C(), noInput, R0, safeSets, unsafeSets, params.tFinal);
    TaylorLinSys taylor(A);
    AdaptiveSaveData savedata;
    const AdaptiveParams ap{R0, GU, uTrans, sys.C(), params.tFinal};
    while (true) {
        ++res.iterations;
        if (!(emax > 0) || !std::isfinite(emax))
            throw std::runtime_error("LinearSys::verify: the error bound left the valid range");

        // The outer approximation of the reachable set with the error bound emax.
        std::vector<VerifyTime> times;
        for (const Poly &P : unsafeSets) times.push_back(P.time);
        for (const Poly &P : safeSets) times.push_back(P.time);
        const AdaptiveResult R = priv_reach_adaptive(taylor, ap, times, emax, true, savedata);
        res.nrSteps = static_cast<int>(R.timeInt.size());
        res.timeStep = kInf;
        for (std::size_t j = 0; j + 1 < R.time.size(); ++j)
            res.timeStep = std::min(res.timeStep, R.time[j + 1] - R.time[j]);

        // Specifications verified at all times already?
        bool allVerified = true;
        for (const VerifyTime &T : times) allVerified = allVerified && T.empty();
        if (allVerified) {
            res.verified = true;
            return finish();
        }

        // Distances of the outer approximation (Go, Fo) and the inner approximation (Gi, Fi) to
        // the safe sets G and the unsafe sets F, and the steps where they are largest.
        double dGo = -kInf, dGi = -kInf, dFo = -kInf, dFi = kInf;
        int indGo = -1, indGi = -1, indFo = -1, indFi = -1;
        std::vector<std::optional<InnerSet>> inner(R.timeInt.size());
        auto innerOf = [&](std::size_t j) -> const InnerSet & {
            if (!inner[j]) {
                // Z minus the ball of the 1-norm with radius sqrt(q) times the error of the step.
                const ZHost Z = aux_zhost(*R.timeInt[j]);
                inner[j] = InnerSet{Z, std::sqrt(double(Z.n)) * R.timeIntError[j]};
            }
            return *inner[j];
        };
        auto falsified = [&]() {
            res.verified = false;
            return finish();
        };

        // Safe sets -------------------------------------------------------------------------------
        // The outer approximation inside the safe sets, the inner one not inside: unsafe.
        for (Poly &P : safeSets)
            for (std::size_t j = 0; j < R.timeInt.size(); ++j) {
                if (!R.timeInt[j] || !P.time.isIntersecting(R.time[j], R.time[j + 1])) continue;
                const ZHost Z = aux_zhost(*R.timeInt[j]);
                const double err = R.timeIntError[j];
                double d = aux_containmentCheckZono(Z, P);
                if (d < 0) P.time.setdiff(R.time[j], R.time[j + 1]);
                if (d > dGo) dGo = d, indGo = static_cast<int>(j);
                // One halfspace perpendicular to an axis: no need for the inner approximation.
                if (P.fastInner && d > err) return falsified();
                if (d > err) {
                    d = aux_containmentCheckInner(innerOf(j), P);
                    if (d > 0) return falsified();
                } else {
                    d = d - err;
                }
                if (d > dGi) dGi = d, indGi = static_cast<int>(j);
            }

        // Unsafe sets -----------------------------------------------------------------------------
        // The outer approximation outside the unsafe sets, the inner one inside: unsafe.
        for (Poly &P : unsafeSets)
            for (std::size_t j = 0; j < R.timeInt.size(); ++j) {
                if (!R.timeInt[j] || !P.time.isIntersecting(R.time[j], R.time[j + 1])) continue;
                const ZHost Z = aux_zhost(*R.timeInt[j]);
                if (!aux_intersectionCheckFast(Z, P)) {
                    // No intersection: the time interval is verified for this set.
                    P.time.setdiff(R.time[j], R.time[j + 1]);
                    continue;
                }
                double d = aux_distanceIntersectionZono(Z, P);
                if (d > dFo) dFo = d, indFo = static_cast<int>(j);
                if (P.fastInner && d > R.timeIntError[j]) return falsified();
                d = aux_intersectionCheckInner(innerOf(j), P);
                if (d < 0) return falsified();
                if (d < dFi) dFi = d, indFi = static_cast<int>(j);
            }

        // Decided: verified, or an inner approximation that violates a specification.
        if ((dGo <= 0 && dFo <= 0) || dGi > 0 || dFi == 0) {
            res.verified = dGo <= 0 && dFo <= 0;
            return finish();
        }

        // Update the error bound by the distance that is closest to being decided.
        double d;
        int ind;
        if (-dGi < dFi) {
            d = -dGi;
            ind = indGi;
        } else {
            d = dFi;
            ind = indFi;
        }
        if (!unsafeSets.empty() && dFo != 0 && dFo < d) d = dFo, ind = indFo;
        if (!safeSets.empty() && dGo >= 0 && dGo < d) d = dGo, ind = indGo;
        if (ind < 0)
            throw std::runtime_error("LinearSys::verify: no step to refine the error bound by");
        emax = std::max(0.1 * emax, std::min(d * emax / R.timeIntError[ind], 0.9 * emax));
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
