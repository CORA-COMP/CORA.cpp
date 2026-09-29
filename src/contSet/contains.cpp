// Exact point containment in a zonotope.
//
// An LP in CORA; here it is exact without one on all but a handful of points. Two paths,
// the ones CORA.py and CORA.julia take: through the facets while there are few of them,
// otherwise by certified alternating projections, with the LP for what neither settles.
//
// Each set of a batch is handled on its own, so the batch is one parallel loop.

#include "lp.h"
#include "sets.h"
#include "threads.h"

#include <Eigen/QR>

#include <limits>
#include <vector>

namespace cora {
namespace {

/// Most facet normals `C(m, n-1)` for which containment goes through the facets; above it,
/// the projections are cheaper.
constexpr unsigned long long kMaxFacetNormals = 300;

/// Relative tolerance of a facet-based containment answer.
constexpr double kFacetTol = 1e-12;

/// Box half-width of the containment projections; see `by_projection`.
constexpr double kShrink = 0.9;

/// Iterations before a point goes to the LP, and how often the certificates are checked.
constexpr int kIterations = 60;
constexpr int kCheckEvery = 4;

using Index = Eigen::Index;

/// `C(m, k)`, stopped as soon as it is past the threshold it is compared against.
unsigned long long binomial(Index m, Index k) {
    k = std::min(k, m - k);
    unsigned long long c = 1;
    for (Index i = 0; i < k; ++i) {
        c = c * static_cast<unsigned long long>(m - i) / static_cast<unsigned long long>(i + 1);
        if (c > kMaxFacetNormals) return c;
    }
    return c;
}

/// Every `k`-subset of `0..m`, in lexicographic order.
std::vector<std::vector<Index>> combinations(Index m, Index k) {
    std::vector<std::vector<Index>> out;
    std::vector<Index> s(static_cast<std::size_t>(k));
    for (Index i = 0; i < k; ++i) s[static_cast<std::size_t>(i)] = i;
    while (true) {
        out.push_back(s);
        Index i = k;
        while (i >= 1 && s[static_cast<std::size_t>(i - 1)] == m - k + i - 1) --i;
        if (i == 0) return out;
        ++s[static_cast<std::size_t>(i - 1)];
        for (Index j = i; j < k; ++j)
            s[static_cast<std::size_t>(j)] = s[static_cast<std::size_t>(j - 1)] + 1;
    }
}

/// Every undecided point of one set, settled by the exact LP.
void settle_by_lp(const Eigen::Ref<const Eigen::MatrixXd> &g, const Eigen::MatrixXd &r,
                  const std::vector<char> &open, double tol, std::uint8_t *out) {
    for (Index k = 0; k < r.cols(); ++k)
        if (open[static_cast<std::size_t>(k)])
            out[k] = contains_lp(g, r.col(k), tol) ? 1 : 0;
}

/// A full-dimensional zonotope is the intersection of the slabs
/// `|h'(x - c)| <= sum_j |h'g_j|`, one for each `n - 1` generators whose normal `h` — their
/// generalized cross product — is nonzero. A zonotope whose normals all vanish is flat and
/// goes to the LP instead.
void by_facets(const Eigen::Ref<const Eigen::MatrixXd> &g, const Eigen::MatrixXd &r,
               const std::vector<std::vector<Index>> &subsets, std::uint8_t *out) {
    const Index n = g.rows(), points = r.cols();
    const std::vector<char> all_open(static_cast<std::size_t>(points), 1);

    if (n == 1) {
        const double reach = g.array().abs().sum();
        if (reach == 0.0) {
            settle_by_lp(g, r, all_open, kFacetTol, out);
            return;
        }
        const double bound = reach * (1.0 + kFacetTol);
        for (Index k = 0; k < points; ++k) out[k] = std::abs(r(0, k)) <= bound ? 1 : 0;
        return;
    }

    // One normal per (n-1)-subset of the generators, by cofactor expansion.
    const Index faces = static_cast<Index>(subsets.size());
    Eigen::MatrixXd normals(faces, n);
    Eigen::MatrixXd minor(n - 1, n - 1);
    for (Index f = 0; f < faces; ++f) {
        const auto &s = subsets[static_cast<std::size_t>(f)];
        for (Index i = 0; i < n; ++i) {
            for (Index col = 0; col < n - 1; ++col) {
                Index row = 0;
                for (Index t = 0; t < n; ++t)
                    if (t != i) minor(row++, col) = g(t, s[static_cast<std::size_t>(col)]);
            }
            normals(f, i) = (i % 2 == 0 ? 1.0 : -1.0) * minor.determinant();
        }
    }

    const Eigen::VectorXd scale = normals.cwiseAbs().rowwise().maxCoeff();
    const double top = scale.maxCoeff();
    if (top == 0.0) {
        settle_by_lp(g, r, all_open, kFacetTol, out);
        return;
    }
    // A normal far smaller than the rest is numerical dust, not a facet.
    std::vector<Index> keep;
    for (Index f = 0; f < faces; ++f)
        if (scale(f) > kFacetTol * top) keep.push_back(f);
    if (keep.empty()) {
        settle_by_lp(g, r, all_open, kFacetTol, out);
        return;
    }

    Eigen::MatrixXd h(static_cast<Index>(keep.size()), n);
    for (std::size_t i = 0; i < keep.size(); ++i) h.row(static_cast<Index>(i)) = normals.row(keep[i]);
    const Eigen::VectorXd reach =
        (h * g).cwiseAbs().rowwise().sum() * (1.0 + kFacetTol);
    const Eigen::MatrixXd projected = (h * r).cwiseAbs(); // (F, points)
    for (Index k = 0; k < points; ++k)
        out[k] = (projected.col(k).array() <= reach.array()).all() ? 1 : 0;
}

/// `p` is in the zonotope iff the affine set `A = {b | G b = r}` meets `[-1, 1]^m`.
///
/// Alternating projections between `A` and the shrunk box `[-s, s]^m` find a `b` in `A`
/// with `||b||_inf <= 1` within a few iterations for every point well inside, which proves
/// containment; a gap `u` between the two sets gives a direction `d` that proves the
/// opposite if `d'r > ||G'd||_1`. The shrunk box is what makes interior points converge
/// fast: its intersection with `A` has slack, where `[-1, 1]^m` would only be approached
/// asymptotically. A point neither proves in `kIterations` goes to the exact LP; in the
/// dimensions this path serves, hardly any point lies that close to the boundary.
///
/// With `G' = Q R` (thin QR, `G` of full row rank), projecting onto `A` is `b - Q Q'b + b0`
/// with the least-norm solution `b0 = Q R^-T r`, and `d = R^-1 Q'u` gives `d'r = (Q'u)'w`
/// and `G'd = Q Q'u` without a solve.
void by_projection(const Eigen::Ref<const Eigen::MatrixXd> &g, const Eigen::MatrixXd &r,
                   std::uint8_t *out) {
    const Index n = g.rows(), m = g.cols(), points = r.cols();
    const double tol = 1000.0 * std::numeric_limits<double>::epsilon();

    const Eigen::MatrixXd gt = g.transpose();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(gt);
    const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(m, n);
    const Eigen::MatrixXd upper = qr.matrixQR().topLeftCorner(n, n);

    const Eigen::MatrixXd w =
        upper.transpose().triangularView<Eigen::Lower>().solve(r); // (n, points)
    const Eigen::MatrixXd b0 = q * w;                              // (m, points)
    const Eigen::RowVectorXd wnorm = w.cwiseAbs().colwise().sum();

    std::vector<char> inside(static_cast<std::size_t>(points), 0);
    std::vector<char> outside(static_cast<std::size_t>(points), 0);
    Eigen::MatrixXd x = b0, a(m, points), y(n, points), spread(m, points);

    for (int it = 0; it <= kIterations; ++it) {
        a = x.cwiseMax(-kShrink).cwiseMin(kShrink);
        if (it % kCheckEvery == 0 || it == kIterations) {
            const Eigen::RowVectorXd reach = x.cwiseAbs().colwise().maxCoeff();
            y.noalias() = q.transpose() * (x - a);
            spread.noalias() = q * y;
            const Eigen::RowVectorXd gap = (y.array() * w.array()).colwise().sum()
                                           - spread.cwiseAbs().colwise().sum().array();
            const Eigen::RowVectorXd slack = y.cwiseAbs().colwise().sum();
            bool done = true;
            for (Index k = 0; k < points; ++k) {
                if (reach(k) <= 1.0 + tol) inside[static_cast<std::size_t>(k)] = 1;
                // Relative to the terms compared, so rounding cannot fake a separation.
                if (gap(k) > tol * slack(k) * (1.0 + wnorm(k)))
                    outside[static_cast<std::size_t>(k)] = 1;
                done &= inside[static_cast<std::size_t>(k)] | outside[static_cast<std::size_t>(k)];
            }
            if (done) break;
        }
        y.noalias() = q.transpose() * a;
        spread.noalias() = q * y;
        x = a - spread + b0;
    }

    std::vector<char> open(static_cast<std::size_t>(points));
    bool any_open = false;
    for (Index k = 0; k < points; ++k) {
        const auto i = static_cast<std::size_t>(k);
        out[k] = inside[i];
        open[i] = static_cast<char>(!(inside[i] | outside[i]));
        any_open |= open[i] != 0;
    }
    if (any_open) settle_by_lp(g, r, open, tol, out);
}

} // namespace

Mask contains(const Zonotope<double> &s, const Mat<double> &p, Eigen::Index points) {
    const Index n = s.dim(), m = s.m, batch = s.batch();
    Mask out(static_cast<std::size_t>(points * batch));
    // Fewer generators than dimensions is a flat zonotope: no facet normals to speak of
    // and no full-row-rank QR, so the LP is the only exact answer left.
    const bool flat = m < n;
    const bool facets = !flat && binomial(m, n - 1) <= kMaxFacetNormals;
    const auto subsets = facets && n > 1 ? combinations(m, n - 1)
                                         : std::vector<std::vector<Index>>();

    const int nt = threads_for(batch, batch * points * n * m);
#pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
    for (Index b = 0; b < batch; ++b) {
        const Eigen::MatrixXd r =
            p.middleCols(b * points, points).colwise() - s.c.col(b);
        std::uint8_t *answers = out.data() + b * points;
        if (flat) {
            settle_by_lp(s.block(b), r,
                         std::vector<char>(static_cast<std::size_t>(points), 1),
                         1000.0 * std::numeric_limits<double>::epsilon(), answers);
        } else if (facets) {
            by_facets(s.block(b), r, subsets, answers);
        } else {
            by_projection(s.block(b), r, answers);
        }
    }
    return out;
}

} // namespace cora
