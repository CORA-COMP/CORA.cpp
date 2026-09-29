// Exact point containment in a zonotope, on tensors.
//
// The libtorch counterpart of contains.cpp, and the same two paths — the facets while
// there are few of them, otherwise certified alternating projections, with the LP for
// what neither settles. Where contains.cpp gives each set of a batch to a thread, here
// the whole batch is one set of tensor calls, which is what lets a GPU do anything with
// it: the batch is flattened to one leading axis, so the shapes read as `(B, ...)`.
//
// The LP runs on the host. It settles a handful of points at most, and only the points
// the certificates leave open ever reach it.

#include "torch_sets.h"

#include "lp.h"

#include <Eigen/Dense>

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace cora::tb {
namespace {

/// Most facet normals `C(m, n-1)` for which containment goes through the facets; above
/// it, the projections are cheaper.
constexpr unsigned long long kMaxFacetNormals = 300;

/// Relative tolerance of a facet-based containment answer.
constexpr double kFacetTol = 1e-12;

/// Box half-width of the containment projections; see `by_projection`.
constexpr double kShrink = 0.9;

/// Iterations before a point goes to the LP, and how often the certificates are checked.
constexpr int kIterations = 60;
constexpr int kCheckEvery = 4;

/// Largest containment projector held as a matrix, in entries: 128 MB of `double`.
constexpr int64_t kMaxProjectorEntries = 1 << 24;

/// Scalar operations a host thread needs before it pays for being gathered into one of
/// the projection loop's products.
constexpr int64_t kMinWorkPerHostThread = 1 << 19;

/// Narrows torch's intra-op pool to what the work can use, restoring it on the way out.
///
/// The projection loop is a few hundred products of a few hundred thousand operations
/// each. MKL spreads every one of them across every core, and on the competition's
/// two-socket worker gathering eighty threads costs an order of magnitude more than the
/// product: an unbatched 50d instance takes 1.82 s at the default width and 0.043 s at
/// one thread. A device has no such pool and is left alone.
class HostThreads {
  public:
    HostThreads(bool host, int64_t work) : previous_(host ? torch::get_num_threads() : 0) {
        if (previous_ <= 0) return;
        const int64_t by_work = std::max<int64_t>(1, work / kMinWorkPerHostThread);
        torch::set_num_threads(static_cast<int>(std::min<int64_t>(by_work, previous_)));
    }
    ~HostThreads() {
        if (previous_ > 0) torch::set_num_threads(previous_);
    }

    HostThreads(const HostThreads &) = delete;
    HostThreads &operator=(const HostThreads &) = delete;

  private:
    int previous_;
};

/// `C(m, k)`, stopped as soon as it is past the threshold it is compared against.
unsigned long long binomial(int64_t m, int64_t k) {
    k = std::min(k, m - k);
    unsigned long long c = 1;
    for (int64_t i = 0; i < k; ++i) {
        c = c * static_cast<unsigned long long>(m - i) / static_cast<unsigned long long>(i + 1);
        if (c > kMaxFacetNormals) return c;
    }
    return c;
}

/// Every `k`-subset of `0..m`, in lexicographic order, flattened.
std::vector<int64_t> combinations(int64_t m, int64_t k) {
    std::vector<int64_t> out;
    std::vector<int64_t> s(static_cast<std::size_t>(k));
    for (int64_t i = 0; i < k; ++i) s[static_cast<std::size_t>(i)] = i;
    while (true) {
        out.insert(out.end(), s.begin(), s.end());
        int64_t i = k;
        while (i >= 1 && s[static_cast<std::size_t>(i - 1)] == m - k + i - 1) --i;
        if (i == 0) return out;
        ++s[static_cast<std::size_t>(i - 1)];
        for (int64_t j = i; j < k; ++j)
            s[static_cast<std::size_t>(j)] = s[static_cast<std::size_t>(j - 1)] + 1;
    }
}

/// The index tensors of the facet normals' minors: every `(n-1)`-subset of the `m`
/// generators `(F, n-1)`, for each row the other `n-1` rows `(n, n-1)`, and the cofactor
/// signs `(n,)` of the generalized cross product. They depend only on the shape and the
/// device, and building them means a host-to-device copy, so they are kept. Handed back
/// by value: copying a tensor is a refcount, and a reference into the cache would be one
/// more thing to keep straight.
struct Facets {
    torch::Tensor subsets, rows, signs;
};

/// The Leibniz terms of a `k x k` determinant: the flat indices `i*k + sigma(i)` of each
/// permutation `(k!, k)` and its sign `(k!,)`. Cached like the facet indices.
struct Perms {
    torch::Tensor index, signs;
};

Perms permutations(int64_t k, const torch::TensorOptions &opts) {
    using Key = std::tuple<int64_t, int8_t, int8_t>;
    static std::map<Key, Perms> cache;
    static std::mutex guard;

    const torch::Device dev = opts.device();
    const Key key{k, static_cast<int8_t>(dev.type()), dev.index()};
    const std::lock_guard<std::mutex> lock(guard);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;

    std::vector<int64_t> sigma(static_cast<std::size_t>(k));
    for (int64_t i = 0; i < k; ++i) sigma[static_cast<std::size_t>(i)] = i;
    std::vector<int64_t> index;
    std::vector<double> signs;
    do {
        for (int64_t i = 0; i < k; ++i) index.push_back(i * k + sigma[static_cast<std::size_t>(i)]);
        int inversions = 0;
        for (std::size_t i = 0; i < sigma.size(); ++i)
            for (std::size_t j = i + 1; j < sigma.size(); ++j)
                if (sigma[i] > sigma[j]) ++inversions;
        signs.push_back(inversions % 2 == 0 ? 1.0 : -1.0);
    } while (std::next_permutation(sigma.begin(), sigma.end()));

    Perms made{
        torch::from_blob(index.data(), {static_cast<int64_t>(index.size())},
                         torch::TensorOptions().dtype(torch::kLong))
            .clone()
            .to(dev),
        torch::from_blob(signs.data(), {static_cast<int64_t>(signs.size())},
                         torch::TensorOptions().dtype(torch::kDouble))
            .clone()
            .to(opts),
    };
    return cache.emplace(key, std::move(made)).first->second;
}

/// Determinants of a batch of `(k, k)` matrices.
///
/// On a device `linalg_det` dispatches to cuSOLVER's batched LU, which costs milliseconds
/// on matrices this small and dominates the facet path; the facet path never needs more
/// than `k = n - 1 <= 4`, where the Leibniz sum is one gather and one product. On the host
/// LAPACK wins instead — the sum materializes `k!` terms per minor, which at `k = 4` costs
/// sixfold on a batch of a hundred 5d zonotopes.
torch::Tensor small_det(const torch::Tensor &minors) {
    const int64_t k = minors.size(-1);
    if (k > 4 || minors.device().is_cpu()) return torch::linalg_det(minors);
    const Perms p = permutations(k, minors.options());
    return minors.flatten(-2, -1)
        .index_select(-1, p.index)
        .unflatten(-1, {-1, k})
        .prod(-1)
        .mul(p.signs)
        .sum(-1);
}

Facets facet_indices(int64_t n, int64_t m, const torch::TensorOptions &opts) {
    using Key = std::tuple<int64_t, int64_t, int8_t, int8_t>;
    static std::map<Key, Facets> cache;
    static std::mutex guard;

    const torch::Device dev = opts.device();
    const Key key{n, m, static_cast<int8_t>(dev.type()), dev.index()};
    const std::lock_guard<std::mutex> lock(guard);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;

    const std::vector<int64_t> subsets = combinations(m, n - 1);
    std::vector<int64_t> rows;
    std::vector<double> signs;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            if (j != i) rows.push_back(j);
        signs.push_back(i % 2 == 0 ? 1.0 : -1.0);
    }
    // from_blob only borrows, and `.to()` onto the device it already sits on would hand
    // back that borrow, so each one is cloned before the vectors go out of scope.
    const auto index = torch::TensorOptions().dtype(torch::kLong);
    Facets made{
        torch::from_blob(const_cast<int64_t *>(subsets.data()),
                         {static_cast<int64_t>(subsets.size()) / (n - 1), n - 1}, index)
            .clone()
            .to(dev),
        torch::from_blob(rows.data(), {n, n - 1}, index).clone().to(dev),
        torch::from_blob(signs.data(), {n}, torch::TensorOptions().dtype(torch::kDouble))
            .clone()
            .to(opts),
    };
    return cache.emplace(key, std::move(made)).first->second;
}

/// A tensor's numbers on the host, row-major and in double precision.
std::vector<double> host(const torch::Tensor &t) {
    const torch::Tensor flat =
        t.contiguous().reshape({-1}).to(torch::kDouble).to(torch::kCPU).contiguous();
    const double *data = flat.data_ptr<double>();
    return std::vector<double>(data, data + flat.numel());
}

std::vector<bool> host_mask(const torch::Tensor &t) {
    const torch::Tensor flat = t.contiguous().reshape({-1}).to(torch::kCPU).contiguous();
    const bool *data = flat.data_ptr<bool>();
    return std::vector<bool>(data, data + flat.numel());
}

/// Replaces the answers `open` marks with what the exact LP says. `r` is `(B, N, n)`, the
/// points relative to their centres; `g` is `(B, n, m)`.
torch::Tensor patch_by_lp(const torch::Tensor &g, const torch::Tensor &r,
                          const torch::Tensor &inside, const torch::Tensor &open,
                          double tol) {
    const int64_t n = g.size(1), m = g.size(2), points = r.size(1);
    const std::vector<double> gh = host(g);
    const std::vector<double> rh = host(r);
    std::vector<bool> answers = host_mask(inside);
    const std::vector<bool> todo = host_mask(open);

    using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    for (std::size_t i = 0; i < answers.size(); ++i) {
        if (!todo[i]) continue;
        const auto set = static_cast<int64_t>(i) / points;
        const Eigen::MatrixXd block =
            Eigen::Map<const RowMajor>(gh.data() + set * n * m, n, m);
        const Eigen::VectorXd point =
            Eigen::Map<const Eigen::VectorXd>(rh.data() + static_cast<int64_t>(i) * n, n);
        answers[i] = contains_lp(block, point, tol);
    }

    std::vector<std::uint8_t> bytes(answers.size());
    for (std::size_t i = 0; i < answers.size(); ++i) bytes[i] = answers[i] ? 1 : 0;
    return torch::from_blob(bytes.data(), {static_cast<int64_t>(bytes.size())},
                            torch::TensorOptions().dtype(torch::kUInt8))
        .to(torch::kBool)
        .reshape(inside.sizes())
        .to(inside.device());
}

/// Every point settled by the LP: the answer for a zonotope neither path can describe.
torch::Tensor all_by_lp(const torch::Tensor &g, const torch::Tensor &c,
                        const torch::Tensor &p) {
    const torch::Tensor all = torch::ones(
        {p.size(0), p.size(1)}, torch::TensorOptions().dtype(torch::kBool).device(g.device()));
    return patch_by_lp(g, p - c.unsqueeze(1), all, all,
                       1000.0 * std::numeric_limits<double>::epsilon());
}

/// A full-dimensional zonotope is the intersection of the slabs
/// `|h'(x - c)| <= sum_j |h'g_j|`, one for each `n - 1` generators whose normal `h` —
/// their generalized cross product — is nonzero. A set whose normals all vanish is flat
/// and goes to the LP instead.
torch::Tensor by_facets(const torch::Tensor &g, const torch::Tensor &c,
                        const torch::Tensor &p) {
    const int64_t b = g.size(0), n = g.size(1), m = g.size(2);
    const torch::Tensor r = p - c.unsqueeze(1); // (B, N, n)
    torch::Tensor inside, flat;

    if (n == 1) {
        const torch::Tensor reach = g.abs().sum({1, 2}); // (B,)
        inside = r.squeeze(-1).abs() <= (reach * (1.0 + kFacetTol)).unsqueeze(-1);
        flat = reach == 0.0;
    } else {
        const Facets f = facet_indices(n, m, g.options());
        const int64_t faces = f.subsets.size(0), k = n - 1;
        // The (n-1)-subsets of the generators, then for each row the minor omitting it.
        const torch::Tensor gs = g.index_select(2, f.subsets.reshape({-1}))
                                     .reshape({b, n, faces, k})
                                     .permute({0, 2, 1, 3}); // (B, F, n, n-1)
        const torch::Tensor minors =
            gs.index_select(2, f.rows.reshape({-1})).reshape({b, faces, n, k, k});
        const torch::Tensor h = small_det(minors) * f.signs; // (B, F, n)

        const torch::Tensor reach = h.matmul(g).abs().sum(-1);
        const torch::Tensor scale = h.abs().amax({-1});       // (B, F)
        const torch::Tensor top = scale.amax({-1}, true);     // (B, 1)
        const torch::Tensor valid = scale > top * kFacetTol;
        const torch::Tensor within =
            r.matmul(h.transpose(1, 2)).abs() <= (reach * (1.0 + kFacetTol)).unsqueeze(1);
        inside = torch::logical_or(within, valid.logical_not().unsqueeze(1)).all(-1);
        flat = torch::logical_or(valid.any(-1).logical_not(), top.squeeze(-1) == 0.0);
    }

    if (flat.any().item<bool>())
        return patch_by_lp(g, r, inside, flat.unsqueeze(-1).expand_as(inside), kFacetTol);
    return inside;
}

/// `p` is in the zonotope iff the affine set `A = {b | G b = p - c}` meets `[-1, 1]^m`.
///
/// Alternating projections between `A` and the shrunk box `[-s, s]^m` find a `b` in `A`
/// with `||b||_inf <= 1` within a few iterations for every point that lies well inside,
/// which proves containment; a gap `u` between the two sets gives a direction `d` that
/// proves the opposite if `d'(p - c) > ||G'd||_1`. The shrunk box is what makes interior
/// points converge fast: its intersection with `A` has slack, where `[-1, 1]^m` would be
/// approached only asymptotically. A point neither proves in `kIterations` goes to the
/// exact LP; in the dimensions this path serves, hardly any point lies that close to the
/// boundary.
///
/// Projecting onto `A` is `b - G'(G G')^-1 G b + b0`, with the least-norm solution
/// `b0 = G'(G G')^-1 (p - c)`, so one Cholesky of the Gram matrix serves every iteration;
/// `d = (G G')^-1 G u` is the separating direction, and `G'd` is the same projection
/// again. A thin QR of `G'` would do as well and be better conditioned, but it factors an
/// `m x n` matrix where this factors `n x n`: at 1000d over a hundred sets that is the
/// difference between finishing inside the catalog's minute and not. The Gram matrix
/// squares the condition number, which costs accuracy, not soundness — every answer is
/// still certified, and a point whose certificates both fail goes to the LP.
torch::Tensor by_projection(const torch::Tensor &g, const torch::Tensor &c,
                            const torch::Tensor &p) {
    const double tol = 1000.0 * std::numeric_limits<double>::epsilon();
    const int64_t b = g.size(0), m = g.size(2), points = p.size(1);
    const torch::Tensor gt = g.transpose(-2, -1); // (B, m, n)
    const HostThreads threads(!g.device().is_cuda(), b * m * m * points);

    // Which factorization gives the projector, and it differs by device. cuSOLVER is far
    // from peak on a batched QR of an `m x n` matrix, where the Gram matrix is a GEMM and
    // the factorization only `n x n`: on an A100 that is the difference between finishing
    // the largest batched instances and timing out. MKL's QR is not far from peak, so on
    // the host the Gram matrix only adds arithmetic — `2 n^2 N` an iteration — besides
    // squaring the condition number, and it costs three- to sixfold.
    const bool gram = g.device().is_cuda();

    torch::Tensor chol, q, r;
    if (gram) {
        auto [factor, info] = torch::linalg_cholesky_ex(g.matmul(gt), /*upper=*/false,
                                                        /*check_errors=*/false);
        // A Gram matrix that will not factor means `G` is not of full row rank, which
        // leaves no projector and no facets either.
        if (info.any().item<bool>()) return all_by_lp(g, c, p);
        chol = factor;
    } else {
        auto [factor, upper] = torch::linalg_qr(gt, "reduced"); // (B,m,n), (B,n,n)
        q = factor;
        r = upper;
    }

    // The points are the columns of one matrix per set. Carrying them as `(B, N, m, 1)`
    // instead would broadcast the set over N in every product, and torch materializes
    // that: at 1000d the iteration reads a 1.6 GB expansion of a 160 MB tensor.
    const torch::Tensor rel = (p - c.unsqueeze(1)).transpose(-2, -1); // (B, n, N)
    // The least-norm solution `G'(G G')^-1 (p - c)`; with the QR it is `Q R^-T (p - c)`,
    // which skips the second solve.
    const torch::Tensor b0 =
        gram ? gt.matmul(torch::cholesky_solve(rel, chol))
             : q.matmul(torch::linalg_solve_triangular(r.transpose(-2, -1), rel,
                                                       /*upper=*/false, /*left=*/true));

    // Where the whole projector fits, an iteration is one fused product instead of a
    // product, a solve and a product. At small `m` this path runs tiny tensors over many
    // repetitions, where an iteration costs almost only its dispatches: a batch of a
    // hundred 10d zonotopes spends 1.3 s on 20x10 matrices. Above the bound the factored
    // form is the only one that fits — at 1000d over a hundred sets the projector alone
    // would be 3.2 GB.
    const bool held = b * m * m <= kMaxProjectorEntries;
    const torch::Tensor pi = // I - G'(G G')^-1 G, the projector onto the null space of G
        !held ? torch::Tensor()
        : gram ? torch::eye(m, g.options()) - gt.matmul(torch::cholesky_solve(g, chol))
               : torch::eye(m, g.options()) - q.matmul(q.transpose(-2, -1));

    /// `P t`, the component of `t` in the row space of `G`.
    const auto row_space = [&](const torch::Tensor &t) {
        if (held) return t - pi.matmul(t);
        if (gram) return gt.matmul(torch::cholesky_solve(g.matmul(t), chol));
        return q.matmul(q.transpose(-2, -1).matmul(t));
    };

    const auto flags = torch::TensorOptions().dtype(torch::kBool).device(g.device());
    torch::Tensor inside = torch::zeros({rel.size(0), rel.size(2)}, flags);
    torch::Tensor outside = torch::zeros({rel.size(0), rel.size(2)}, flags);
    torch::Tensor x = b0;

    for (int it = 0; it <= kIterations; ++it) {
        const torch::Tensor a = x.clamp(-kShrink, kShrink);
        if (it % kCheckEvery == 0 || it == kIterations) {
            inside = torch::logical_or(inside, x.abs().amax({-2}) <= 1.0 + tol);
            const torch::Tensor u = x - a;
            const torch::Tensor reach = row_space(u).abs().sum(-2); // ||G'd||_1
            // d'(p - c) is u'G'(G G')^-1 (p - c) = u'b0, so it needs no product.
            const torch::Tensor along = u * b0;
            const torch::Tensor gap = along.sum(-2) - reach;
            // Relative to the terms compared, so rounding cannot fake a separation.
            const torch::Tensor slack = (along.abs().sum(-2) + reach) * tol;
            outside = torch::logical_or(outside, gap > slack);
            if (torch::logical_or(inside, outside).all().item<bool>()) return inside;
        }
        x = held ? torch::baddbmm(b0, pi, a) : a - row_space(a) + b0;
    }
    const torch::Tensor open = torch::logical_or(inside, outside).logical_not();
    return patch_by_lp(g, p - c.unsqueeze(1), inside, open, tol);
}

} // namespace

torch::Tensor Zonotope::contains(const torch::Tensor &p) const {
    const int64_t n = g.size(-2), m = g.size(-1);
    const std::vector<int64_t> batch(g.sizes().begin(), g.sizes().end() - 2);
    int64_t b = 1;
    for (const int64_t d : batch) b *= d;
    const int64_t points = p.size(-2);

    const torch::Tensor gb = g.reshape({b, n, m});
    const torch::Tensor cb = c.reshape({b, n});
    const torch::Tensor pb = p.reshape({b, points, n});

    // Fewer generators than dimensions is a flat zonotope: no facet normals to speak of
    // and no full-row-rank QR, so the LP is the only exact answer left.
    const torch::Tensor inside = m < n            ? all_by_lp(gb, cb, pb)
                                 : binomial(m, n - 1) <= kMaxFacetNormals
                                     ? by_facets(gb, cb, pb)
                                     : by_projection(gb, cb, pb);

    std::vector<int64_t> shape = batch;
    shape.push_back(points);
    return inside.reshape(shape);
}

} // namespace cora::tb
