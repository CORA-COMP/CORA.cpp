// test_tensor_equivalence - every Tensor operation on eigen and torch agrees with the lean backend
//
// The lean backend rounds as CORALean does (test_lean_tensor_ops checks it against the oracle).
// Here each operation of ops.def runs on seeded operands (oracle_testing.h) on lean and on every
// other backend, and the results agree as the table says: exact (the same bits), ulp (at most k
// ulp of lean's), sum (at most k n u times the operation on |operands|, n its length) or norm (at
// most k n u times lean's largest entry). Every operation of the table has a case here.

#include "cora/global/tensor/tensor.h"
#include "global/oracle/oracle_testing.h"
#include "testing.h"

#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>

using namespace cora;
using test::check;

namespace {

using Norm = Tensor::Norm;

/// Seeded operands on the current backend; under `abs` every value is |value| (for the scales).
struct Operands {
    test::oracle::Random rnd;
    bool abs;
    Operands(uint64_t seed, bool abs) : rnd(seed), abs(abs) {}

    Tensor fix(const Tensor &t) const { return abs ? t.abs() : t; }
    Tensor matrix(int64_t r, int64_t c) { return fix(rnd.matrix(r, c)); }
    /// A batch (b, r, c).
    Tensor batch(int64_t b, int64_t r, int64_t c) { return matrix(b * r, c).reshape({b, r, c}); }
    /// Entries in (lo, hi).
    Tensor within(int64_t r, int64_t c, double lo, double hi) {
        std::vector<double> v(r * c);
        for (double &x : v) x = rnd.uniform(lo, hi);
        return Tensor::from_data(v, {r, c});
    }
    /// A matrix that is far from singular: random plus n times the identity.
    Tensor regular(int64_t n) {
        const Tensor A = rnd.matrix(n, n);
        return fix(A + A.eye_like() * double(n));
    }
    /// A symmetric positive definite matrix with distinct eigenvalues.
    Tensor spd(int64_t n) {
        const Tensor A = rnd.matrix(n, n);
        return A.matmul(A.matrix_transpose()) + A.eye_like() * double(n);
    }
};

/// One operation's case: the length n of its sums (for the bound) and the results on the
/// current backend.
struct Case {
    int64_t n;
    std::function<std::vector<Tensor>(Operands &)> run;
};

/// The host values of a pair of results, and their shapes.
bool aux_shapes(const std::string &what, const Tensor &ref, const Tensor &got) {
    if (ref.shape() == got.shape()) return true;
    check(false, what + ": the shapes differ");
    return false;
}

/// The spacing of the doubles at |x|.
double aux_ulp(double x) {
    const double a = std::fabs(x);
    return std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
}

/// Whether got agrees with ref as `info` states, scale the results on |operands|.
void aux_agree(const OpInfo &info, int64_t n, const Tensor &ref, const Tensor &got,
               const Tensor &scale, const std::string &what) {
    if (!aux_shapes(what, ref, got)) return;
    if (info.agreement == Agreement::exact) {
        check(test::oracle::same_bits(what, got, ref), what + ": the same bits as lean");
        return;
    }
    const std::vector<double> r = ref.data(), g = got.data(), s = scale.data();
    const double u = std::numeric_limits<double>::epsilon() / 2;
    double largest = 0;
    for (const double x : r) largest = std::max(largest, std::fabs(x));
    for (std::size_t i = 0; i < r.size(); ++i) {
        const double diff = std::fabs(r[i] - g[i]);
        double bound = 0;
        if (info.agreement == Agreement::ulp) bound = info.k * aux_ulp(r[i]);
        if (info.agreement == Agreement::sum) bound = info.k * n * u * std::fabs(s[i]);
        if (info.agreement == Agreement::norm) bound = info.k * n * u * largest;
        if (!(diff <= bound) && !(std::isnan(r[i]) && std::isnan(g[i]))) {
            check(false, what + " [" + std::to_string(i) + "]: " + std::to_string(g[i]) +
                             " against lean's " + std::to_string(r[i]) + ", beyond the bound " +
                             std::to_string(bound));
            return;
        }
    }
}

/// The host indices as a tensor like x.
Tensor aux_indices(const Tensor &x, const std::vector<int64_t> &idx) {
    return Tensor::like(x, std::vector<double>(idx.begin(), idx.end()),
                        {static_cast<int64_t>(idx.size())});
}

/// The cases, by operation.
std::map<std::string, Case> aux_cases() {
    std::map<std::string, Case> c;
    using O = Operands;
    using V = std::vector<Tensor>;
    // construction
    c["like"] = {1, [](O &o) { return V{Tensor::like(o.matrix(2, 2), {1, 2, 3}, {3, 1})}; }};
    c["full"] = {1, [](O &o) { return V{o.matrix(1, 1).full_like({2, 3}, 0.1)}; }};
    c["arange"] = {1, [](O &o) { return V{o.matrix(1, 1).arange_like(5)}; }};
    c["eye"] = {1, [](O &o) { return V{o.matrix(1, 1).eye_like(4), o.batch(2, 3, 3).eye_like()}; }};
    c["to"] = {1, [](O &o) { return V{o.matrix(3, 2).to("cpu"), o.matrix(2, 2).to(DType::float64)}; }};
    c["data"] = {1, [](O &o) {
        const Tensor x = o.matrix(3, 4);
        return V{Tensor::like(x, x.data(), x.shape())};
    }};
    // elementwise, with broadcasting and a batch
    const auto binary = [](std::function<Tensor(const Tensor &, const Tensor &)> f) {
        return Case{1, [f](O &o) {
            const Tensor a = o.matrix(3, 4), b = o.matrix(3, 1), z = o.batch(2, 3, 4);
            const Tensor w = o.matrix(3, 4);
            return V{f(a, w), f(a, b), f(z, a), f(b, z)};
        }};
    };
    c["add"] = binary([](const Tensor &a, const Tensor &b) { return a + b; });
    c["subtract"] = binary([](const Tensor &a, const Tensor &b) { return a - b; });
    c["multiply"] = binary([](const Tensor &a, const Tensor &b) { return a.multiply(b); });
    c["divide"] = binary([](const Tensor &a, const Tensor &b) { return a.divide(b); });
    c["maximum"] = binary([](const Tensor &a, const Tensor &b) { return a.maximum(b); });
    c["minimum"] = binary([](const Tensor &a, const Tensor &b) { return a.minimum(b); });
    c["remainder"] = binary([](const Tensor &a, const Tensor &b) { return a.remainder(b); });
    c["atan2"] = binary([](const Tensor &a, const Tensor &b) { return a.atan2(b); });
    c["equal"] = binary([](const Tensor &a, const Tensor &b) { return a.equal(b.maximum(a)); });
    c["not_equal"] = binary([](const Tensor &a, const Tensor &b) { return a.not_equal(b.maximum(a)); });
    c["less"] = binary([](const Tensor &a, const Tensor &b) { return a.less(b); });
    c["less_equal"] = binary([](const Tensor &a, const Tensor &b) { return a.less_equal(b.maximum(a)); });
    c["greater"] = binary([](const Tensor &a, const Tensor &b) { return a.greater(b); });
    c["greater_equal"] =
        binary([](const Tensor &a, const Tensor &b) { return a.greater_equal(b.minimum(a)); });
    c["pow"] = {1, [](O &o) {
        const Tensor base = o.within(3, 4, 0.1, 3.0), p = o.matrix(3, 4);
        return V{base.pow(p), base.pow(2.5)};
    }};
    const auto unary = [](std::function<Tensor(const Tensor &)> f, double lo, double hi) {
        return Case{1, [f, lo, hi](O &o) { return V{f(o.within(4, 5, lo, hi))}; }};
    };
    c["negative"] = unary([](const Tensor &x) { return x.negative(); }, -3, 3);
    c["abs"] = unary([](const Tensor &x) { return x.abs(); }, -3, 3);
    c["sign"] = unary([](const Tensor &x) { return x.sign(); }, -3, 3);
    c["floor"] = unary([](const Tensor &x) { return x.floor(); }, -3, 3);
    c["ceil"] = unary([](const Tensor &x) { return x.ceil(); }, -3, 3);
    c["round"] = unary([](const Tensor &x) { return (x * 2.0).round(); }, -3, 3);
    c["sqrt"] = unary([](const Tensor &x) { return x.sqrt(); }, 0, 9);
    c["exp"] = unary([](const Tensor &x) { return x.exp(); }, -5, 5);
    c["log"] = unary([](const Tensor &x) { return x.log(); }, 0.01, 50);
    c["sin"] = unary([](const Tensor &x) { return x.sin(); }, -6, 6);
    c["cos"] = unary([](const Tensor &x) { return x.cos(); }, -6, 6);
    c["tan"] = unary([](const Tensor &x) { return x.tan(); }, -1.5, 1.5);
    c["asin"] = unary([](const Tensor &x) { return x.asin(); }, -0.99, 0.99);
    c["acos"] = unary([](const Tensor &x) { return x.acos(); }, -0.99, 0.99);
    c["atan"] = unary([](const Tensor &x) { return x.atan(); }, -5, 5);
    c["sinh"] = unary([](const Tensor &x) { return x.sinh(); }, -3, 3);
    c["cosh"] = unary([](const Tensor &x) { return x.cosh(); }, -3, 3);
    c["tanh"] = unary([](const Tensor &x) { return x.tanh(); }, -3, 3);
    c["asinh"] = unary([](const Tensor &x) { return x.asinh(); }, -5, 5);
    c["acosh"] = unary([](const Tensor &x) { return x.acosh(); }, 1.01, 9);
    c["atanh"] = unary([](const Tensor &x) { return x.atanh(); }, -0.99, 0.99);
    c["square"] = unary([](const Tensor &x) { return x.square(); }, -3, 3);
    c["sigmoid"] = unary([](const Tensor &x) { return x.sigmoid(); }, -6, 6);
    c["clip"] = unary([](const Tensor &x) { return x.clip(-1, 0.5); }, -3, 3);
    const auto special = [](std::function<Tensor(const Tensor &)> f) {
        return Case{1, [f](O &o) {
            const Tensor x = o.matrix(1, 4);
            const double inf = std::numeric_limits<double>::infinity();
            return V{f(Tensor::concat({x, Tensor::like(x, {std::nan(""), inf, -inf, 0.0}, {1, 4})}, -1))};
        }};
    };
    c["isnan"] = special([](const Tensor &x) { return x.isnan(); });
    c["isinf"] = special([](const Tensor &x) { return x.isinf(); });
    c["isfinite"] = special([](const Tensor &x) { return x.isfinite(); });
    c["where"] = {1, [](O &o) {
        const Tensor a = o.matrix(3, 4), b = o.batch(2, 3, 4), m = o.matrix(3, 1);
        return V{Tensor::where(m.scalar_op(Tensor::Binary::greater, 0.0), a, b)};
    }};
    // reductions along every axis of a batch
    const auto along = [](std::function<Tensor(const Tensor &, int)> f) {
        return Case{6, [f](O &o) {
            const Tensor x = o.batch(3, 4, 6);
            return V{f(x, 0), f(x, 1), f(x, -1)};
        }};
    };
    c["sum"] = along([](const Tensor &x, int a) { return x.sum(a, true); });
    c["prod"] = along([](const Tensor &x, int a) { return x.prod(a); });
    c["max"] = along([](const Tensor &x, int a) { return x.max(a); });
    c["min"] = along([](const Tensor &x, int a) { return x.min(a, true); });
    c["argmax"] = along([](const Tensor &x, int a) { return x.argmax(a); });
    c["argmin"] = along([](const Tensor &x, int a) { return x.argmin(a, true); });
    c["cumsum"] = along([](const Tensor &x, int a) { return x.cumsum(a); });
    c["mean"] = along([](const Tensor &x, int a) { return x.mean(a); });
    c["softmax"] = along([](const Tensor &x, int a) { return x.softmax(a); });
    const auto counting = [](std::function<Tensor(const Tensor &, int)> f) {
        return Case{1, [f](O &o) {
            const Tensor x = o.batch(3, 4, 6).scalar_op(Tensor::Binary::greater, 0.5);
            return V{f(x, 0), f(x, 1), f(x, -1)};
        }};
    };
    c["any"] = counting([](const Tensor &x, int a) { return x.any(a); });
    c["all"] = counting([](const Tensor &x, int a) { return x.all(a); });
    c["count_nonzero"] = counting([](const Tensor &x, int a) { return x.count_nonzero(a); });
    // shape and selection
    c["reshape"] = {1, [](O &o) { return V{o.batch(2, 3, 4).reshape({4, 6})}; }};
    c["permute_dims"] = {1, [](O &o) { return V{o.batch(2, 3, 4).permute_dims({2, 0, 1})}; }};
    c["broadcast_to"] = {1, [](O &o) { return V{o.matrix(3, 1).broadcast_to({2, 3, 4})}; }};
    c["slice"] = {1, [](O &o) {
        const Tensor x = o.batch(2, 5, 7);
        return V{x.slice(-1, 1, 7, 2), x.slice(1, 2, 3)};
    }};
    c["take"] = {1, [](O &o) {
        const Tensor x = o.batch(2, 5, 7);
        return V{x.take({6, 0, 3, 3}, -1), x.take({4}, 1)};
    }};
    c["take_along_axis"] = {1, [](O &o) {
        const Tensor x = o.batch(2, 5, 7);
        const Tensor idx = Tensor::like(x, {6, 0, 3}, {1, 1, 3});
        return V{x.take_along_axis(idx, -1)};
    }};
    c["concat"] = {1, [](O &o) {
        const Tensor a = o.batch(2, 3, 4), b = o.batch(2, 3, 2), d = o.batch(2, 1, 4);
        return V{Tensor::concat({a, b}, -1), Tensor::concat({a, d}, 1), Tensor::concat({a, a}, 0)};
    }};
    c["matrix_transpose"] = {1, [](O &o) { return V{o.batch(2, 3, 4).matrix_transpose()}; }};
    c["expand_dims"] = {1, [](O &o) { return V{o.matrix(3, 4).expand_dims(1)}; }};
    c["squeeze"] = {1, [](O &o) { return V{o.batch(3, 1, 4).squeeze(1)}; }};
    c["stack"] = {1, [](O &o) {
        const Tensor a = o.matrix(3, 4), b = o.matrix(3, 4);
        return V{Tensor::stack({a, b}), Tensor::stack({a, b}, -1)};
    }};
    c["flip"] = {1, [](O &o) { return V{o.batch(2, 3, 4).flip(-1)}; }};
    c["tile"] = {1, [](O &o) { return V{o.matrix(2, 3).tile({2, 3})}; }};
    c["diag"] = {1, [](O &o) { return V{o.batch(2, 4, 1).diag()}; }};
    c["diagonal"] = {1, [](O &o) { return V{o.batch(2, 3, 5).diagonal(), o.matrix(5, 3).diagonal()}; }};
    c["tril"] = {1, [](O &o) { return V{o.batch(2, 4, 5).tril(), o.matrix(4, 4).tril(-1)}; }};
    c["triu"] = {1, [](O &o) { return V{o.batch(2, 4, 5).triu(1), o.matrix(4, 4).triu()}; }};
    c["block_diag"] = {1, [](O &o) {
        return V{Tensor::block_diag({o.matrix(2, 3), o.matrix(1, 1), o.matrix(3, 2)})};
    }};
    // sorting, with ties
    c["argsort"] = {1, [](O &o) {
        const Tensor x = o.batch(2, 3, 8).round();
        return V{x.argsort(-1), x.argsort(1, true)};
    }};
    c["sort"] = {1, [](O &o) {
        const Tensor x = o.batch(2, 3, 8).round();
        return V{x.sort(-1), x.sort(0, true)};
    }};
    c["topk"] = {1, [](O &o) {
        const auto [v, i] = o.batch(2, 3, 8).round().topk(3);
        const auto [w, j] = o.batch(2, 3, 8).topk(2, 1, false);
        return V{v, i, w, j};
    }};
    c["nonzero"] = {1, [](O &o) {
        const Tensor x = o.matrix(4, 5).scalar_op(Tensor::Binary::greater, 0.3);
        return V{aux_indices(x, x.nonzero())};
    }};
    // linear algebra
    c["matmul"] = {7, [](O &o) {
        const Tensor A = o.matrix(5, 7), B = o.matrix(7, 3), Z = o.batch(2, 7, 3);
        return V{A.matmul(B), A.matmul(Z)};
    }};
    c["solve"] = {5, [](O &o) { return V{o.regular(5).solve(o.matrix(5, 2))}; }};
    c["inv"] = {5, [](O &o) { return V{o.regular(5).inv()}; }};
    c["lstsq"] = {7, [](O &o) { return V{o.matrix(7, 3).lstsq(o.matrix(7, 2))}; }};
    c["cholesky"] = {5, [](O &o) { return V{o.spd(5).cholesky()}; }};
    c["qr"] = {5, [](O &o) { const auto [Q, R] = o.regular(5).qr(); return V{Q, R}; }};
    c["svd"] = {5, [](O &o) { const auto [U, S, W] = o.regular(5).svd(); return V{U, S, W}; }};
    c["eigh"] = {5, [](O &o) { const auto [w, V_] = o.spd(5).eigh(); return V{w, V_}; }};
    c["pinv"] = {5, [](O &o) { return V{o.regular(5).pinv()}; }};
    c["matrix_rank"] = {1, [](O &o) {
        const Tensor A = o.matrix(5, 2);
        const Tensor low = A.matmul(A.matrix_transpose());
        return V{Tensor::like(low, {double(low.matrix_rank())}, {1, 1})};
    }};
    c["trace"] = {5, [](O &o) { return V{o.batch(2, 5, 5).trace()}; }};
    c["kron"] = {1, [](O &o) {
        return V{o.matrix(2, 3).kron(o.matrix(3, 2)), o.batch(2, 2, 2).kron(o.matrix(2, 1))};
    }};
    c["expm"] = {4, [](O &o) { return V{(o.matrix(4, 4) * 0.5).expm(), (o.batch(2, 3, 3) * 3.0).expm()}; }};
    c["matrix_norm"] = {6, [](O &o) {
        const Tensor A = o.batch(2, 6, 4);
        return V{A.matrix_norm(Norm::one), A.matrix_norm(Norm::inf), A.matrix_norm(Norm::fro),
                 A.matrix_norm(Norm::two)};
    }};
    c["vector_norm"] = {6, [](O &o) {
        const Tensor x = o.batch(2, 6, 4);
        return V{x.vector_norm(Norm::one, 1), x.vector_norm(Norm::inf, -1),
                 x.vector_norm(Norm::two, 1, true)};
    }};
    // fused primitives
    c["zonotope_linear_map"] = {5, [](O &o) {
        const auto [Mc, MG] = Tensor::zonotope_linear_map(o.matrix(4, 5), o.matrix(5, 1), o.batch(2, 5, 9));
        return V{Mc, MG};
    }};
    c["interval_matmul"] = {6, [](O &o) {
        const Tensor lo = o.batch(2, 4, 6), w = o.batch(2, 4, 6).abs();
        const auto [a, b] = Tensor::interval_matmul(lo, lo + w, o.matrix(6, 3));
        return V{a, b};
    }};
    c["matmul_interval"] = {6, [](O &o) {
        const Tensor lo = o.matrix(6, 3), w = o.matrix(6, 3).abs();
        const auto [a, b] = Tensor::matmul_interval(o.batch(2, 4, 6), lo, lo + w);
        return V{a, b};
    }};
    c["girard_split"] = {1, [](O &o) {
        const auto [kept, rest] = o.batch(2, 4, 11).girard_split(5);
        return V{kept, rest};
    }};
    c["abs_sum"] = {9, [](O &o) { return V{o.batch(2, 4, 9).abs_sum()}; }};
    return c;
}

/// The results of a case on the backend `name`, with operands of the seed.
std::vector<Tensor> aux_run(const std::string &name, const Case &c, bool abs) {
    const BackendScope scope(name);
    Operands o(17, abs);
    return c.run(o);
}

} // namespace

int main() {
    const std::map<std::string, Case> cases = aux_cases();
    for (const OpInfo &info : tensor_ops())
        check(cases.count(info.name) == 1, std::string(info.name) + ": a case of the equivalence test");
    for (const OpInfo &info : tensor_ops()) {
        const auto found = cases.find(info.name);
        if (found == cases.end()) continue;
        const Case &c = found->second;
        // lean is the reference; without a lean version (expm), eigen is
        std::vector<Tensor> ref, scale;
        std::string reference = "lean";
        try {
            ref = aux_run("lean", c, false);
        } catch (const Error &) {
            reference = "eigen";
            ref = aux_run("eigen", c, false);
        }
        if (info.agreement == Agreement::sum) scale = aux_run(reference, c, true);
        for (const std::string &b : test::backends()) {
            if (b == reference) continue;
            const std::vector<Tensor> got = aux_run(b, c, false);
            check(got.size() == ref.size(), std::string(info.name) + " on " + b + ": the results");
            for (std::size_t i = 0; i < std::min(got.size(), ref.size()); ++i)
                aux_agree(info, c.n, ref[i], got[i].to("cpu"), scale.empty() ? ref[i] : scale[i].abs(),
                          std::string(info.name) + " on " + b + ", result " + std::to_string(i));
        }
    }
    return test::finish("tensor equivalence");
}
