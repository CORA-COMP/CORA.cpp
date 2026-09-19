// The operations against their definitions, and zonotope containment against the LP.
//
// The containment tests are the point of this file: both fast paths — the facets and the
// alternating projections — claim to be exact, so they are checked against the LP on
// points placed just inside and just outside the boundary, where an approximation breaks.
//
//   make test

#include "lp.h"
#include "rng.h"
#include "sets.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cora;
using Eigen::Index;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

bool close(double a, double b) {
    return std::abs(a - b) <= 1e-9 * (1.0 + std::max(std::abs(a), std::abs(b)));
}

void the_operations_match_their_definitions() {
    Rng rng(1);
    const Index n = 3, m = 5, batch = 2;
    const Interval<double> box = random_interval(rng, n, batch);
    const Zonotope<double> zono = random_zonotope(rng, n, m, batch);
    Mat<double> d(n, batch), matrix(n, n);
    rng.normal(d.data(), static_cast<std::size_t>(d.size()), 1.0);
    rng.normal(matrix.data(), static_cast<std::size_t>(matrix.size()), 1.0);

    const auto box_support = support_func(box, d);
    const auto zono_support = support_func(zono, d);
    const Interval<double> image = mat_mul(box, matrix);
    const Zonotope<double> mapped = mat_mul(zono, matrix);

    for (Index b = 0; b < batch; ++b) {
        // An interval's support is attained at a vertex.
        double best = -std::numeric_limits<double>::infinity();
        std::vector<Eigen::Vector3d> vertices;
        for (int mask = 0; mask < 8; ++mask) {
            Eigen::Vector3d v;
            for (Index i = 0; i < n; ++i)
                v(i) = (mask >> i & 1) ? box.hi(i, b) : box.lo(i, b);
            vertices.push_back(v);
            best = std::max(best, d.col(b).dot(v));
        }
        check(close(box_support(b), best), "interval support at a vertex");

        // A zonotope's is c'd + ||G'd||_1.
        const double want = zono.c.col(b).dot(d.col(b))
                            + (zono.block(b).transpose() * d.col(b)).array().abs().sum();
        check(close(zono_support(b), want), "zonotope support");

        // The image of every vertex lies in the interval hull.
        for (const auto &v : vertices) {
            const Eigen::Vector3d image_point = matrix * v;
            check((image_point.array() >= image.lo.col(b).array() - 1e-12).all()
                      && (image_point.array() <= image.hi.col(b).array() + 1e-12).all(),
                  "a vertex left the interval hull");
        }

        // M maps every generator alike.
        check((mapped.block(b) - matrix * zono.block(b)).cwiseAbs().maxCoeff() < 1e-9,
              "matMul on the generators");
    }

    check(mink_sum(zono, zono).m == 2 * m, "minkSum concatenates the generators");
    const Mat<double> drawn = rand_point(box, 20, rng);
    const Mask inside = contains(box, drawn, 20);
    check(std::find(inside.begin(), inside.end(), 0) == inside.end(),
          "a point drawn from a box fell outside it");

    // Every generator was scaled to a length in [0, 1].
    check(zono.g.colwise().norm().maxCoeff() <= 1.0 + 1e-12, "generator longer than a unit vector");
}

/// Points placed just inside and just outside the boundary, plus a scatter around the set
/// checked against the LP — for every dimension, so both containment paths are covered.
void zonotope_containment_is_exact() {
    for (Index n : {Index(1), Index(2), Index(3), Index(5), Index(10), Index(30)}) {
        Rng rng(static_cast<unsigned long long>(7 + n));
        const Index m = 2 * n, batch = 3, points = 12;
        const Zonotope<double> zono = random_zonotope(rng, n, m, batch);

        Mat<double> near(n, points * batch), far(n, points * batch), betas(m, points * batch),
            dirs(n, points * batch);
        rng.uniform(betas.data(), static_cast<std::size_t>(betas.size()), -1.0, 1.0);
        rng.normal(dirs.data(), static_cast<std::size_t>(dirs.size()), 1.0);

        for (Index b = 0; b < batch; ++b) {
            for (Index k = 0; k < points; ++k) {
                const Index col = b * points + k;
                // Saturate the largest coordinate, so the point sits on a facet.
                Eigen::VectorXd beta = betas.col(col);
                Index top = 0;
                beta.cwiseAbs().maxCoeff(&top);
                beta(top) = beta(top) >= 0 ? 1.0 : -1.0;
                near.col(col) = zono.c.col(b) + zono.block(b) * (0.999 * beta);

                const Eigen::VectorXd d = dirs.col(col);
                const double reach = (zono.block(b).transpose() * d).array().abs().sum();
                far.col(col) = zono.c.col(b) + 1.001 * reach / d.squaredNorm() * d;
            }
        }

        const Mask in_answers = contains(zono, near, points);
        check(std::find(in_answers.begin(), in_answers.end(), 0) == in_answers.end(),
              "n=" + std::to_string(n) + ": a point inside the set was reported outside");
        const Mask out_answers = contains(zono, far, points);
        check(std::find(out_answers.begin(), out_answers.end(), 1) == out_answers.end(),
              "n=" + std::to_string(n) + ": a point outside the set was reported inside");

        // A scatter around the set: the LP decides, and the fast path must agree.
        Mat<double> mixed(n, points * batch), jitter(n, points * batch), offset(n, batch);
        rng.normal(jitter.data(), static_cast<std::size_t>(jitter.size()), 0.3);
        rng.uniform(offset.data(), static_cast<std::size_t>(offset.size()), -1.0, 1.0);
        for (Index b = 0; b < batch; ++b) {
            const Eigen::VectorXd reach = zono.block(b).cwiseAbs().rowwise().sum();
            const Eigen::VectorXd corner =
                zono.c.col(b) + reach.cwiseProduct(offset.col(b));
            for (Index k = 0; k < points; ++k)
                mixed.col(b * points + k) =
                    corner + jitter.col(b * points + k).cwiseProduct(reach);
        }
        const Mask answers = contains(zono, mixed, points);
        for (Index b = 0; b < batch; ++b) {
            for (Index k = 0; k < points; ++k) {
                const Index col = b * points + k;
                const Eigen::VectorXd r = mixed.col(col) - zono.c.col(b);
                const bool want = contains_lp(zono.block(b), r, 1e-9);
                check(static_cast<bool>(answers[static_cast<std::size_t>(col)]) == want,
                      "n=" + std::to_string(n) + ", point " + std::to_string(col)
                          + ": disagreed with the LP");
            }
        }
    }
}

/// Box–Muller has to produce genuine standard normals, and the streams must not repeat.
void the_random_numbers_are_what_they_claim() {
    Rng rng(4);
    const std::size_t count = 1 << 20;
    std::vector<double> sample(count);
    rng.normal(sample.data(), count, 1.0);

    double mean = 0.0, second = 0.0, fourth = 0.0;
    for (double v : sample) {
        mean += v;
        second += v * v;
        fourth += v * v * v * v;
    }
    mean /= static_cast<double>(count);
    second /= static_cast<double>(count);
    fourth /= static_cast<double>(count);
    check(std::abs(mean) < 0.01, "normal mean");
    check(std::abs(second - 1.0) < 0.01, "normal variance");
    check(std::abs(fourth - 3.0) < 0.1, "normal fourth moment");

    // Successive draws must differ: a daemon reusing one stream would repeat an instance.
    std::vector<double> again(16);
    rng.normal(again.data(), again.size(), 1.0);
    check(again[0] != sample[0], "two draws returned the same numbers");

    std::vector<double> u(count);
    rng.uniform(u.data(), count, -1.0, 1.0);
    double lo = 1e9, hi = -1e9, sum = 0.0;
    for (double v : u) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        sum += v;
    }
    check(lo >= -1.0 && hi <= 1.0, "uniform stayed in range");
    check(std::abs(sum / static_cast<double>(count)) < 0.01, "uniform mean");
}

} // namespace

int main() {
    the_operations_match_their_definitions();
    zonotope_containment_is_exact();
    the_random_numbers_are_what_they_claim();
    if (failures == 0) std::cout << "all tests passed\n";
    return failures == 0 ? 0 : 1;
}
