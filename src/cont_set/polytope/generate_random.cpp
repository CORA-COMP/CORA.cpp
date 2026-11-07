// generate_random - a random polytope, as CORA's polytope.generateRandom
//
// The options (CORA's 'Dimension', 'NrConstraints', 'IsDegenerate', 'IsBounded') are checked for
// consistency as CORA does, and the missing ones drawn: the dimension in 1..10 (or from the number
// of rows), the number of rows 2n + (n/2..n), degenerate false, bounded true. A bounded set is the
// simplex plus random unit rows; an unbounded one has a free coordinate (n > 2) or is a wedge or a
// half-line; every set is rotated by a random orthogonal matrix, and a bounded one shifted by a
// random point. A degenerate set has two opposite rows. A one-dimensional degenerate set is a
// point (CORA's other rows may exclude it; here they are offset from it).
//
// Syntax:   P = Polytope::generate_random(rng);   P = Polytope::generate_random(rng, 3, 8);
// Inputs:   rng - the random numbers
//           dimension, nr_constraints, is_degenerate, is_bounded - the options, each optional
// Outputs:  P - a polytope of halfspaces, without equalities
// See also: Zonotope::generate_random, Polytope::compact
//
// CORA: contSet/@polytope/generateRandom.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_random_dimension.h"
#include "cont_set/polytope/private/priv.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The options after the consistency checks and the defaults.
struct Options {
    int64_t n = 0, rows = 0;
    bool degenerate = false, bounded = true;
};

// The random numbers ------------------------------------------------------------------------------

/// A uniform integer in [lo, hi] (MATLAB's randi) and a uniform number in [0, 1).
int64_t aux_randi(Rng &rng, int64_t lo, int64_t hi) {
    double x = 0.0;
    rng.uniform(&x, 1, double(lo), double(hi + 1));
    return std::min(hi, int64_t(std::floor(x)));
}

double aux_rand(Rng &rng) {
    double x = 0.0;
    rng.uniform(&x, 1, 0.0, 1.0);
    return x;
}

double aux_randn(Rng &rng) {
    double x = 0.0;
    rng.normal(&x, 1, 1.0);
    return x;
}

/// The random unit vector of n numbers.
Row aux_unit(Rng &rng, int64_t n) {
    Row v(static_cast<std::size_t>(n));
    for (double &x : v) x = aux_randn(rng);
    const double norm = priv_norm(v);
    for (double &x : v) x /= norm;
    return v;
}

// The options -------------------------------------------------------------------------------------

/// CORA's consistency checks of the options that are given.
void aux_check_consistency(const std::optional<int64_t> &n, const std::optional<int64_t> &rows,
                           const std::optional<bool> &degenerate,
                           const std::optional<bool> &bounded) {
    const auto fail = [](const std::string &text) {
        CORA_FAIL(Error::wrong_value, "generate_random: " + text);
    };
    if (rows && *rows == 1) {
        if (bounded && *bounded) fail("is_bounded cannot be true if only one constraint is given");
        if (degenerate && *degenerate)
            fail("is_degenerate cannot be true if only one constraint is given");
    }
    if (n && rows && bounded && *bounded && *rows < *n + 1)
        fail("is_bounded cannot be true if the number of constraints < dimension + 1");
    if (n && *n == 1 && bounded && !*bounded && degenerate && *degenerate)
        fail("one-dimensional polytopes cannot be unbounded and degenerate at the same time");
}

/// The dimension, drawn from the number of rows or at random, when it is not given.
int64_t aux_dimension(Rng &rng, const std::optional<int64_t> &rows,
                      const std::optional<bool> &degenerate, const std::optional<bool> &bounded) {
    if (rows && (!bounded || *bounded))
        return aux_randi(rng, 1, std::max<int64_t>(1, *rows - (degenerate && *degenerate ? 2 : 1)));
    if (bounded && !*bounded && degenerate && *degenerate) return 2;
    return priv_random_dimension(rng);
}

/// CORA's defaults of the options (aux_setDefaultValues), with its errors for a bounded set with
/// too few rows (CORA also refuses is_bounded = false then, which is what few rows force).
Options aux_options(Rng &rng, std::optional<int64_t> n, std::optional<int64_t> rows,
                    std::optional<bool> degenerate, std::optional<bool> bounded) {
    Options o;
    o.n = n ? *n : aux_dimension(rng, rows, degenerate, bounded);
    if (n && rows) {
        if (bounded && *bounded) {
            const int64_t need = o.n + (degenerate && *degenerate ? 2 : 1);
            if (*rows < need)
                CORA_FAIL(Error::wrong_value,
                          "generate_random: a bounded polytope needs at least " +
                              std::to_string(need) + " constraints in this dimension");
        } else if (!bounded) {
            // not asked for: bounded if there are enough rows for a simplex
            bounded = *rows >= o.n + 1;
        }
    }
    o.degenerate = degenerate.value_or(false);
    o.bounded = bounded.value_or(true);
    if (rows) {
        o.rows = *rows;
    } else {
        const bool point = o.n == 1 && o.degenerate;
        o.rows = point ? 2 : 2 * o.n + aux_randi(rng, o.n / 2, o.n);
    }
    return o;
}

// The rows ----------------------------------------------------------------------------------------

/// The rows A (as rows of numbers) and offsets b before the rotation and the shift.
struct Rows2 {
    Rows A;
    Row b;
};

/// The rows of a bounded set: the simplex x <= 1, -sum x / sqrt(n) <= 1 and random unit rows with
/// offset 1; a degenerate set (n > 1) has its last two rows opposite.
Rows2 aux_bounded(Rng &rng, const Options &o) {
    const std::size_t n = std::size_t(o.n);
    Rows2 r;
    if (o.n == 1 && o.degenerate) {
        // the point p and rows that contain it
        const double p = aux_randn(rng);
        r.A = {{1.0}, {-1.0}};
        r.b = {p, -p};
        for (int64_t i = 2; i < o.rows; ++i) {
            r.A.push_back({1.0});
            r.b.push_back(p + aux_rand(rng));
        }
        return r;
    }
    // the simplex: the n axes and the diagonal, then random unit rows
    for (std::size_t i = 0; i < n; ++i) {
        Row e(n, 0.0);
        e[i] = 1.0;
        r.A.push_back(e);
        r.b.push_back(1.0);
    }
    r.A.emplace_back(n, -1.0 / std::sqrt(double(n)));
    r.b.push_back(1.0);
    for (int64_t i = o.n + 1; i < o.rows; ++i) {
        r.A.push_back(aux_unit(rng, o.n));
        r.b.push_back(1.0);
    }
    if (o.degenerate) {
        // a random order, then the last two rows opposite
        std::vector<std::size_t> order(r.A.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        for (std::size_t i = order.size(); i > 1; --i)
            std::swap(order[i - 1], order[std::size_t(aux_randi(rng, 0, int64_t(i) - 1))]);
        Rows A;
        Row b;
        for (const std::size_t i : order) {
            A.push_back(r.A[i]);
            b.push_back(r.b[i]);
        }
        r.A = A;
        r.b = b;
        const std::size_t last = r.A.size() - 1;
        for (std::size_t j = 0; j < n; ++j) r.A[last - 1][j] = -r.A[last][j];
        r.b[last - 1] = -r.b[last];
    }
    return r;
}

/// The rows of an unbounded set: in one dimension rows of one direction, in two a wedge or a strip
/// of parallel rows, else unit rows in n - 1 dimensions with one coordinate free.
Rows2 aux_unbounded(Rng &rng, const Options &o) {
    const std::size_t n = std::size_t(o.n), count = std::size_t(o.rows);
    Rows2 r;
    if (o.n == 1) {
        const double s = aux_randn(rng) < 0 ? -1.0 : 1.0;
        for (std::size_t i = 0; i < count; ++i) {
            r.A.push_back({s});
            r.b.push_back(s * aux_rand(rng));
        }
        return r;
    }
    if (o.n == 2 && !o.degenerate) {
        for (std::size_t i = 0; i < count; ++i) {
            const double a = aux_randn(rng);
            r.A.push_back({a, aux_rand(rng)});
            r.b.push_back(aux_rand(rng));
        }
        return r;
    }
    if (o.n == 2) {
        // a strip of parallel rows
        const Row v = aux_unit(rng, 2);
        const double offset = aux_randn(rng);
        r.A = {v, {-v[0], -v[1]}};
        r.b = {offset, -offset};
        for (std::size_t i = 2; i < count; ++i) {
            const double s = aux_randn(rng) < 0 ? -1.0 : 1.0;
            r.A.push_back({s * v[0], s * v[1]});
            r.b.push_back(s * offset + aux_rand(rng));
        }
        return r;
    }
    // unit rows in n - 1 dimensions with a zero column inserted at a random position
    const std::size_t free_column = std::size_t(aux_randi(rng, 1, o.n)) - 1;
    for (std::size_t i = 0; i < count; ++i) {
        Row reduced = aux_unit(rng, o.n - 1);
        reduced.insert(reduced.begin() + std::ptrdiff_t(free_column), 0.0);
        r.A.push_back(reduced);
        r.b.push_back(1.0);
    }
    if (o.degenerate) {
        for (std::size_t j = 0; j < n; ++j) r.A[count - 2][j] = -r.A[count - 1][j];
        r.b[count - 2] = -r.b[count - 1];
    }
    return r;
}

} // namespace


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Polytope Polytope::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                     std::optional<int64_t> nr_constraints,
                                                     std::optional<bool> is_degenerate,
                                                     std::optional<bool> is_bounded) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be a positive integer, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!nr_constraints || *nr_constraints >= 1, Error::wrong_value, [&] {
        return "generate_random: the number of constraints must be a positive integer, got " +
               std::to_string(*nr_constraints);
    });
    aux_check_consistency(dimension, nr_constraints, is_degenerate, is_bounded);
    const Options o = aux_options(rng, dimension, nr_constraints, is_degenerate, is_bounded);
    Rows2 r = o.bounded ? aux_bounded(rng, o) : aux_unbounded(rng, o);
    const int64_t m = int64_t(r.A.size());
    Tensor A = priv_points_tensor(Tensor::zeros({1, 1}), o.n, r.A).matrix_transpose();
    Tensor b = Tensor::from_data(r.b, {m, 1});
    // rotate by the orthogonal factor of a random matrix: A M^-1 = A M'
    if (o.n > 1) {
        std::vector<double> noise(std::size_t(o.n * o.n));
        rng.normal(noise.data(), noise.size(), 1.0);
        const Tensor U = Tensor::from_data(noise, {o.n, o.n}).svd()[0];
        A = A.matmul(U.matrix_transpose());
    }
    // a bounded set is moved by a random point z: A (x - z) <= b
    if (o.bounded) {
        std::vector<double> z(std::size_t(o.n));
        rng.normal(z.data(), z.size(), 1.0);
        b = b + A.matmul(Tensor::from_data(z, {o.n, 1}));
    }
    return {A, b};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
