// priv_compact - the removal of redundant halfspaces of a polytope on the host
//
// The function returns true if the set is found to be empty (H is then cleared).
//
// Syntax:   priv_normalize_rows(H);   bool empty = priv_compact_all(H, tol, minimal);
// Inputs:   H - a single polytope in halfspace form
//           tol - the tolerance of "is zero" and "is aligned"
// Outputs:  H - without rows 0 <= b, aligned rows and rows that no point needs
// See also: Polytope::compact, priv_linear_programs
//
// CORA: contSet/@polytope/private/priv_normalizeConstraints.m
// CORA: contSet/@polytope/private/priv_equalityToInequality.m
// CORA: contSet/@polytope/private/priv_compact_zeros.m
// CORA: contSet/@polytope/private/priv_compact_toEquality.m
// CORA: contSet/@polytope/private/priv_compact_alignedIneq.m
// CORA: contSet/@polytope/private/priv_compact_alignedEq.m
// CORA: contSet/@polytope/private/priv_compact_1D.m
// CORA: contSet/@polytope/private/priv_compact_2D.m
// CORA: contSet/@polytope/private/priv_compact_nD.m
// CORA: contSet/@polytope/private/priv_compact_all.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Clears all rows: the set is found to be empty.
void aux_clear(HostH &H) {
    H.A.clear();
    H.b.clear();
    H.Ae.clear();
    H.be.clear();
}

/// Whether x is zero up to MATLAB's default of withinTol, eps.
bool aux_zero(double x) { return within_tol(x, 0.0, machine_eps()); }

/// Whether the two numbers agree up to MATLAB's default of withinTol, eps.
bool aux_same(double a, double b) { return within_tol(a, b, machine_eps()); }

/// The rows of `rows` at the positions where keep is true.
template <class T>
std::vector<T> aux_select(const std::vector<T> &values, const std::vector<bool> &keep) {
    std::vector<T> out;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (keep[i]) out.push_back(values[i]);
    return out;
}

// Small helpers -----------------------------------------------------------------------------------

/// The rank of the 2 x 2 matrix with the columns u and v, by MATLAB's tolerance
/// max(size) eps(largest singular value).
int aux_rank_2x2(const Row &u, const Row &v) {
    const double det = u[0] * v[1] - u[1] * v[0];
    const double frobenius = u[0] * u[0] + u[1] * u[1] + v[0] * v[0] + v[1] * v[1];
    const double s1 = std::sqrt(
        (frobenius + std::sqrt(std::max(frobenius * frobenius - 4.0 * det * det, 0.0))) / 2.0);
    if (s1 == 0.0) return 0;
    const double s2 = std::abs(det) / s1;
    const double singular = 2.0 * (std::nextafter(s1, HUGE_VAL) - s1);
    return (s1 > singular ? 1 : 0) + (s2 > singular ? 1 : 0);
}

/// Whether the rows i and j of the sorted matrix are equal up to a sum of absolute differences
/// below tol.
bool aux_equal_rows(const Row &a, const Row &b, double tol) {
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) sum += std::abs(a[k] - b[k]);
    return sum < tol;
}

// 2D ----------------------------------------------------------------------------------------------

/// Whether the angle from alpha to beta (degrees in [-180, 180]) is at least 180.
bool aux_angles_too_large(double alpha, double beta) {
    alpha += 180.0;
    beta += 180.0;
    const double shifted = beta + (360.0 - alpha);
    return shifted - 360.0 * std::floor(shifted / 360.0) >= 180.0;
}

/// The next triple of indices, wrapping at the end of the list.
void aux_increment(std::size_t &start, std::size_t &middle, std::size_t &end, std::size_t count) {
    start = middle;
    middle = end;
    end = end + 1 < count ? end + 1 : 0;
}

/// The rows of two normalized halfspaces: parallel ones keep the smaller offset, anti-parallel
/// ones that miss each other are empty (CORA's aux_twoHalfspaces); true if empty.
bool aux_two_halfspaces(HostH &H) {
    const double dot = priv_dot(H.A[0], H.A[1]);
    if (aux_same(dot, 1.0)) {
        const std::size_t keep = H.b[1] < H.b[0] ? 1 : 0;
        H.A = {H.A[keep]};
        H.b = {H.b[keep]};
        return false;
    }
    return aux_same(dot, -1.0) && -H.b[1] > H.b[0];
}

// Rows by angle -----------------------------------------------------------------------------------

/// The normalized rows ordered by angle, and the angles in degrees.
std::vector<double> aux_sort_by_angle(HostH &H) {
    std::vector<double> angles(H.A.size());
    for (std::size_t i = 0; i < H.A.size(); ++i)
        angles[i] = std::atan2(H.A[i][1], H.A[i][0]) * 180.0 / kPi;
    std::vector<std::size_t> order(H.A.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return angles[a] < angles[b]; });
    Rows A(H.A.size());
    Row b(H.A.size()), sorted(H.A.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        A[i] = H.A[order[i]];
        b[i] = H.b[order[i]];
        sorted[i] = angles[order[i]];
    }
    H.A = A;
    H.b = b;
    return sorted;
}

/// Drops the parallel rows of the angle-ordered rows, keeping the smallest offset of each group.
void aux_remove_parallel(HostH &H, std::vector<double> &angles) {
    const std::size_t count = H.A.size();
    std::vector<bool> kept(count, true), unchecked(count, true);
    std::size_t start = count - 1, end = 0;
    while (std::find(unchecked.begin(), unchecked.end(), true) != unchecked.end()) {
        if (unchecked[start] && aux_same(priv_dot(H.A[start], H.A[end]), 1.0)) {
            // the parallel group of the start row: keep its smallest offset
            std::vector<bool> parallel(count);
            std::size_t best = count;
            for (std::size_t k = 0; k < count; ++k) {
                parallel[k] = aux_same(priv_dot(H.A[start], H.A[k]), 1.0);
                if (parallel[k] && (best == count || H.b[k] < H.b[best])) best = k;
            }
            for (std::size_t k = 0; k < count; ++k) {
                if (parallel[k]) kept[k] = false;
                if (parallel[k]) unchecked[k] = false;
            }
            kept[best] = true;
        }
        unchecked[start] = false;
        start = end;
        ++end;
    }
    H.A = aux_select(H.A, kept);
    H.b = aux_select(H.b, kept);
    angles = aux_select(angles, kept);
}

// nD ----------------------------------------------------------------------------------------------

/// The rows that are aligned with the row i (same normal up to eps), by the products `dot`.
std::vector<bool> aux_aligned(const HostH &H, const Rows &dot, std::size_t i) {
    std::vector<bool> aligned(H.A.size(), true);
    for (std::size_t j = 0; j < H.A.size(); ++j)
        for (std::size_t k = 0; k < H.A[i].size(); ++k)
            if (!aux_zero(H.A[i][k] - dot[j][i] * H.A[i][k])) aligned[j] = false;
    return aligned;
}

/// CORA's aux_irredundantIneq: drops the aligned rows with the larger offsets and marks the rows
/// orthogonal to all others as irredundant; returns the marks of the rows that are left.
std::vector<bool> aux_irredundant(HostH &H) {
    const std::size_t count = H.A.size();
    Rows dot(count, Row(count));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < count; ++j) dot[i][j] = priv_dot(H.A[i], H.A[j]);
    std::vector<bool> irredundant(count, false), redundant(count, false);
    for (std::size_t i = 0; i < count; ++i) {
        const std::vector<bool> aligned = aux_aligned(H, dot, i);
        if (std::count(aligned.begin(), aligned.end(), true) > 1) {
            double smallest = HUGE_VAL;
            for (std::size_t j = 0; j < count; ++j)
                if (aligned[j]) smallest = std::min(smallest, H.b[j]);
            for (std::size_t j = 0; j < count; ++j)
                if (aligned[j] && H.b[j] != smallest) redundant[j] = true;
        }
        // orthogonal to all the rows that are not aligned with it (or anti-parallel)
        bool orthogonal = true;
        for (std::size_t j = 0; j < count; ++j)
            if (!aligned[j] && !aux_zero(dot[j][i])) orthogonal = false;
        if (orthogonal) irredundant[i] = true;
    }
    std::vector<bool> keep(count);
    for (std::size_t i = 0; i < count; ++i) keep[i] = !redundant[i];
    H.A = aux_select(H.A, keep);
    H.b = aux_select(H.b, keep);
    return aux_select(irredundant, keep);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Normalization -----------------------------------------------------------------------------------

void priv_normalize_offsets(HostH &H) {
    // an inequality's offset becomes +1 or -1, an equality's 1; zero stays
    for (std::size_t i = 0; i < H.A.size(); ++i) {
        if (H.b[i] == 0.0) continue;
        const double scale = std::abs(H.b[i]);
        for (double &a : H.A[i]) a /= scale;
        H.b[i] = H.b[i] > 0.0 ? 1.0 : -1.0;
    }
    // and each equality's offset 1
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        if (H.be[i] == 0.0) continue;
        for (double &a : H.Ae[i]) a /= H.be[i];
        H.be[i] = 1.0;
    }
}

void priv_normalize_rows(HostH &H) {
    // every row of nonzero length gets length 1, its offset scaled alike
    for (std::size_t i = 0; i < H.A.size(); ++i) {
        const double norm = priv_norm(H.A[i]);
        if (aux_zero(norm)) continue;
        for (double &a : H.A[i]) a /= norm;
        H.b[i] /= norm;
    }
    // the equalities too
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        const double norm = priv_norm(H.Ae[i]);
        if (aux_zero(norm)) continue;
        for (double &a : H.Ae[i]) a /= norm;
        H.be[i] /= norm;
    }
}

std::pair<Rows, Row> priv_equality_to_inequality(const HostH &H) {
    Rows A = H.A;
    Row b = H.b;
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        A.push_back(H.Ae[i]);
        b.push_back(H.be[i]);
    }
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        Row negated = H.Ae[i];
        for (double &a : negated) a = -a;
        A.push_back(negated);
        b.push_back(-H.be[i]);
    }
    // the rows as A x <= b
    return {A, b};
}

// Rows that say nothing, rows that repeat ---------------------------------------------------------

bool priv_compact_zeros(HostH &H, double tol) {
    const auto is_zero_row = [](const Row &a) {
        return std::all_of(a.begin(), a.end(), [](double x) { return aux_zero(x); });
    };
    std::vector<bool> keep(H.A.size(), true), keep_eq(H.Ae.size(), true);
    bool any_zero = false, any_zero_eq = false;
    for (std::size_t i = 0; i < H.A.size(); ++i)
        if (is_zero_row(H.A[i])) {
            keep[i] = false;
            any_zero = true;
            // 0 <= b with b < 0 has no solution
            if (H.b[i] < 0.0 && !within_tol(H.b[i], 0.0, tol)) {
                aux_clear(H);
                return true;
            }
        }
    for (std::size_t i = 0; i < H.Ae.size(); ++i)
        if (is_zero_row(H.Ae[i])) {
            keep_eq[i] = false;
            any_zero_eq = true;
            // 0 = be with be != 0 has no solution
            if (!within_tol(H.be[i], 0.0, tol)) {
                aux_clear(H);
                return true;
            }
        }
    if (any_zero) {
        H.A = aux_select(H.A, keep);
        H.b = aux_select(H.b, keep);
    }
    if (any_zero_eq) {
        H.Ae = aux_select(H.Ae, keep_eq);
        H.be = aux_select(H.be, keep_eq);
    }
    return false;
}

void priv_compact_to_equality(HostH &H, double tol) {
    const std::size_t count = H.A.size();
    std::vector<bool> moved(count, false);
    // the rows with their offsets, a x <= c as (a, c)
    Rows joined(count);
    for (std::size_t i = 0; i < count; ++i) {
        joined[i] = H.A[i];
        joined[i].push_back(H.b[i]);
    }
    // a row with a partner of opposite sign becomes an equality, the partner goes
    for (std::size_t i = 0; i < count; ++i) {
        if (moved[i]) continue;
        bool any_inverted = false;
        for (std::size_t j = 0; j < count; ++j) {
            bool inverted = true;
            for (std::size_t k = 0; k < joined[i].size(); ++k)
                if (!within_tol(joined[j][k] + joined[i][k], 0.0, tol)) inverted = false;
            if (inverted) {
                moved[j] = true;
                any_inverted = true;
            }
        }
        if (any_inverted) {
            moved[i] = true;
            H.Ae.push_back(H.A[i]);
            H.be.push_back(H.b[i]);
        }
    }
    // the rows that became equalities are no inequalities any more
    std::vector<bool> keep(count);
    for (std::size_t i = 0; i < count; ++i) keep[i] = !moved[i];
    H.A = aux_select(H.A, keep);
    H.b = aux_select(H.b, keep);
}

// Rows that repeat --------------------------------------------------------------------------------

void priv_compact_aligned_inequalities(HostH &H, double tol) {
    const std::size_t count = H.A.size();
    // sortrows: ascending by the first column, then the second, and so on; ties keep their order
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return H.A[a] < H.A[b]; });
    // the rows in that order, with their offsets
    Rows A(count);
    Row b(count);
    for (std::size_t i = 0; i < count; ++i) {
        A[i] = H.A[order[i]];
        b[i] = H.b[order[i]];
    }
    // runs of equal rows give one row with the smallest offset
    Rows out;
    Row out_b;
    std::size_t counter = 0;
    while (counter + 1 < count) {
        if (aux_equal_rows(A[counter], A[counter + 1], tol)) {
            const Row &a = A[counter];
            double smallest = std::min(b[counter], b[counter + 1]);
            counter += 2;
            while (counter < count && aux_equal_rows(A[counter], a, tol)) {
                smallest = std::min(smallest, b[counter]);
                ++counter;
            }
            out.push_back(a);
            out_b.push_back(smallest);
        } else {
            // a row of its own
            out.push_back(A[counter]);
            out_b.push_back(b[counter]);
            ++counter;
        }
    }
    // the last row, if the loop stopped in front of it
    if (count > 0 && counter == count - 1) {
        out.push_back(A.back());
        out_b.push_back(b.back());
    }
    H.A = out;
    H.b = out_b;
}

bool priv_compact_aligned_equalities(HostH &H) {
    const std::size_t count = H.Ae.size();
    Rows dot(count, Row(count));
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < count; ++j) dot[i][j] = priv_dot(H.Ae[i], H.Ae[j]);
    std::vector<bool> irredundant(count, true);
    for (std::size_t i = 0; i < count; ++i) {
        if (!irredundant[i]) continue;
        std::vector<bool> aligned(count, true);
        for (std::size_t j = 0; j < count; ++j)
            for (std::size_t k = 0; k < H.Ae[i].size(); ++k)
                if (!aux_zero(H.Ae[i][k] - dot[j][i] * H.Ae[i][k])) aligned[j] = false;
        if (std::count(aligned.begin(), aligned.end(), true) <= 1) continue;
        // aligned equalities must agree in the offset, else no point satisfies both
        for (std::size_t j = 0; j < count; ++j)
            if (aligned[j] && !aux_same(H.be[j], H.be[i])) {
                aux_clear(H);
                return true;
            }
        for (std::size_t j = 0; j < count; ++j)
            if (aligned[j]) irredundant[j] = false;
        irredundant[i] = true;
    }
    H.Ae = aux_select(H.Ae, irredundant);
    H.be = aux_select(H.be, irredundant);
    return false;
}

// One dimension -----------------------------------------------------------------------------------

bool priv_compact_1d(HostH &H, double tol) {
    if (!H.Ae.empty()) {
        // equalities become a x = 1 or a x = 0
        priv_normalize_offsets(H);
        if (H.Ae.size() > 1) {
            std::vector<double> ones, zeros;
            bool near_one = false, near_zero = false;
            for (std::size_t i = 0; i < H.Ae.size(); ++i) {
                near_one = near_one || within_tol(H.be[i], 1.0, tol);
                near_zero = near_zero || within_tol(H.be[i], 0.0, tol);
                if (aux_same(H.be[i], 1.0)) ones.push_back(H.Ae[i][0]);
                if (aux_same(H.be[i], 0.0)) zeros.push_back(H.Ae[i][0]);
            }
            // x = 0 and x = c cannot both hold
            const auto disagree = [&](const std::vector<double> &a) {
                return a.size() > 1 && !std::all_of(a.begin(), a.end(), [&](double x) {
                           return within_tol(x, a[0], tol);
                       });
            };
            if ((near_one && near_zero) || disagree(ones) || disagree(zeros)) {
                aux_clear(H);
                return true;
            }
            // all equalities are the first: keep it as x = be / a (CORA keeps x = be)
            const double a = H.Ae[0][0], offset = H.be[0] / a;
            H.Ae = {{1.0}};
            H.be = {offset};
        }
    }
    if (H.A.empty()) return false;
    // the unit rows x <= b and -x <= b with the smallest offsets
    Rows equalities = H.Ae;
    Row offsets = H.be;
    HostH normalized = H;
    priv_normalize_rows(normalized);
    // be_ = be / |Ae| of CORA's normalization of the equalities too
    for (std::size_t i = 0; i < equalities.size(); ++i)
        offsets[i] = H.be[i] / std::abs(equalities[i][0]);
    bool has_upper = false, has_lower = false;
    double upper = HUGE_VAL, lower = HUGE_VAL;
    for (std::size_t i = 0; i < normalized.A.size(); ++i) {
        if (normalized.A[i][0] > 0.0) {
            has_upper = true;
            upper = std::min(upper, normalized.b[i]);
        } else {
            has_lower = true;
            lower = std::min(lower, normalized.b[i]);
        }
    }
    if (has_upper != has_lower) {
        H.A = {{has_upper ? 1.0 : -1.0}};
        H.b = {has_upper ? upper : lower};
        return false;
    }
    // x <= upper and -x <= lower: x >= -lower
    if (upper < -lower) {
        aux_clear(H);
        return true;
    }
    H.A = {{1.0}, {-1.0}};
    H.b = {upper, lower};
    if (!H.Ae.empty()) {
        for (const double value : offsets)
            if (value > upper || value < -lower) {
                aux_clear(H);
                return true;
            }
        // the equality alone satisfies the rows
        H.A.clear();
        H.b.clear();
    }
    return false;
}

// Two dimensions ----------------------------------------------------------------------------------

bool priv_compact_2d(HostH &H) {
    if (!H.Ae.empty() && priv_compact_aligned_equalities(H)) {
        aux_clear(H);
        return true;
    }
    if (H.A.empty()) return false;
    std::vector<double> angles = aux_sort_by_angle(H);
    aux_remove_parallel(H, angles);
    std::size_t count = H.A.size();
    if (count == 1) return false;
    if (count == 2) {
        const bool empty = aux_two_halfspaces(H);
        if (empty) aux_clear(H);
        return empty;
    }
    // the rows turned by 90 degrees counter-clockwise
    Rows turned(count);
    for (std::size_t i = 0; i < count; ++i) turned[i] = {H.A[i][1], -H.A[i][0]};
    std::vector<bool> kept(count, true), unchecked(count, true);
    std::size_t start = 0, middle = 1, end = 2;
    while (std::find(unchecked.begin(), unchecked.end(), true) != unchecked.end()) {
        unchecked[middle] = false;
        // the rows around a middle one may be at most 180 degrees apart
        if (aux_angles_too_large(angles[start], angles[end])) {
            aux_increment(start, middle, end, count);
            continue;
        }
        // the point where the lines of the outer rows meet, and the middle row at that point
        const Row p1 = {H.A[start][0] * H.b[start], H.A[start][1] * H.b[start]};
        const Row p2 = {H.A[end][0] * H.b[end], H.A[end][1] * H.b[end]};
        const Row &p1vec = turned[start], p2vec = {-turned[end][0], -turned[end][1]};
        if (aux_rank_2x2(p1vec, p2vec) < 2) {
            aux_increment(start, middle, end, count);
            continue;
        }
        const Row factors = *priv_solve({{p1vec[0], p2vec[0]}, {p1vec[1], p2vec[1]}},
                                        {p2[0] - p1[0], p2[1] - p1[1]}, 0.0);
        const Row x = {p1[0] + factors[0] * turned[start][0],
                       p1[1] + factors[0] * turned[start][1]};
        const double value = priv_dot(H.A[middle], x);
        if (H.b[middle] > value || within_tol(H.b[middle], value, numerics::kPolytopeMiddle)) {
            // the middle row is redundant
            kept[middle] = false;
            middle = end;
            end = end + 1 < count ? end + 1 : 0;
        } else {
            aux_increment(start, middle, end, count);
        }
    }
    H.A = aux_select(H.A, kept);
    H.b = aux_select(H.b, kept);
    return false;
}

// Several dimensions ------------------------------------------------------------------------------

bool priv_compact_nd(HostH &H, double tol) {
    if (H.b.empty()) return false;
    std::vector<bool> irredundant = aux_irredundant(H);
    if (std::all_of(irredundant.begin(), irredundant.end(), [](bool x) { return x; })) return false;
    if (priv_is_empty(H)) {
        aux_clear(H);
        return true;
    }
    // a row is redundant if the set without it stays below its offset along it
    const std::size_t count = H.A.size();
    std::vector<bool> keep(count, true);
    for (std::size_t i = 0; i < count; ++i) {
        if (irredundant[i]) continue;
        keep[i] = false;
        HostH rest = H;
        rest.A = aux_select(H.A, keep);
        rest.b = aux_select(H.b, keep);
        const Support s = priv_support(rest, H.A[i], true);
        keep[i] = true;
        if (s.status == Support::Status::infeasible) {
            aux_clear(H);
            return true;
        }
        if (s.status == Support::Status::optimal &&
            (s.value < H.b[i] || within_tol(s.value, H.b[i], tol))) {
            keep[i] = false;
            // the rows active at the optimal point are needed
            for (std::size_t j = 0; j < count; ++j)
                if (within_tol(priv_dot(H.A[j], s.x), H.b[j], tol)) irredundant[j] = true;
        }
    }
    H.A = aux_select(H.A, keep);
    H.b = aux_select(H.b, keep);
    return false;
}

bool priv_compact_all(HostH &H, double tol, bool &minimal) {
    minimal = false;
    if (priv_compact_zeros(H, tol)) return true;
    // equalities can only be redundant if they are aligned
    priv_compact_to_equality(H, tol);
    if (priv_compact_aligned_equalities(H)) return true;
    priv_compact_aligned_inequalities(H, tol);
    const bool empty = H.n == 1   ? priv_compact_1d(H, tol)
                       : H.n == 2 ? priv_compact_2d(H)
                                  : priv_compact_nd(H, tol);
    minimal = true;
    return empty;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
