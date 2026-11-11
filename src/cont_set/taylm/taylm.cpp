// taylm - the factories of a Taylor model, checked, as CORA's taylm(int) and taylm(zonotope)
//
// A box gives one variable per coordinate (mid + rad a), a zonotope one variable per generator.
// The models of one call share the variables, so that x - x is 0 and not an interval.
//
// Syntax:   x = Taylm::from_interval(I, max_order);   x = Taylm::from_zonotope(Z, max_order);
//           c = Taylm::constant(value, vars, max_order);
// Inputs:   I - a bounded box (n, 1)
//           Z - a zonotope
//           max_order - the degree of the polynomial
// Outputs:  x - one model per coordinate
// See also: evaluate, interval
//
// CORA: contSet/@taylm/taylm.m
// CORA: contSet/@taylm/dim.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The monomial with exponent 1 at variable i and 0 elsewhere.
Taylm::Monomial aux_unit(int64_t vars, int64_t i) {
    Taylm::Monomial m(std::size_t(vars), 0);
    m[std::size_t(i)] = 1;
    return m;
}

void aux_check_order(int max_order, const char *operation) {
    detail::Validation entry;
    CORA_CHECK(max_order >= 1, Error::wrong_value, [&] {
        return std::string(operation) + ": max_order must be at least 1, got " +
               std::to_string(max_order);
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constant -----------------------------------------------------------------------------------

Taylm Taylm::constant(double c, int64_t vars, int max_order) {
    detail::Entry entry;
    aux_check_order(max_order, "Taylm::constant");
    CORA_CHECK(vars >= 0, Error::wrong_value, [] {
        return std::string("Taylm::constant: the number of variables must be nonnegative");
    });
    Taylm t(vars, max_order);
    if (c != 0.0) t.terms_[Monomial(std::size_t(vars), 0)] = c;
    return t;
}

// The range ---------------------------------------------------------------------------------------

Taylm Taylm::from_range(double lo, double hi, int64_t vars, int max_order) {
    detail::Entry entry;
    CORA_CHECK(lo <= hi, Error::wrong_value, [] {
        return std::string("Taylm::from_range: the lower bound must not exceed the upper one");
    });
    // the midpoint is exact, the half-width is the remainder
    Taylm t = constant(0.5 * (lo + hi), vars, max_order);
    t.rem_lo_ = -0.5 * (hi - lo);
    t.rem_hi_ = 0.5 * (hi - lo);
    return t;
}

// The box and the zonotope ------------------------------------------------------------------------

std::vector<Taylm> Taylm::from_interval(const Interval &I, int max_order) {
    detail::Entry entry;
    aux_check_order(max_order, "Taylm::from_interval");
    CORA_CHECK(I.is_bounded() && !I.is_empty() && I.inf().ndim() == 2 && I.inf().shape()[1] == 1,
               Error::wrong_value, [] {
                   return std::string("Taylm::from_interval: the box must be one bounded column");
               });
    // one variable per coordinate of the box
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    const auto n = int64_t(lo.size());
    std::vector<Taylm> out;
    for (int64_t i = 0; i < n; ++i) {
        Taylm t(n, max_order);
        // x_i = mid + rad a_i, with the constant and the linear term
        const double mid = 0.5 * (lo[std::size_t(i)] + hi[std::size_t(i)]);
        const double rad = 0.5 * (hi[std::size_t(i)] - lo[std::size_t(i)]);
        if (mid != 0.0) t.terms_[Monomial(std::size_t(n), 0)] = mid;
        if (rad != 0.0) t.terms_[aux_unit(n, i)] = rad;
        out.push_back(std::move(t));
    }
    return out;
}

Taylm Taylm::embedded(int64_t before, int64_t after) const {
    Taylm out(before + vars_ + after, max_order_);
    // every monomial gets zero exponents for the variables around
    for (const auto &[m, c] : terms_) {
        Monomial wide(std::size_t(before), 0);
        wide.insert(wide.end(), m.begin(), m.end());
        wide.resize(std::size_t(before + vars_ + after), 0);
        out.terms_[wide] = c;
    }
    out.rem_lo_ = rem_lo_;
    out.rem_hi_ = rem_hi_;
    return out;
}

// The state and the input -------------------------------------------------------------------------

std::vector<Taylm> Taylm::from_zonotope_and_box(const Zonotope &Z, const Interval &U,
                                                int max_order) {
    detail::Entry entry;
    std::vector<Taylm> x = from_zonotope(Z, max_order), u = from_interval(U, max_order);
    // the state keeps the first variables, the input the last ones
    const int64_t p = x.front().vars_, m = u.front().vars_;
    std::vector<Taylm> out;
    out.reserve(x.size() + u.size());
    for (const Taylm &t : x) out.push_back(t.embedded(0, m));
    for (const Taylm &t : u) out.push_back(t.embedded(p, 0));
    return out;
}

std::vector<Taylm> Taylm::from_zonotope(const Zonotope &Z, int max_order) {
    detail::Entry entry;
    aux_check_order(max_order, "Taylm::from_zonotope");
    CORA_CHECK(Z.center().ndim() == 2, Error::wrong_value, [] {
        return std::string("Taylm::from_zonotope: the zonotope must not be a batch");
    });
    // one variable per generator, shared by all coordinates
    const std::vector<double> c = Z.center().data(), G = Z.generators().data();
    const int64_t n = Z.dim(), p = Z.generators().shape().back();
    std::vector<Taylm> out;
    for (int64_t i = 0; i < n; ++i) {
        Taylm t(p, max_order);
        if (c[std::size_t(i)] != 0.0) t.terms_[Monomial(std::size_t(p), 0)] = c[std::size_t(i)];
        for (int64_t j = 0; j < p; ++j)
            if (G[std::size_t(i * p + j)] != 0.0)
                t.terms_[aux_unit(p, j)] = G[std::size_t(i * p + j)];
        out.push_back(std::move(t));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
