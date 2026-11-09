// check - whether a set satisfies a specification, as CORA's specification.check
//
// A safe set and an unsafe set of one halfspace are decided by support functions. An unsafe
// polytope (several halfspaces) is avoided iff the set and the polytope have no common point:
// a halfspace that separates them proves it (any set); otherwise a zonotope is decided exactly by
// a linear program, as CORA's isIntersecting, and any other set is reported as touching. So a
// set is never wrongly called safe; a non-zonotope may wrongly be called unsafe. A custom
// specification asks its predicate, once for the whole set.
//
// Syntax:   ok = spec.check(S);   each = spec.holds(S);
// Inputs:   S - any set; a batch of sets gives one answer per member in holds
// Outputs:  ok - whether all members satisfy the specification
//           each - one flag per member
// See also: Specification::first_violation, ContSet::support_function
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/specification/specification.h"
#include "global/format.h"
#include "specification/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Halfspaces --------------------------------------------------------------------------------------

/// The direction whose support decides a halfspace: a for a safe set (the largest a'x must be at
/// most b), -a for an unsafe one (the smallest a'x, minus the support along -a, must exceed b).
Tensor aux_direction(SpecType type, const Tensor &a) {
    if (type == SpecType::safe_set) return a;
    if (type == SpecType::unsafe_set) return a * -1.0;
    CORA_FAIL(Error::report_to_dev,
              "Specification: a type that is neither safe_set nor unsafe_set");
}

/// One halfspace: `extreme` is the support in the direction of aux_direction.
bool aux_satisfied(SpecType type, double extreme, double b) {
    if (type == SpecType::safe_set) return extreme <= b;
    if (type == SpecType::unsafe_set) return -extreme > b;
    CORA_FAIL(Error::report_to_dev,
              "Specification: a type that is neither safe_set nor unsafe_set");
}

// Polytopes ---------------------------------------------------------------------------------------

/// Per batch element, whether the set is clear of the polytope (the intersection of the
/// halfspaces): separated by one halfspace, else by the linear program for a zonotope.
std::vector<bool> aux_clear_of_polytope(const ContSet &S,
                                        const std::vector<Halfspace> &halfspaces) {
    std::vector<bool> clear;
    for (const Halfspace &h : halfspaces) {
        const std::vector<double> extreme =
            S.support_function(aux_direction(SpecType::unsafe_set, h.a)).data();
        if (clear.empty()) clear.assign(extreme.size(), false);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (aux_satisfied(SpecType::unsafe_set, extreme[i], h.b)) clear[i] = true;
    }
    if (S.class_id() != ClassId::zonotope) return clear;
    const auto *Z = static_cast<const Zonotope *>(&S);

    // the members no single halfspace separates are decided exactly
    const std::vector<int64_t> cs = Z->center().shape(), Gs = Z->generators().shape();
    const std::size_t n = static_cast<std::size_t>(cs[cs.size() - 2]);
    const std::size_t m = static_cast<std::size_t>(Gs.back());
    const std::vector<double> c_data = Z->center().data(), GData = Z->generators().data();
    const std::size_t nm = std::max<std::size_t>(1, n * m);
    const std::size_t batch_g = std::max<std::size_t>(1, GData.size() / nm);
    std::vector<std::vector<double>> a;
    std::vector<double> b;
    for (const Halfspace &h : halfspaces) {
        a.push_back(h.a.data());
        b.push_back(h.b);
    }
    for (std::size_t i = 0; i < clear.size(); ++i) {
        if (clear[i]) continue;
        const std::vector<double> c(c_data.begin() + i * n, c_data.begin() + (i + 1) * n);
        const std::size_t gi = (i % batch_g) * n * m;
        const std::vector<double> G(GData.begin() + gi, GData.begin() + gi + n * m);
        clear[i] = !priv_zonotope_meets_polytope(c, G, static_cast<int>(m), a, b);
    }
    // the sets that stay clear of the polytope
    return clear;
}

/// The first halfspace whose normal is not a column of the set's dimension, or -1.
int aux_wrong_normal(const ContSet &S, const std::vector<Halfspace> &halfspaces) {
    for (std::size_t k = 0; k < halfspaces.size(); ++k)
        if (!is_column(halfspaces[k].a, S.dim())) return static_cast<int>(k);
    return -1;
}

std::string aux_wrong_normal_message(const ContSet &S, const Halfspace &h) {
    return "check: the set has the dimension " + std::to_string(S.dim()) +
           ", a halfspace normal the shape " + shape_text(h.a.shape());
}

/// Per batch element, whether the specification holds for S (the inputs are valid).
std::vector<bool> aux_holds(const ContSet &S, SpecType type,
                            const std::vector<Halfspace> &halfspaces) {
    // an unsafe polytope: the set must stay clear of it; else each halfspace on its own
    if (type == SpecType::unsafe_set && halfspaces.size() > 1)
        return aux_clear_of_polytope(S, halfspaces);
    std::vector<bool> ok;
    for (const Halfspace &h : halfspaces) {
        const std::vector<double> extreme = S.support_function(aux_direction(type, h.a)).data();
        if (ok.empty()) ok.assign(extreme.size(), true);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (!aux_satisfied(type, extreme[i], h.b)) ok[i] = false;
    }
    return ok;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<bool> Specification::holds(const ContSet &S) const {
    detail::Entry entry;
    if (entry.check()) {
        const int wrong = aux_wrong_normal(S, halfspaces_);
        CORA_CHECK(wrong < 0, Error::dimension_mismatch,
                   [&] { return aux_wrong_normal_message(S, halfspaces_[wrong]); });
    }
    if (type_ == SpecType::custom) return {custom_(S)};
    return aux_holds(S, type_, halfspaces_);
}

bool Specification::check(const ContSet &S) const {
    detail::Entry entry;
    if (entry.check()) {
        const int wrong = aux_wrong_normal(S, halfspaces_);
        CORA_CHECK(wrong < 0, Error::dimension_mismatch,
                   [&] { return aux_wrong_normal_message(S, halfspaces_[wrong]); });
    }
    if (type_ == SpecType::custom) return custom_(S);
    // holds for every member of a batch
    const std::vector<bool> ok = aux_holds(S, type_, halfspaces_);
    return std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
