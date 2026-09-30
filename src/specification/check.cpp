// check - whether a set satisfies a specification, as CORA's specification.check
//
// A safe set and an unsafe set of one halfspace are decided by support functions. An unsafe
// polytope (several halfspaces) is avoided iff the set and the polytope have no common point:
// a halfspace that separates them proves it (any set); otherwise a zonotope is decided exactly by
// a linear program, as CORA's isIntersecting, and any other set is reported as touching. So a
// set is never wrongly called safe; a non-zonotope may wrongly be called unsafe.
//
// Syntax:   ok = spec.check(S);   each = spec.holds(S);
// Inputs:   S - any set; a batch of sets gives one answer per member in holds
// Outputs:  ok - whether all members satisfy the specification; each - one flag per member
// See also: Specification::firstViolation, ContSet::supportFunc

#include "contSet/zonotope/zonotope.h"
#include "specification/private/priv.h"
#include "specification/specification.h"

#include <algorithm>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The direction whose support decides a halfspace: a for a safe set (the largest a'x must be at
/// most b), -a for an unsafe one (the smallest a'x, minus the support along -a, must exceed b).
Tensor aux_direction(SpecType type, const Tensor &a) {
    if (type == SpecType::SafeSet) return a;
    if (type == SpecType::UnsafeSet) return a * -1.0;
    throw std::invalid_argument("Specification: unknown type; use SafeSet or UnsafeSet");
}

/// One halfspace: `extreme` is the support in the direction of aux_direction.
bool aux_satisfied(SpecType type, double extreme, double b) {
    if (type == SpecType::SafeSet) return extreme <= b;
    if (type == SpecType::UnsafeSet) return -extreme > b;
    throw std::invalid_argument("Specification: unknown type; use SafeSet or UnsafeSet");
}

/// Per batch element, whether the set is clear of the polytope (the intersection of the
/// halfspaces): separated by one halfspace, else by the linear program for a zonotope.
std::vector<bool> aux_clearOfPolytope(const ContSet &S, const std::vector<Halfspace> &halfspaces) {
    std::vector<bool> clear;
    for (const Halfspace &h : halfspaces) {
        const std::vector<double> extreme =
            S.supportFunc(aux_direction(SpecType::UnsafeSet, h.a)).data();
        if (clear.empty()) clear.assign(extreme.size(), false);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (aux_satisfied(SpecType::UnsafeSet, extreme[i], h.b)) clear[i] = true;
    }
    const auto *Z = dynamic_cast<const Zonotope *>(&S);
    if (!Z) return clear;

    // the members no single halfspace separates are decided exactly
    const std::vector<int64_t> cs = Z->c.shape(), Gs = Z->G.shape();
    const std::size_t n = static_cast<std::size_t>(cs[cs.size() - 2]);
    const std::size_t m = static_cast<std::size_t>(Gs.back());
    const std::vector<double> cData = Z->c.data(), GData = Z->G.data();
    const std::size_t nm = std::max<std::size_t>(1, n * m);
    const std::size_t batchG = std::max<std::size_t>(1, GData.size() / nm);
    std::vector<std::vector<double>> a;
    std::vector<double> b;
    for (const Halfspace &h : halfspaces) {
        a.push_back(h.a.data());
        b.push_back(h.b);
    }
    for (std::size_t i = 0; i < clear.size(); ++i) {
        if (clear[i]) continue;
        const std::vector<double> c(cData.begin() + i * n, cData.begin() + (i + 1) * n);
        const std::size_t gi = (i % batchG) * n * m;
        const std::vector<double> G(GData.begin() + gi, GData.begin() + gi + n * m);
        clear[i] = !priv_zonotopeMeetsPolytope(c, G, static_cast<int>(m), a, b);
    }
    return clear;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<bool> Specification::holds(const ContSet &S) const {
    if (type_ == SpecType::UnsafeSet && halfspaces_.size() > 1)
        return aux_clearOfPolytope(S, halfspaces_);
    std::vector<bool> ok;
    for (const Halfspace &h : halfspaces_) {
        const std::vector<double> extreme = S.supportFunc(aux_direction(type_, h.a)).data();
        if (ok.empty()) ok.assign(extreme.size(), true);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (!aux_satisfied(type_, extreme[i], h.b)) ok[i] = false;
    }
    return ok;
}

bool Specification::check(const ContSet &S) const {
    const std::vector<bool> ok = holds(S);
    return std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
