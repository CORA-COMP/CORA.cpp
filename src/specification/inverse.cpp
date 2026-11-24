// inverse - the inverse of specifications, as CORA's inverse
//
// Syntax:   specs2 = inverse(specs);
// Inputs:   specs - untimed safe sets, or untimed unsafe sets (not both)
// Outputs:  specs2 - the inverted specifications
// See also: add
//
// CORA: specification/@specification/inverse.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/specification/specification.h"
#include "specification/private/priv_polytope_of.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The complements of the halfspaces of the set, whose union is the complement of the set
/// (CORA's aux_getUnsafeSets).
std::vector<Polytope> aux_unsafe_sets(const Set &S) {
    const Polytope::Constraints C = priv_polytope_of(S).constraints();
    const int64_t n = C.A.shape()[1];
    const std::vector<double> A = C.A.data(), b = C.b.data();
    std::vector<Polytope> out;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const std::vector<double> row(A.begin() + i * n, A.begin() + (i + 1) * n);
        out.push_back(
            Polytope(Tensor::like(C.A, row, {1, n}), Tensor::like(C.b, {b[i]}, {1})).complement());
    }
    return out;
}

/// The unsafe sets that cover the complement of the union of the sets (CORA's safeSet2unsafeSet):
/// the intersections of one complement halfspace of each, without the empty ones.
std::vector<Polytope> aux_safe_to_unsafe(const std::vector<Set> &sets) {
    std::vector<Polytope> F = aux_unsafe_sets(sets[0]);
    for (std::size_t i = 1; i < sets.size(); ++i) {
        std::vector<Polytope> next;
        for (const Polytope &Fj : F)
            for (const Polytope &Fk : aux_unsafe_sets(sets[i])) {
                Polytope P = Fj.intersect(Fk);
                if (!P.representsa("emptySet")) next.push_back(std::move(P));
            }
        F = std::move(next);
    }
    return F;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Specification> inverse(const std::vector<Specification> &specs) {
    detail::Entry entry;
    std::vector<Set> safe, unsafe;
    for (const Specification &spec : specs) {
        CORA_CHECK(!spec.time(), Error::not_supported, [] {
            return std::string("inverse: the inverse of a timed specification is not supported");
        });
        CORA_CHECK(spec.type() == SpecType::safe_set || spec.type() == SpecType::unsafe_set,
                   Error::not_supported, [] {
                       return std::string("inverse: only safe sets and unsafe sets are supported");
                   });
        (spec.type() == SpecType::safe_set ? safe : unsafe).push_back(spec.set());
    }
    CORA_CHECK(safe.empty() || unsafe.empty(), Error::not_supported, [] {
        return std::string("inverse: mixed safe sets and unsafe sets are not supported");
    });
    CORA_CHECK(!safe.empty() || !unsafe.empty(), Error::not_supported,
               [] { return std::string("inverse: give at least one specification"); });
    entry.release(); // the polytope operations check their operands
    if (!safe.empty()) {
        // all the safe sets hold on their intersection: one unsafe set
        Polytope P = priv_polytope_of(safe[0]);
        for (std::size_t i = 1; i < safe.size(); ++i) P = P.intersect(priv_polytope_of(safe[i]));
        return {Specification::unsafe_set(P)};
    }
    if (unsafe.size() == 1) return {Specification::safe_set(unsafe[0])};
    std::vector<Specification> out;
    for (const Polytope &P : aux_safe_to_unsafe(unsafe))
        out.push_back(Specification::unsafe_set(P));
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
