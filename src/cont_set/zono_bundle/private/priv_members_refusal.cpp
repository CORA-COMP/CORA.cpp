// priv_members_refusal - what is wrong with the members of a bundle or with its operands
//
// Syntax:   Refusal r = priv_members_refusal(zonotopes);   Refusal r = priv_operand_refusal(bundle,
// n, like, "intersect"); Inputs:   zonotopes - the members
//           bundle, n, like - a bundle, and the dimension and a tensor of the batch shape of the
//           other operand
// Outputs:  the identifier and text for CORA_CHECK, or nothing
// See also: ZonoBundle::ZonoBundle
//
// CORA.cpp only: the checks of zonoBundle.m (aux_checkInputArgs) and of the operations on bundles
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The batch dimensions of a tensor (..., r, c).
std::vector<int64_t> aux_batch(const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - 2};
}

} // namespace


// ===========================================  MAIN  =========================================== //

detail::Refusal priv_members_refusal(const std::vector<Zonotope> &zonotopes) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_input_in_constructor, "ZonoBundle: " + text});
    };
    if (zonotopes.empty()) return fail("the list of zonotopes must not be empty");
    // every member must share the first one's dimension and batch shape
    const Zonotope &first = zonotopes.front();
    for (const Zonotope &Z : zonotopes) {
        if (Z.dim() != first.dim())
            return fail("Zonotopes have to be embedded in the same affine space.");
        if (aux_batch(Z.center()) != aux_batch(first.center()))
            return fail("the zonotopes must have one batch shape, got " +
                        shape_text(aux_batch(Z.center())) + " and " +
                        shape_text(aux_batch(first.center())));
    }
    return std::nullopt;
}

detail::Refusal priv_operand_refusal(const ZonoBundle &bundle, int64_t n, const Tensor &like,
                                     const char *who, bool same_dimension) {
    // the dimension first, then the batch shape
    const std::string name(who);
    if (same_dimension && n != bundle.dim())
        return std::pair{Error::dimension_mismatch, name + ": the sets differ in dimension, " +
                                                        std::to_string(bundle.dim()) + " and " +
                                                        std::to_string(n)};
    if (aux_batch(like) != aux_batch(bundle.zonotopes().front().center()))
        return std::pair{Error::dimension_mismatch,
                         name + ": the sets must have the same batch shape, got " +
                             shape_text(aux_batch(bundle.zonotopes().front().center())) + " and " +
                             shape_text(aux_batch(like))};
    return std::nullopt;
}

detail::Refusal priv_batch_refusal(const Tensor &like, const char *who) {
    if (like.ndim() == 2) return std::nullopt;
    return std::pair{Error::not_supported, std::string(who) +
                                               ": a linear program per set: it supports a single "
                                               "set, not a batch"};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
