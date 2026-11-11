// complement - the complement of a level set with one inequality, as CORA's levelSet.not
//
// {f <= 0} has the complement {-f < 0} and {f < 0} the complement {-f <= 0}. A set with several
// constraints or an '==' constraint has no such complement (Error::not_supported, as CORA).
//
// Syntax:   LevelSet c = ls.complement();
// Inputs:   -
// Outputs:  c - the level set of the points outside
// See also: split_equality
//
// CORA: contSet/@levelSet/not.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

LevelSet LevelSet::complement() const {
    detail::Entry entry;
    CORA_CHECK(num_equations() == 1 && ops_.front() != CompOp::eq, Error::not_supported, [] {
        return std::string("complement: only a single constraint with '<' or '<=' has a "
                           "complement");
    });
    // the sign of f flips and the strictness swaps
    return {-eq_.front(), dim(), ops_.front() == CompOp::le ? CompOp::lt : CompOp::le};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
