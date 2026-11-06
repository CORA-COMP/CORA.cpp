// priv_set_type - the set types of CORA's representsa, and whether a type is in a list
//
// Syntax:   bool known = priv_is_set_type(type);   bool in = priv_type_in(type, {"zonotope"});
// Inputs:   type - a name as CORA's representsa takes ("origin", "zonotope", "fullspace", ...)
// Outputs:  known - whether CORA's representsa admits the type
//           in - whether it is in the list
// See also: Interval::representsa, EmptySet::representsa, Fullspace::representsa
//
// CORA.cpp only: CORA lists the admissible types inside representsa.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <initializer_list>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_is_set_type(const std::string &type);

bool priv_type_in(const std::string &type, std::initializer_list<const char *> types);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
