// to_string - the text of a formula, as CORA displays it
//
// Syntax:   std::string s = phi.to_string();   std::cout << phi;
// Inputs:   none
// Outputs:  s - e.g. "((x1 < 5) & F[0, 2](x2 >= 0))"
// See also: stl.h
//
// CORA: specification/@stl/disp.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the formula is written without brackets inside & | ~: temporal operators, constants.
bool aux_bare(const Stl &p) {
    const StlKind k = p.kind();
    return k == StlKind::until || k == StlKind::globally || k == StlKind::finally ||
           k == StlKind::next || k == StlKind::true_ || k == StlKind::false_;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::string Stl::to_string() const {
    detail::Entry entry;
    const std::string w =
        kind() == StlKind::comparison || kind() == StlKind::true_ || kind() == StlKind::false_
            ? ""
            : interval().to_string();
    switch (kind()) {
    // the constants and comparisons are written plainly
    case StlKind::true_:
        return "true";
    case StlKind::false_:
        return "false";
    case StlKind::comparison:
        return expr_lhs().to_string() +
               (cmp() == StlCmp::lt   ? " < "
                : cmp() == StlCmp::le ? " <= "
                : cmp() == StlCmp::gt ? " > "
                                      : " >= ") +
               expr_rhs().to_string();
    // the temporal operators put the window after the letter
    case StlKind::until:
        return "(" + lhs().to_string() + " U" + w + " " + rhs().to_string() + ")";
    case StlKind::release:
        return "(" + lhs().to_string() + " R" + w + " " + rhs().to_string() + ")";
    case StlKind::globally:
        return "G" + w + "(" + lhs().to_string() + ")";
    case StlKind::finally:
        return "F" + w + "(" + lhs().to_string() + ")";
    case StlKind::next:
        return "X" + w + "(" + lhs().to_string() + ")";
    // ~ brackets its operand unless that is a temporal operator or a constant
    case StlKind::not_:
        return (aux_bare(lhs()) ? "~" : "~(") + lhs().to_string() + (aux_bare(lhs()) ? "" : ")");
    case StlKind::and_:
    case StlKind::or_: {
        // an operand is bracketed unless it is a temporal operator or a constant
        const auto side = [](const Stl &p) {
            return aux_bare(p) ? p.to_string() : "(" + p.to_string() + ")";
        };
        return "(" + side(lhs()) + (kind() == StlKind::and_ ? " & " : " | ") + side(rhs()) + ")";
    }
    }
    CORA_FAIL(Error::report_to_dev, "to_string: unknown formula kind");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
