// display - the text that describes a zonotope bundle, as CORA's zonoBundle.display
//
// Syntax:   std::string text = zB.display();   std::cout << zB;
// Inputs:   zB - zonotope bundle
// Outputs:  text - its dimension, the number of members and every member as Zonotope::display
// See also: Zonotope::display, src/global/format.h
//
// CORA: contSet/@zonoBundle/display.m

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string ZonoBundle::display_impl() const {
    std::string text = "zonotope bundle\n";
    text += "- dimension: " + std::to_string(dim()) + "\n";
    text += "- zonotopes: " + std::to_string(parallel_sets()) + "\n";
    for (std::size_t i = 0; i < Z_.size(); ++i) {
        // the member's own text, indented under its number
        text += "- zonotope " + std::to_string(i + 1) + ":\n";
        const std::string member = Z_[i].display();
        for (std::size_t start = 0; start < member.size();) {
            const std::size_t end = member.find('\n', start);
            text += "    " + member.substr(start, end - start) + "\n";
            start = end == std::string::npos ? member.size() : end + 1;
        }
    }
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
