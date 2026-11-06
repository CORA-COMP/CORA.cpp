// test_ellipsoid_display - Ellipsoid::display: the text of an ellipsoid
//
// CORA: contSet/@ellipsoid/display.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

#include <sstream>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
        const std::string text = E.display();
        check(text.find("ellipsoid") != std::string::npos &&
                  text.find("dimension: 2") != std::string::npos,
              b + ": the class and the dimension");
        check(text.find("center") != std::string::npos &&
                  text.find("shape matrix") != std::string::npos,
              b + ": the center and the shape matrix");
        std::ostringstream stream;
        stream << E;
        check(stream.str() == text, b + ": operator<< prints the same");
    });
    // no input to get wrong; the same with the checks off
    const Ellipsoid E(Tensor({{1.0}}));
    const std::string on = E.display();
    ChecksScope off(false);
    check(E.display() == on, "display: the same with the checks off");
    return test::finish("ellipsoid display");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
