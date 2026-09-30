// banner - the CORA START and CORA END blocks that every example prints
//
// The Makefile compiles the examples with `-include global/banner.h`, so no example includes
// it and none can forget it: an inline variable is constructed before main and destroyed after
// it. A program that crashes or ends on an exception prints no END block.
//
// Syntax:   -include global/banner.h -DCORACPP_PROGRAM='"name.cpp"'   (compiler flags)
// Output:   ==== CORA START ==== with `CORA.cpp | <program> | default backend: <name>`;
//           ==== CORA END ==== with `CORA.cpp | <program> | runtime: <seconds>`
// See also: Makefile (the example rule), tensor/tensor.h

#pragma once

#include "tensor/tensor.h"

#include <chrono>
#include <iostream>
#include <string>

#ifndef CORACPP_PROGRAM
#define CORACPP_PROGRAM "program.cpp"
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Prints the START block when constructed and the END block when destroyed.
struct Banner {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    static constexpr std::size_t width = 70;

    Banner() {
        std::cout << block("CORA START") << prefix() << "default backend: " << backend().name()
                  << "\n" << std::string(width, '=') << "\n\n";
    }

    ~Banner() {
        const std::chrono::duration<double> run = std::chrono::steady_clock::now() - start;
        std::cout << "\n" << block("CORA END") << prefix() << "runtime: " << run.count() << " s\n"
                  << std::string(width, '=') << "\n";
    }

    /// "CORA.cpp | <program> | ", the start of the line of both blocks.
    static std::string prefix() { return std::string("CORA.cpp | ") + CORACPP_PROGRAM + " | "; }

    /// A line of `=` of the block's width with the title in its middle.
    static std::string block(const std::string &title) {
        const std::string text = " " + title + " ";
        const std::size_t left = (width - text.size()) / 2;
        return std::string(left, '=') + text + std::string(width - left - text.size(), '=') + "\n";
    }
};

/// One per program: an inline variable is defined once however many files include this.
inline const Banner banner;

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
