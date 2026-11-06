// server - the warm daemon; see server.cpp for the protocol
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <iosfwd>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::comp {

/// The daemon's localhost port.
int port();

/// Warms up, then serves until killed. Never returns normally.
int serve(const std::string &srv_dir);

/// One instance; any failure is the verdict `error`, with the reason in `log`.
std::string run_logged(const std::string &params, const std::string &results_file,
                       std::ostream &log);

} // namespace cora::comp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
