// snapshot - compares the layers or the TikZ text of a figure with a stored snapshot
//
// The layers are dumped as text with every number in `%.9g`, and compared number by number
// within a tolerance, so that rounding does not break the test but a changed geometry does; TikZ
// text is compared exactly. Run the tests with CORACPP_UPDATE_SNAPSHOTS=1 to write the snapshots
// again after a change that is meant, and look at the difference with git before committing.
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/plot/figure.h"
#include "testing.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The path of the snapshot `name` in the folder snapshots/ next to the test source `source`
/// (pass __FILE__).
inline std::string snapshot_path(const std::string &source, const std::string &name) {
    return source.substr(0, source.find_last_of("/\\") + 1) + "snapshots/" + name;
}

/// The layers of the figure as text: one line per layer, then one per polygon or face.
inline std::string dump_layers(const cora::Figure &fig) {
    std::ostringstream out;
    out.precision(9);
    out << "dimension " << fig.dimension << "\n";
    for (const cora::Figure::Layer &l : fig.layers()) {
        static const char *kinds[] = {"polygons", "polyline", "points", "region", "surface"};
        out << kinds[static_cast<int>(l.kind)] << " edge " << l.edge.hex() << " face "
            << l.face.hex() << " filled " << l.filled << " alpha " << l.face_alpha << " width "
            << l.line_width << " zorder " << l.zorder << " label '" << l.label << "'\n";
        for (const cora::Polygon &p : l.polygons) {
            out << "  polygon";
            for (const cora::Point &v : p) out << " " << v[0] << " " << v[1];
            out << "\n";
        }
        for (const std::vector<cora::Point3> &f : l.faces) {
            out << "  face";
            for (const cora::Point3 &v : f) out << " " << v[0] << " " << v[1] << " " << v[2];
            out << "\n";
        }
    }
    return out.str();
}

/// Whether two texts are the same, numbers within `tol` (relative to 1 + |x|); with tol = 0
/// exactly.
inline bool same_text(const std::string &a, const std::string &b, double tol) {
    if (tol == 0) return a == b;
    std::istringstream x(a), y(b);
    std::string s, t;
    while (true) {
        const bool more_x = static_cast<bool>(x >> s), more_y = static_cast<bool>(y >> t);
        if (more_x != more_y) return false;
        if (!more_x) return true;
        char *end_s = nullptr, *end_t = nullptr;
        const double u = std::strtod(s.c_str(), &end_s), v = std::strtod(t.c_str(), &end_t);
        const bool numbers = *end_s == '\0' && *end_t == '\0' && !s.empty() && !t.empty();
        if (numbers ? !close(u, v, tol) : s != t) return false;
    }
}

/// Checks `text` against the snapshot file `path` (written instead if CORACPP_UPDATE_SNAPSHOTS
/// is set), and reports under `what`.
inline void check_snapshot(const std::string &text, const std::string &path,
                           const std::string &what, double tol = 0) {
    if (std::getenv("CORACPP_UPDATE_SNAPSHOTS")) {
        std::ofstream(path) << text;
        return;
    }
    std::ifstream file(path);
    std::stringstream stored;
    stored << file.rdbuf();
    check(file.good() && same_text(text, stored.str(), tol), what + ": differs from " + path);
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
