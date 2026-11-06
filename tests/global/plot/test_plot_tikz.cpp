// test_plot_tikz - the options of the TikZ export: colors, standalone documents, thinning, tables,
//
// sizes and the shared axes of a group, and what each kind of layer is written as
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/plot/plot.h"
#include "testing.h"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

bool has(const std::string &text, const std::string &what) {
    return text.find(what) != std::string::npos;
}

std::size_t count(const std::string &text, const std::string &what) {
    std::size_t n = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1))
        ++n;
    return n;
}

std::string read(const std::string &path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

Zonotope shifted(double x) { return Zonotope(test::column({x, 0}), Tensor({{1, 0}, {0, 1}})); }

// A figure with every kind of layer: a set, a specification, lines and points.
Figure every_layer() {
    use_cora_colors("CORA:default");
    Figure fig;
    plot(fig, shifted(0), {0, 1}, {.label = "set"});
    plot(fig, Specification::unsafe_set(test::column({-1, 0}), -0.5), {0, 1}, {.label = "unsafe"});
    plot(fig, Tensor({{-1, 0, 1}, {0, 1, 0}}), {0, 1}, {.label = "points"});
    plot(fig, std::vector<Tensor>{Tensor({{-1}, {-1}}), Tensor({{1}, {0.5}})}, {0, 1},
         {.label = "line"});
    return fig;
}

void layers_are_written_as_their_kind() {
    const std::string text = every_layer().tikz();
    check(has(text, "plot filled=CORAcolorBlue"),
          "a set with CORA's blue and 0.2 is a `plot filled`");
    check(has(text, "draw=CORAcolorUnsafe, fill=CORAcolorUnsafeLight"),
          "a region has its two colors");
    check(has(text, "only marks, mark=*"), "points are marks");
    check(has(text, "draw=CORAcolorBlue, fill=CORAcolorBlue, fill opacity=0.2") == false,
          "and not spelled out");
    check(count(text, "\\addplot") == 4, "one \\addplot per layer");
    // one addplot per layer
    for (const char *label : {"set", "unsafe", "points", "line"})
        check(has(text, std::string("\\addlegendentry{") + label + "}"),
              std::string("legend: ") + label);
    // The region reaches the limits of the axes, which are written as options.
    check(has(text, "xmin=") && has(text, "ymax="), "limits are written");
}

void colors_by_name_or_value() {
    const Figure fig = every_layer();
    check(has(fig.tikz(), "CORAcolorBlue") && !has(fig.tikz(), "rgb,1"), "by name in cora-macs");
    const std::string explicit_colors = fig.tikz({.coramacs = false});
    check(!has(explicit_colors, "CORAcolor") &&
              has(explicit_colors, "{rgb,1:red,0;green,0.447;blue,0.741}"),
          "coramacs = false writes RGB values");
    check(has(explicit_colors, "fill opacity=0.2") && has(explicit_colors, "area legend"),
          "with the style spelled out");
    Figure other;
    plot(other, shifted(0), {0, 1}, {.color = Color(0.1, 0.2, 0.3)});
    check(has(other.tikz(), "{rgb,1:red,0.1;green,0.2;blue,0.3}"), "a color that is not CORA's");
}

void standalone_documents() {
    const Figure fig = every_layer();
    const std::string doc = fig.tikz({.standalone = true});
    // a standalone document
    check(doc.rfind("\\documentclass", 0) == 0 && has(doc, "\\begin{document}") &&
              has(doc, "\\end{document}"),
          "a document");
    check(has(doc, "\\definecolor{CORAcolorBlue}{rgb}{0,0.447,0.741}") &&
              has(doc, "\\definecolor{CORAcolorUnsafe}{rgb}{0.9451,0.5529,0.5686}"),
          "the colors it names, as cora-macs defines them");
    check(!has(doc, "CORAcolorYellow"), "and only those");
    check(has(doc, "plot filled/.style"), "with its style");
    check(!has(fig.tikz({.standalone = true, .coramacs = false}), "plot filled/.style"),
          "no style without cora-macs");
    check(!has(doc, "groupplots"), "no group library for one axes");
    check(fig.tikz().rfind('%', 0) == 0 && !has(fig.tikz(), "\\documentclass"), "not a document");
}

void thinning_and_numbers() {
    Figure fig;
    Polygon square;
    for (int i = 0; i < 100; ++i) square.push_back({i * 0.01, 0});
    for (int i = 0; i < 100; ++i) square.push_back({1, i * 0.01});
    for (int i = 0; i < 100; ++i) square.push_back({1 - i * 0.01, 1});
    for (int i = 0; i < 100; ++i) square.push_back({0, 1 - i * 0.01});
    // collinear points are thinned
    fig.add_polygons({square}, Color(0, 0, 0), std::nullopt, 1, "", false);
    check(count(fig.tikz(), "(") == 5, "a square of 400 collinear points is its four corners");
    check(count(fig.tikz({.thin = false}), "(") == 401,
          "without thinning all points and the closing one");
    check(count(fig.tikz({.thin_tolerance = 1e-9}), "(") == 5,
          "collinear points go at any tolerance");
    Figure wavy;
    Polygon curve;
    for (int i = 0; i <= 100; ++i) curve.push_back({i * 0.01, 0.0005 * (i % 2)});
    curve.push_back({1, 1});
    wavy.add_polyline(curve, Color(0, 0, 0), 1, "");
    check(count(wavy.tikz(), "(") < 10 && count(wavy.tikz({.thin_tolerance = 1e-5}), "(") == 102,
          "a ripple within the tolerance goes, a tighter tolerance keeps it");
    check(has(fig.tikz(), "(0,0) (1,0) (1,1) (0,1)") || has(fig.tikz(), "(0,0)"),
          "numbers have no trailing zeros");
}

void size_and_labels_and_sharing() {
    Figure fig = every_layer();
    fig.xlabel = "$x_1$";
    fig.ylabel = "$\\dot{x}$";
    // sizes and labels
    fig.title = "a {title}";
    const std::string text = fig.tikz({.width = "6cm", .height = "0.5\\linewidth"});
    check(has(text, "width=6cm") && has(text, "height=0.5\\linewidth"), "sizes are TeX lengths");
    check(has(text, "xlabel={$x_1$}") && has(text, "ylabel={$\\dot{x}$}") &&
              has(text, "title={a {title}}"),
          "texts pass through as LaTeX");
    FigureGrid grid(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) {
            plot(grid.at(r, c), shifted(r + 3 * c));
            grid.at(r, c).xlabel = "$x$";
            grid.at(r, c).ylabel = "$y$";
        }
    const std::string separate = grid.tikz();
    check(has(separate, "group size=2 by 2") && count(separate, "\\nextgroupplot") == 4 &&
              !has(separate, "descriptions"),
          "a group plot of four");
    check(count(separate, "xlabel=") == 4 && count(separate, "ylabel=") == 4,
          "each axes has its labels");
    grid.share_x = true;
    // shared axes
    grid.share_y = true;
    const std::string shared = grid.tikz();
    check(has(shared, "x descriptions at=edge bottom") &&
              has(shared, "y descriptions at=edge left"),
          "shared axes name them at the edge");
    check(count(shared, "xlabel=") == 2 && count(shared, "ylabel=") == 2,
          "and are labeled once per column and row");
    check(has(shared, "width=5cm") && has(shared, "height=4cm"),
          "a group has a smaller default size");
    check(count(shared, "xmin=-1.1") == 4 || has(shared, "xmin=-1.2"),
          "shared limits are the union");
}

void a_three_dimensional_figure() {
    use_cora_colors("CORA:default");
    Figure fig;
    plot(fig, Zonotope(test::column({0, 0, 0}), Tensor({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})),
         {0, 1, 2}, {.label = "cube"});
    fig.azimuth = -60;
    fig.elevation = 20;
    const std::string text = fig.tikz();
    // the 3-D figure
    check(count(text, "\\addplot3") == 6 && count(text, "forget plot") == 5,
          "six faces, one legend entry");
    check(has(text, "view={-60}{20}") && has(text, "zmin=") && has(text, "\\addlegendentry{cube}"),
          "the view and the z limits");
}

void tables_beside_the_file() {
    Figure fig = every_layer();
    check(test::raises([&] { fig.tikz({.external_data = true}); }, Error::wrong_value),
          "tikz returns text only, so it cannot write tables");
    const std::string path = "test_plot_tikz_output.tikz";
    fig.save_tikz(path, {.external_data = true});
    const std::string text = read(path);
    check(has(text, "table[x=x, y=y] {test_plot_tikz_output_1.dat}") &&
              has(text, "{test_plot_tikz_output_4.dat}") && !has(text, "coordinates"),
          "every layer reads its table");
    check(read("test_plot_tikz_output_1.dat").rfind("x y\n", 0) == 0, "a table has a header");
    for (int k = 1; k <= 4; ++k)
        std::remove(("test_plot_tikz_output_" + std::to_string(k) + ".dat").c_str());
    std::remove(path.c_str());
    check(test::throws([&] { fig.save_tikz("/nonexistent-folder/f.tikz"); }), "an unwritable path");
    // saving grids
    FigureGrid grid(1, 2);
    plot(grid.at(0, 0), shifted(0));
    plot(grid.at(0, 1), shifted(1));
    grid.save_tikz(path, {.standalone = true});
    check(has(read(path), "\\usepgfplotslibrary{groupplots}"),
          "a group document loads its library");
    std::remove(path.c_str());
    ChecksScope off(false);
    check(!fig.tikz().empty(), "with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    layers_are_written_as_their_kind();
    colors_by_name_or_value();
    standalone_documents();
    thinning_and_numbers();
    size_and_labels_and_sharing();
    a_three_dimensional_figure();
    tables_beside_the_file();
    return test::finish("plot tikz");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
