add_requires("opencv")
add_rules("mode.release", "mode.debug")
set_languages("cxx17")

target("GeneticPainter")
    set_kind("binary")
    add_packages("opencv")
    add_files("draw.cpp")
    set_targetdir("bin")