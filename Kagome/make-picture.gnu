#!/usr/bin/env gnuplot

#
# Creates a pretty picture of the model.
#

if (ARGC < 1) {
    print "Usage: gnuplot -c make-lattice.gnu <N>"
    exit 1
}

N = ARG1 + 0

set terminal pngcairo enhanced
set output sprintf("Kagome-%i.png", N)
set title sprintf("J_1-J_2 model on %i-site Kagome lattice", N)
set border 0
unset xtics
unset ytics

positions = sprintf("<(python3 -c 'from kagome import *; Kagome%i.positions_for_gnuplot()')", N)
solid_edges(type) = sprintf("<(python3 -c 'from kagome import *; Kagome%i.solid_edges_for_gnuplot(\"%s\")')", N, type)
dashed_edges(type) = sprintf("<(python3 -c 'from kagome import *; Kagome%i.dashed_edges_for_gnuplot(\"%s\")')", N, type)

plot \
    solid_edges("j1") w l lt 1 lw 3 lc rgb "#000000" notitle, \
    solid_edges("j2") w l lt 1 lw 3 lc rgb "#CCCCCC" notitle, \
    dashed_edges("j1") w l lt 0 dt 2 lw 3 lc rgb "#000000" notitle, \
    dashed_edges("j2") w l lt 0 dt 2 lw 3 lc rgb "#808080" notitle, \
    positions w p pt 7 ps 1 lc rgb "#000000" notitle

set output
