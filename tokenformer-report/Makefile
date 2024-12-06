
NAME := main
LATEXMK := latexmk
LATEXMKFLAGS := -xelatex
TEX = $(shell find . -name '*.tex')
BIB = $(shell find . -name '*.bib')
STY = $(shell find . -name '*.sty')
CLS = $(shell find . -name '*.cls')
FIGS = $(shell find . -name '*.png' -o -name '*.jpg' -o -name '*.pdf')

all:${NAME}.pdf

${NAME}.pdf:$(TEX) $(BIB) $(STY) $(CLS) $(FIGS)
	$(LATEXMK) $(LATEXMKFLAGS) $(NAME)
	${MAKE} clean

clean:
	$(LATEXMK) -c $(MAIN)
	rm -f *.bbl *.thm *.xdv *.xml *blx.bib *.snm *.nav

cleanall:
	$(LATEXMK) -C $(MAIN)
	rm -f *.bbl *.thm *.xdv *.xml *blx.bib *.snm *.nav main.pdf

.PHONY: all clean cleanall