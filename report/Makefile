FILES = $(wildcard *.tex)

comp:
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error report.tex
	bibtex report

figures:
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=tikzfigures tikzfigures/nn.tex 
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=tikzfigures tikzfigures/universal.tex 
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=tikzfigures tikzfigures/forward.tex
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=tikzfigures tikzfigures/backwards.tex
	xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=tikzfigures tikzfigures/nnActivation.tex


clean:
	find . -name "*.aux" -type f -delete
	find . -name "*.log" -type f -delete
	find . -name "*.toc" -type f -delete	
	find . -name "*.bbl" -type f -delete
	find . -name "*.blg" -type f -delete
	find . -name "*.out" -type f -delete
	find . -name "*.fls" -type f -delete
	find . -name "*.fdb_latexmk" -type f -delete
	find . -name "*.synctex.gz" -type f -delete
	find . -name "*.xdv" -type f -delete
	-rm -rr -f _minted-report

