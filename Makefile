.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-apidoc -f -o docs/source/api ./pylastmlextension
	sphinx-autobuild -b html --watch pylastmlextension/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ pylastmlextension/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
