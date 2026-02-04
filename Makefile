.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch pylastmlextension/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ pylastmlextension/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

.PHONY : good
good :
	isort .
	black .
	mypy pylastmlextension/
	pytest -v --color=yes --doctest-modules tests/ pylastmlextension/