PYTHON ?= .venv/bin/python
.PHONY: check
check:
	$(PYTHON) -m compileall -q release.py lab.py service.py flow.py tests
	$(PYTHON) -m unittest discover -s tests -v
