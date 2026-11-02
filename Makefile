.PHONY: install test smoke prepare check-data train watermark clean

PYTHON ?= python
THREAD_ENV = OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

install:
	$(PYTHON) -m pip install -e .

test:
	$(THREAD_ENV) $(PYTHON) -m pytest -q

smoke:
	PYTHON="$(PYTHON)" bash scripts/run_smoke.sh

watermark:
	PYTHON="$(PYTHON)" bash run.sh

prepare:
	PYTHON="$(PYTHON)" bash run.sh --prepare

check-data:
	PYTHON="$(PYTHON)" bash run.sh --check-data

train:
	PYTHON="$(PYTHON)" bash run.sh --background

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
