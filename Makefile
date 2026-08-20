.PHONY: install test smoke watermark clean

THREAD_ENV = OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

install:
	python -m pip install -e .

test:
	$(THREAD_ENV) pytest -q

smoke:
	bash scripts/run_smoke.sh

watermark:
	bash scripts/run_watermark.sh --pilot-ratios 0.0625 0.125 0.25 --seeds 17 37 57 --skip-existing

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
