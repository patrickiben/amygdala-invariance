.PHONY: test smoke run clean

test:
	python -m pytest -q

smoke:
	python -m amyg_inv.cli smoke

run:
	python -m amyg_inv.cli run

clean:
	rm -rf .pytest_cache **/__pycache__ results/*.json results/*.md
