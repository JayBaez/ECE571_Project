install:
	python -m pip install -r requirements.txt

test:
	pytest -q

fast:
	python run_all.py --data data.xlsx --fast
