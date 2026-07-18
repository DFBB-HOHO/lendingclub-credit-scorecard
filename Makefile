.PHONY: install data train test report

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python scripts/train_scorecard.py

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

report:
	python scripts/build_report_docx.py

