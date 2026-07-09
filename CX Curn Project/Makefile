# Convenience commands. On Windows, `make` may not be installed by default.
# If `make` is unavailable, just run the command shown after the ':' manually.
#
# Install make on Windows (optional):  choco install make   (in an admin shell)

.PHONY: setup data train mlflow-ui clean

setup:            ## Install Python dependencies
	pip install -r requirements.

data:             ## Generate the synthetic churn dataset
	python -m src.data.generate_dataset

train:            ## Train models + log to MLflow + save best model
	python -m src.training.train

mlflow-ui:        ## Launch the MLflow UI (SQLite backend) at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

registry-check:   ## Load the Production model from the registry and predict
	python -m src.models.load_from_registry

serve:            ## Run the FastAPI app at http://localhost:8000 (docs at /docs)
	uvicorn app.main:app --reload --port 8000

test:             ## Run the pytest suite
	pytest

docker-build:     ## Build the API image
	docker build -t churn-api:latest .

compose-up:       ## Bring up the full stack (mlflow + trainer + api)
	docker compose up --build

compose-down:     ## Stop the stack (add ARGS=-v to wipe volumes)
	docker compose down $(ARGS)

simulate-data:    ## Generate drifted "new" data for the drift/retrain demo
	python -m src.monitoring.simulate

drift:            ## Detect drift between training data and data/raw/churn_new.csv
	python -m src.monitoring.drift --current data/raw/churn_new.csv

retrain:          ## Retrain on new data; promote challenger only if it beats champion
	python -m src.training.retrain --data data/raw/churn_new.csv

rollback:         ## List registered versions (add TO=<v> to roll back)
	python -m src.models.rollback $(if $(TO),--to $(TO),)

clean:            ## Remove generated artifacts (keeps code)
	rm -rf mlruns models/best_model.pkl models/model_metadata.json data/raw/churn.csv
