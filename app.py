from flask import Flask, render_template, request
import pandas as pd
import joblib
import os


app = Flask(__name__)


# Load training schema (columns after get_dummies with drop_first=True)
def load_schema_and_model():
	# Reproduce encoding using the same CSV and logic as in the notebook
	data_path = os.path.join(os.path.dirname(__file__), "heart.csv")
	df = pd.read_csv(data_path)
	df_encoded = pd.get_dummies(df, drop_first=True)
	# Separate X columns (exclude target)
	feature_columns = [c for c in df_encoded.columns if c != "HeartDisease"]
	# Load trained model if available; otherwise, set to None
	model_path = os.path.join(os.path.dirname(__file__), "model_decision_tree_heart.joblib")
	model = None
	if os.path.exists(model_path):
		model = joblib.load(model_path)
	return feature_columns, model


FEATURE_COLUMNS, MODEL = load_schema_and_model()


# Helper to encode a single input row matching the training schema
def encode_input(form_dict):
	# Build a single-row DataFrame with original feature names
	# Original features from CSV
	original_cols = [
		"Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
		"RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
	]
	row = {col: form_dict.get(col) for col in original_cols}

	# Cast numerical fields
	for col in ["Age", "RestingBP", "Cholesterol", "MaxHR"]:
		row[col] = int(row[col]) if row[col] not in (None, "") else 0
	row["FastingBS"] = int(row["FastingBS"]) if row["FastingBS"] not in (None, "") else 0
	row["Oldpeak"] = float(row["Oldpeak"]) if row["Oldpeak"] not in (None, "") else 0.0

	# Create DataFrame and apply get_dummies with drop_first=True
	df_input = pd.DataFrame([row])
	df_input_encoded = pd.get_dummies(df_input, drop_first=True)

	# Align columns to FEATURE_COLUMNS (add missing with 0, and ensure correct order)
	for col in FEATURE_COLUMNS:
		if col not in df_input_encoded.columns:
			df_input_encoded[col] = 0
	df_input_encoded = df_input_encoded[FEATURE_COLUMNS]

	return df_input_encoded


@app.route("/")
def index():
	# Provide choices for categorical fields based on CSV unique values
	data_path = os.path.join(os.path.dirname(__file__), "heart.csv")
	df = pd.read_csv(data_path)
	categories = {
		"Sex": sorted(df["Sex"].dropna().unique().tolist()),
		"ChestPainType": sorted(df["ChestPainType"].dropna().unique().tolist()),
		"RestingECG": sorted(df["RestingECG"].dropna().unique().tolist()),
		"ExerciseAngina": sorted(df["ExerciseAngina"].dropna().unique().tolist()),
		"ST_Slope": sorted(df["ST_Slope"].dropna().unique().tolist()),
	}
	return render_template("form.html", categories=categories)


@app.route("/predict", methods=["POST"])
def predict():
	if MODEL is None:
		return render_template("result.html", error="Model tidak ditemukan. Pastikan file model_decision_tree_heart.joblib ada.")

	encoded = encode_input(request.form)
	pred = MODEL.predict(encoded)[0]
	proba = None
	if hasattr(MODEL, "predict_proba"):
		proba = float(MODEL.predict_proba(encoded)[0][1])

	label = "Sakit Jantung" if int(pred) == 1 else "Tidak Sakit Jantung"
	return render_template("result.html", label=label, proba=proba)


if __name__ == "__main__":
	# Use PORT from environment for Railway/Heroku-like platforms
	port = int(os.environ.get("PORT", 5000))
	debug = os.environ.get("FLASK_DEBUG", "0") == "1"
	app.run(host="0.0.0.0", port=port, debug=debug)


