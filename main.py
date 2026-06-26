from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("Model/DecisionTreeRegressor_model.lb")
DATA_PATH = "Data/laptop_price.csv"


def create_dictionary(column):
    return {value: index + 1 for index, value in enumerate(column.unique())}


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df.drop_duplicates()

    df.drop(columns=["laptop_ID"], inplace=True)

    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    dictionaries = {}

    columns = [
        "Company",
        "Product",
        "ScreenResolution",
        "Memory",
        "Gpu",
        "Cpu",
        "OpSys"
    ]

    for column in columns:
        dictionaries[column] = create_dictionary(df[column])

    dictionaries["TypeName"] = {
        "Ultrabook": 1,
        "Notebook": 2,
        "Netbook": 3,
        "Gaming": 4,
        "2 in 1 Convertible": 5,
        "Workstation": 6
    }

    return df, dictionaries


def get_dropdown_data():
    df, _ = load_data()

    return {
        "Company": sorted(df["Company"].dropna().unique()),
        "Product": sorted(df["Product"].dropna().unique()),
        "Inches": sorted(df["Inches"].dropna().unique(), key=float),
        "Cpu": sorted(df["Cpu"].dropna().unique()),
        "Memory": sorted(df["Memory"].dropna().unique()),
        "Gpu": sorted(df["Gpu"].dropna().unique()),
        "OpSys": sorted(df["OpSys"].dropna().unique()),
        "Weight": sorted(df["Weight"].dropna().unique())
    }


def preprocess_input(form):
    _, dictionaries = load_data()

    def get_number(key):
        return float(form.get(key, 0))

    input_data = pd.DataFrame([{
        "Company": dictionaries["Company"].get(form.get("Company"), 0),
        "Product": dictionaries["Product"].get(form.get("Product"), 0),
        "TypeName": dictionaries["TypeName"]["Ultrabook"],
        "Inches": get_number("Inches"),
        "ScreenResolution": dictionaries["ScreenResolution"].get("1366x768", 1),
        "Cpu": dictionaries["Cpu"].get(form.get("Cpu"), 0),
        "Ram": int(get_number("Ram")),
        "Memory": dictionaries["Memory"].get(form.get("Memory"), 0),
        "Gpu": dictionaries["Gpu"].get(form.get("Gpu"), 0),
        "OpSys": dictionaries["OpSys"].get(form.get("OpSys"), 0),
        "Weight": get_number("Weight")
    }])

    return input_data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict")
def predict():
    return render_template(
        "predict.html",
        suggestions=get_dropdown_data()
    )


@app.route("/prediction", methods=["POST"])
def prediction():
    user_input = preprocess_input(request.form)
    price = model.predict(user_input)[0]

    return render_template(
        "predict.html",
        suggestions=get_dropdown_data(),
        prediction_text=f"Estimated Laptop Price : ₹ {round(price, 2)}"
    )


if __name__ == "__main__":
    app.run(debug=True)