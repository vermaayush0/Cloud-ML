from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("C:/Users/ayush/Downloads/Cloud/Iris.csv")

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.drop(columns=['Species'])
y = df['Species']

# Train model
model = RandomForestClassifier()
model.fit(X, y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Convert to DataFrame (fix warning)
        sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=X.columns)

        prediction = model.predict(sample)[0]

        return render_template('index.html', prediction_text=f"🌸 Prediction: {prediction}")

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)