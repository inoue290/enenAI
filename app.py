from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# モデル読み込み（起動時に一度だけ）
model = load_model('atari_model.keras')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    input_value = None
    if request.method == 'POST':
        try:
            input_value = float(request.form['input_atari'])
            data = np.array([[input_value]])
            pred = model.predict(data)[0][0]
            prediction = round(pred)
        except Exception as e:
            prediction = f"エラー: {e}"

    return render_template('index.html', prediction=prediction, input_value=input_value)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
