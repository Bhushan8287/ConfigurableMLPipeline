# from flask import Flask, render_template, request
# import joblib
# import pandas as pd

# app=Flask(__name__)

# @app.route("/form", methods=['GET','POST'])
# def form():
#     # loading model and preprocessor 
#     model = joblib.load(r"C:\Users\BW\Desktop\ETL project\ML_project\outputs\RandomForestRegressor_hypertuned.joblib")
#     preprocessor = joblib.load(r"C:\Users\BW\Desktop\ETL project\ML_project\outputs\standard_scaler.joblib")
#     # getting data from user
#     if request.method == 'POST':
#         customers_per_day = int(request.form['number of customers per day'])
#         average_order_value = int(request.form['average order value'])
#         marketing_spend_perday = int(request.form['marketing spend per day'])
#         dict_covert_df = {'Number_of_Customers_Per_Day': [customers_per_day],
#                           'Average_Order_Value': [average_order_value],
#                           'Marketing_Spend_Per_Day': [marketing_spend_perday]
#                           }
#         converted_dataframe = pd.DataFrame(dict_covert_df)
#         transformed_dataframe = preprocessor.transform(converted_dataframe)
#         prediction = model.predict(transformed_dataframe)
#         converted_prediction = float(prediction)
#         converted_prediction = f'{converted_prediction:.2f}'
#         return render_template('result.html', output=converted_prediction)
#     return render_template('form.html')

# if __name__=="__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler once at startup
model = joblib.load(r"C:\Users\BW\Desktop\ETL project\ML_project\outputs\RandomForestRegressor_hypertuned.joblib")
preprocessor = joblib.load(r"C:\Users\BW\Desktop\ETL project\ML_project\outputs\standard_scaler.joblib")

@app.route("/form", methods=['GET', 'POST'])
def form():
    """
    Handle GET/POST requests to display form and return model predictions.
    """
    if request.method == 'POST':
        try:
            # Extract and convert user input
            customers_per_day = int(request.form['number of customers per day'])
            average_order_value = int(request.form['average order value'])
            marketing_spend_perday = int(request.form['marketing spend per day'])

            # Prepare data
            input_data = {
                'Number_of_Customers_Per_Day': [customers_per_day],
                'Average_Order_Value': [average_order_value],
                'Marketing_Spend_Per_Day': [marketing_spend_perday]
            }
            input_df = pd.DataFrame(input_data)
            transformed_input = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(transformed_input)
            formatted_prediction = f"{float(prediction[0]):.2f}"

            return render_template('result.html', output=formatted_prediction)
        except Exception as e:
            return f"Error occurred: {e}"

    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)
