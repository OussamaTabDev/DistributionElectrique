import joblib
import pandas as pd 

def useModel(classifier_name , regressor_name ):

    classifier = joblib.load(classifier_name)
    regressor = joblib.load(regressor_name)


    new_data = insertData()
    

    predictions = predict_falut(new_data , classifier , regressor)

    print("Prediction for new data:")

    

def insertData():
    return {
        'Voltage (V)': 1220,
        'Current (A)': 150,
        'Power Load (MW)': 170,
        'Temperature (°C)': 25,
        'Wind Speed (km/h)': 0,
        'Weather Condition': 'Clear',
        'Maintenance Status': 'Thunderstorm',
        'Component Health': 'Overheated',
        'Latitude': 48.8566,
        'Longitude': 2.3522
    }
    pass

def predict_falut(data , classifier , regressor):
    result = {}
    input_data = pd.DataFrame([data])
    print(input_data)
    if 'Fault Location (Latitude, Longitude)' in input_data.columns:
        print("it's in here")

    if 'Latitude' in input_data.columns:
         print("it's in here 2")
    
    fault_type = classifier.predict(input_data)[0]
    result['predict Fault Type'] = fault_type

    duration = regressor.predict(input_data)[0]
    result['predict Fault Duration'] = duration

    print(result)



def main():
    useModel('fault_type_classifier.pkl' , 'fault_duration_regressor.pkl')


if __name__ == "__main__":
    main()