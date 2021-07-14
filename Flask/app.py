import pandas as pd
from tensorflow.keras.models import load_model

from flask import Flask, request, render_template
app = Flask(__name__)

SUBMISSION_DIRECTORY = "./results/"
SUBMISSION_NAME = "submission.csv"
SUBMISSION_PATH = SUBMISSION_DIRECTORY + SUBMISSION_NAME

SCATTER_PLOT_DIRECTORY = "./static/images/scatterplots/"

# The number of top MOA's to be displayed in the output. (MoA stands for Mechanism of Action)
NUMBER_OF_TOP_MOA = 10

# Load the saved model (Deserialization)
model = load_model('neural_network_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    test_features = pd.read_csv(request.files.get('csvfile'))
    test_features.loc[:, 'cp_type'] = test_features.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    test_features.loc[:, 'cp_dose'] = test_features.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    test_id = x_test=test_features['sig_id'].values
    x_test = test_features.drop(['sig_id'], axis='columns', inplace=False)

    y_pred = model.predict(x_test)
    print("Predicted values (y_pred) = ", y_pred)
    submissionColumns = pd.read_csv('submissionColumns.csv')
    column_names = list(submissionColumns.columns)

    id_list= list(test_id)
    y_pred_list= list(y_pred)
    
    dictionary={}
    for column in column_names:
        dictionary[column]=[]
    column_names.remove('sig_id')
    dictionary['sig_id'] = test_id
    for i in range(len(y_pred_list)):
        for j in range(len(column_names)):
            dictionary[column_names[j]].append(y_pred_list[i][j])
    df=pd.DataFrame(dictionary)
    df.to_csv(SUBMISSION_PATH, index=False)

    dfScatterPlot = pd.DataFrame(columns=["ID of the class of drug", "Probability"]) 
    # Class of drugs varies from 1 to len(column_names). 
    # Note that the last number in the range() function is ignored. That is, the number "len(column_names) + 1" is ignored.
    dfScatterPlot["ID of the class of drug"] = [class_id for class_id in range(1, len(column_names) + 1)] 
    
    classOfDrugs = list(df.columns[1:])
    print("All the classes of drugs = ")
    print(classOfDrugs)

    finalListOfMoa = list()
    for index, row in df.iterrows():
        drugId = row[0]
        dfScatterPlot["Probability"] = row[1:].values
        dfSortedProbability = dfScatterPlot.sort_values(by = "Probability", ascending = False) 
        dfTopValues = dfSortedProbability[:NUMBER_OF_TOP_MOA]
        
        topIdOfClassOfDrugs = list(dfTopValues["ID of the class of drug"])
        topClassOfDrugs = [classOfDrugs[class_id - 1] for class_id in topIdOfClassOfDrugs]

        topProbabilities = list(dfTopValues["Probability"])
        ROUND_OFF_DIGITS = 4
        topRoundedProbabilities = [round(probability, ROUND_OFF_DIGITS) for probability in topProbabilities]
        
        listOfTuples = list(zip(topIdOfClassOfDrugs,topClassOfDrugs, topRoundedProbabilities))

        currentMoa = list()
        currentMoa.append(drugId)
        currentMoa.extend(listOfTuples)
        finalListOfMoa.append(currentMoa)

        imageExtension = ".jpeg"
        imageNumber = str(index)
        imageName = "scatterimage" + imageNumber + imageExtension
        SCATTER_PLOT_PATH = SCATTER_PLOT_DIRECTORY + imageName
        
        print("dfTopValues", dfTopValues)
        figure = dfTopValues.plot(  
                                    kind="scatter",
                                    x='ID of the class of drug', 
                                    y='Probability', 
                                    title="Scatter plot for drug id : " + drugId
                                ).get_figure()
        figure.savefig(SCATTER_PLOT_PATH)
        
    print("DF SCATTER PLOT:")
    print(dfScatterPlot)

    print("Final List of MoA values = ", finalListOfMoa)
    return render_template('index.html', 
                            moaList = finalListOfMoa, 
                            NUMBER_OF_TOP_MOA = NUMBER_OF_TOP_MOA,
                            SCATTER_PLOT_DIRECTORY = SCATTER_PLOT_DIRECTORY
                          )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)