import pickle 
import numpy as np

#loading the saved model
loaded_model=pickle.load(open('diabetesmodel.sav','rb'))


input_data=(5,166,72,19,175,25.8,0.587,51)
#changing to numpy array
id=np.asarray(input_data)
#reshaping the array as we predicting for one instance
input_data_reshaped=id.reshape(1,-1)
#input_data
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("person is not diabetic")
else:
    print("person is  diabetic")