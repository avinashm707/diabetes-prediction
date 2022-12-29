import streamlit as st
#from PIL import Image
import numpy as np
import pickle


st.title("Diabetes Prediction Webb App")

#loading the saved model
loaded_model=pickle.load(open('diabetesmodel.sav','rb'))

#creating a function for preddiction
def diabetes_prediction(input_data):
  #input_data=(5,166,72,19,175,25.8,0.587,51)
  id=np.asarray(input_data)
  input_data_reshaped=id.reshape(1,-1)
  prediction=loaded_model.predict(input_data_reshaped)
  print(prediction)
  if(prediction[0]==0):
    return("person is not diabetic")
  else:
    return("person is not diabetic")



def main():
    
    
    # giving a title
    #st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    