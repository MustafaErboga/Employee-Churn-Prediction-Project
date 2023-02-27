import pandas as pd      
import pickle
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Employee Churn Prediction',
    page_icon='employee churn.jpeg'
                  )

st.markdown("<h1 style='text-align: center; color: red;'>Employee Churn Prediction Application</h1>", unsafe_allow_html=True)





_, col2, _ = st.columns([2, 3, 2])

with col2:  
    img = Image.open("employee churn2.jpeg")
    st.image(img, width=300)

st.success('###### Our model is trained using 14,999 samples with the following parameters.')


st.info("###### Satisfaction Level: It is employee satisfaction point, which ranges from 0-1\n"
        "###### Last Evaluation: It is evaluated performance by the employer, which also ranges from 0-1\n"
        "###### Number of Projects: How many of projects assigned to an employee?\n"
        "###### Average Monthly Hours: How many hours in averega an employee worked in a month?\n"
        "###### Time Spent Company: time_spent_company means employee experience. The number of years spent by an employee in the company\n"
        "###### Work Accident: Whether an employee has had a work accident or not\n"
        "###### Promotion Last 5 years: Whether an employee has had a promotion in the last 5 years or not\n"
        "###### Departments: Employee's working department/division\n"
        "###### Salary: Salary level of the employee such as low, medium and high\n"
        "###### Left: Whether the employee has left the company or not")

st.success("###### Information about your employee")
 
    
col, col2 = st.columns([4, 4])
with col:
    st.markdown("###### Select Your Employee's Department")
    departments = st.selectbox("Departments",
            ('sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD')
        )
st.write("Departments:", departments)

with col2:
    st.markdown("###### Select Your Employee's Salary")
    salary = st.radio(
        "Salary",
        ('low', 'medium', 'high')
        )
st.write("Salary:", salary)    
        
col1, col2 = st.columns([4, 4])

with col1:
    st.markdown("#####")
    st.markdown("###### Has the employee had a work accident?")
    work_accident = st.radio(
        "Work Accident",
        ('No', 'Yes')
        )   

    if work_accident == "Yes":   
        work_accident = 1 
    elif work_accident == "No":     
        work_accident = 0

st.write("Work Accident:", work_accident)

with col2:
    st.markdown("#####")
    st.markdown("###### Has the employee been promoted in the last 5 years?")   
    promotion_last_5years = st.radio(
        "Promotion Last 5 years",
        ('Yes', 'No')
        )   
if promotion_last_5years == "Yes":   
    promotion_last_5years = 1 
elif promotion_last_5years == "No":     
    promotion_last_5years = 0

st.write("Promotion Last 5 years:", promotion_last_5years)
    

    
satisfaction_level = st.sidebar.slider("Satisfaction level:", 0.0, 1.0, step=0.1, value=0.5)
st.write("Satisfaction level:", satisfaction_level)
last_evaluation = st.sidebar.slider("Last Evaluation Score:", 0.0, 1.0, step=0.1, value=0.5)
st.write("Last Evaluation Score is:", last_evaluation)
number_project = st.sidebar.slider("Number of Projects:",min_value=1, max_value=10)
st.write("Number of Projects:", number_project)
average_montly_hours = st.sidebar.slider("Monthly Working Hours:",min_value=40, max_value=350)
st.write("Monthly Working Hours:", average_montly_hours)
time_spend_company = st.sidebar.slider("Years in the Company:",min_value=1, max_value=10)
st.write("Years in the Company:", time_spend_company)



my_dict = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'work_accident': work_accident,
    'promotion_last_5years': promotion_last_5years,  
    'departments': departments,
    'salary': salary
}


df=pd.DataFrame.from_dict([my_dict])


models = {}
models['KNN'] = pickle.load(open('knn_final_model', 'rb'))
models['Random_Forest'] = pickle.load(open('RF_final_model', 'rb'))
models['XGBoost'] = pickle.load(open('xgb_final_model', 'rb'))

knn_model = pickle.load(open('knn_final_model', 'rb'))
rf_model = pickle.load(open('RF_final_model', 'rb'))
xgb_model = pickle.load(open('xgb_final_model', 'rb'))

pred_xgb = xgb_final_model.predict(df)
pred_xgb = ['Churn' if pred_xgb == 1 else 'Stayed']
pred_knn = knn_final_model.predict(df)
pred_knn = ['Churn' if pred_knn == 1 else 'Stayed']
pred_rf = RF_final_model.predict(df)
pred_rf = ['Churn' if pred_rf == 1 else 'Stayed']

st.markdown("""<h4 style='text-align:left; color:#d33685;'>Choose Your Model</h4>
""", unsafe_allow_html=True)

model = st.selectbox('Pick one model and get your prediction',
    ['K-Nearest Neighbors', 'Random Forest', 'XGBoost'])

if model == 'K-Nearest Neighbors':
    x1 = pred_knn
elif model == 'Random Forest':
    x1 = pred_rf
elif model == 'XGBoost':
    x1 = pred_xgb

x2 = ['You lost him/her 😭' if x1[0] == 'Left' else 'Relax, s/he is still yours 🥳']

if st.button('Predict'):
    if model == 'K-Nearest Neighbors':
        st.metric('KNN Prediction', value=x2[0])
    elif model == 'K-Nearest Neighbors':
        st.metric('KNN Prediction', value=x2[0])
    elif model == 'Random Forest':
        st.metric('Random Forest Prediction', value=x2[0])

    if x1[0] == 'Left':
        st.image('churn.jpeg', width=500)
    else:
        st.image('not churn.png', width=500)

# To hide Streamlit style
hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)