import streamlit as st
import pickle
import pandas as pd


def load_model():
    with open('modelop1.pkl', 'rb') as file:
         data = pickle.load(file)
    return data

data = load_model()

prediction = data["model"]

def show_predict_page():
    st.title("Predictor of worker termination")
    st.write("""### Introduce the information of the employee""")
    
   
    MaritalD = (
        "Single",
        "Married",
        "Separated",
        "Widowed",
        
    )
    

    Gender = (
        "M",
        "F",      
    )
    
    Department_E = (
        'Production',
        'IT/IS',
        'Software Engineering',
        'Admin Offices',
        'Sales',
        'Executive Office',
    )
    
    Manager_name= (
        'Michael Albert',
        'Simon Roup',
        'Kissy Sullivan',
        'Elijiah Gray',
        'Webster Butler',
        'Amy Dunn', 
        'Alex Sweetwater',
        'Ketsia Liebig',
        'Brannon Miller', 
        'Peter Monroe', 
        'David Stanley', 
        'Kelley Spirea',
        'Brandon R. LeBlanc',
        'Janet King',
        'John Smith',
        'Jennifer Zamora',
        'Lynn Daneault', 
        'Eric Dougall', 
        'Debra Houlihan',
        'Brian Champaigne',
        'Board of Directors',
    )
    
    Recruitment_Source=(
        'LinkedIn', 
        'Indeed', 
        'Google Search', 
        'Employee Referral',
        'Diversity Job Fair', 
        'On-line Web application', 
        'CareerBuilder',
        'Website',
        'Other',
    )
    Performance_Score=(
        'Exceeds',
        'Fully Meets',
        'Needs Improvement',
        'PIP',
    )
    
    Positions=(
        'Administrative Assistant',          
        'Area Sales Manager',                
        'BI Developer',                      
        'BI Director',                       
        'CIO',                               
        'Data Analyst',                      
        'Data Analyst II',                      
        'Data Architect',                    
        'Database Administrator',            
        'Director of Operations',             
        'Director of Sales',                 
        'Enterprise Architect',              
        'IT Director',                          
        'IT Manager DB',                     
        'IT Manager Infra',                 
        'IT Manager Support',                
        'IT Support',                        
        'Network Engineer',                  
        'President & CEO',                   
        'Principal Data Architect',          
        'Production Manager',               
        'Production Technician I',           
        'Production Technician II',         
        'Sales Manager',                     
        'Senior BI Developer',               
        'Shared Services Manager',           
        'Software Engineer',                 
        'Software Engineering Manager',      
        'Sr Accountant',                     
        'Sr DBA',                            
        'Sr Network Engineer',
    )
    
    
    
    # Enter data for prediction 
      
    MaritalDesc = st.selectbox("Select the Marital Status", MaritalD)
    
    if MaritalDesc=='Single':
           MaritalDesc_Single = 1
    elif MaritalDesc=='Married':
            MaritalDesc_Married = 1
    elif MaritalDesc=='Separated':
            MaritalDesc_Separated = 1
    elif MaritalDesc=='Widowed':
             MaritalDesc_Widowed = 1
              
    Sex = st.selectbox("Select the sex", Gender)
        
    if Sex=='F':
        Sex_M = 1
    else:
        Sex_M = 0
    
    Department = st.selectbox("Select the Department", Department_E)
    Managername = st.selectbox("Select the Manager Name", Manager_name)
    RecruitmentSource = st.selectbox("Select the Recruitment Source", Recruitment_Source)
    Position = st.selectbox("Select the Position", Positions) 
    PerformanceScore = st.selectbox("Select the Performance Score", Performance_Score)
    EmpSatisfaction  = st.selectbox('Select the Satisfaction of the Employee', [0,1,2,3,4,5])
    SpecialProjectsCount = st.selectbox('Select the number of projects of the Employee', [0,1,2,3,4,5])
    Termd=0
    
   
        
    if Department=='Executive Office':
        Department_Executive_Office  = 1
    elif Department=='IT/IS':
        Department_IT_IS = 1
    elif Department=='Production':
        Department_Production = 1
    elif Department=='Sales':
        Department_Sales = 1    
    elif Department=='Software Engineering':
        Department_Software_Engineering = 1 
        
    
    if Managername == 'Amy Dunn':
       ManagerName_Amy_Dunn = 1  
    elif Managername == 'Board of Directors':                     
       ManagerName_Board_of_Directors = 1
    elif Managername == 'Brandon R. LeBlanc':   
       ManagerName_Brandon_R_LeBlanc = 1 
    elif Managername == 'Brandon R. LeBlanc':  
       ManagerName_Brannon_Miller = 1
    elif Managername == 'Brian Champaigne':                     
       ManagerName_Brian_Champaigne = 1
    elif Managername == 'David Stanley':                  
       ManagerName_David_Stanley = 1  
    elif Managername == 'Debra Houlihan':                    
       ManagerName_Debra_Houlihan = 1  
    elif Managername == 'Elijiah Gray':                   
       ManagerName_Elijiah_Gray = 1 
    elif Managername == 'Eric Dougall':                   
       ManagerName_Eric_Dougall = 1 
    elif Managername == 'Janet King':                     
       ManagerName_Janet_King = 1
    elif Managername == 'Jennifer Zamora':                       
       ManagerName_Jennifer_Zamora = 1
    elif Managername == 'John Smith':                   
       ManagerName_John_Smith = 1
    elif Managername == 'Kelley Spirea':                        
       ManagerName_Kelley_Spirea = 1
    elif Managername == 'Ketsia Liebig':                      
       ManagerName_Ketsia_Liebig = 1 
    elif Managername == 'Kissy Sullivan':                   
       ManagerName_Kissy_Sullivan = 1 
    elif Managername == 'Lynn Daneault':                   
       ManagerName_Lynn_Daneault = 1 
    elif Managername == 'Michael Albert':                    
       ManagerName_Michael_Albert = 1 
    elif Managername == 'Peter Monroe':                   
       ManagerName_Peter_Monroe = 1
    elif Managername == 'Simon Roup':                      
       ManagerName_Simon_Roup = 1
    elif Managername == 'Webster Butler':                      
       ManagerName_Webster_Butler = 1
       
      
    if RecruitmentSource == 'DiversityJob Fair': 
       RecruitmentSource_Diversity_Job_Fair = 1       
    elif RecruitmentSource == 'Employee Referral':
       RecruitmentSource_Employee_Referral = 1         
    elif RecruitmentSource == 'Google Search':
        RecruitmentSource_Google_Search = 1      
    elif RecruitmentSource == 'Indeed':
        RecruitmentSource_Indeed = 1  
    elif RecruitmentSource == 'Google Search':                
        RecruitmentSource_LinkedIn = 1  
    elif RecruitmentSource == 'On-line Web application':                
        RecruitmentSource_On_line_Web_application = 1  
    elif RecruitmentSource == 'Other':
        RecruitmentSource_Other = 1      
    elif RecruitmentSource == 'Website':              
        RecruitmentSource_Website = 1 
        
        
    if Position == 'Administrative Assistant':
       Position_Administrative_Assistant = 1  
    elif Position == 'Area Sales Manager':
       Position_Area_Sales_Manager = 1
    elif Position == 'BI Developer': 
       Position_BI_Developer  = 1
    elif Position == 'BI Director':
       Position_BI_Director  = 1 
    elif Position == 'CIO':          
       Position_CIO  = 1
    elif Position == 'Data_Analyst':
       Position_Data_Analyst  = 1 
    elif Position == 'Data Analyst II': 
       Position_Data_Analyst_II = 1
    elif Position == 'Data Architect':
       Position_Data_Architect  = 1  
    elif Position == 'Database Administrator': 
       Position_Database_Administrator  = 1
    elif Position == 'Director of Operations':
       Position_Director_of_Operations  = 1
    elif Position == 'Director of Sales': 
       Position_Director_of_Sales  = 1 
    elif Position == 'Enterprise Architect':
       Position_Enterprise_Architect  = 1
    elif Position == 'IT Director':          
       Position_IT_Director  = 1   
    elif Position == 'Manager DB':
       Position_IT_Manager_DB  = 1 
    elif Position == 'Manager Infra': 
       Position_IT_Manager_Infra  = 1
    elif Position == 'Manager Support ':
       Position_IT_Manager_Support  = 1
    elif Position == 'IT Support ':     
       Position_IT_Support  = 1
    elif Position == 'Network Engineer ':  
       Position_Network_Engineer = 1 
    elif Position == 'President and CEO':                    
       Position_President_and_CEO  = 1 
    elif Position == 'Data Architect':                         
       Position_Principal_Data_Architect  = 1
    elif Position == 'Production Manager':               
       Position_Production_Manager  = 1
    elif Position == 'Production Technician I':                     
       Position_Production_Technician_I  = 1 
    elif Position == 'Production Technician II':               
       Position_Production_Technician_II  = 1 
    elif Position == 'Sales Manager':              
       Position_Sales_Manager  = 1 
    elif Position == 'BI Developer':                        
       Position_Senior_BI_Developer  = 1 
    elif Position == 'Shared Services Manager':                   
       Position_Shared_Services_Manager  = 1
    elif Position == 'Software Engineer':                
       Position_Software_Engineer  = 1
    elif Position == 'Software Engineering Manager':                      
       Position_Software_Engineering_Manager = 1 
    elif Position == 'Sr Accountant ':           
       Position_Sr_Accountant = 1
    elif Position == 'Sr DBA':                          
       Position_Sr_DBA = 1
    elif Position == 'Sr Network Engineer':                                  
       Position_Sr_Network_Engineer = 1  
     
    if PerformanceScore == 'Fully Meets':
       PerformanceScore_Fully_Meets = 1 
    if PerformanceScore == 'Needs Improvement':                
     PerformanceScore_Needs_Improvement = 1
    if PerformanceScore == 'PIP':      
     PerformanceScore_PIP = 1
     
    if st.button("Get Your Prediction"): 
     
        x = [EmpSatisfaction, SpecialProjectsCount,Termd,MaritalDesc
            ]
        st.write(x)
       
        
       
        
    