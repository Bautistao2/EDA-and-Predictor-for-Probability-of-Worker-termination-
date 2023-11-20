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
    
   
    MaritalDesc_Single=0
    MaritalDesc_Married=0  
    MaritalDesc_Separated=0 
    MaritalDesc_Widowed=0
        
       
    MaritalDesc = st.selectbox("Select the Marital Status", MaritalD) 
   
    if MaritalDesc=='Single':
           MaritalDesc_Single = 1
    elif MaritalDesc=='Married':
            MaritalDesc_Married = 1
    elif MaritalDesc=='Separated':
            MaritalDesc_Separated = 1
    elif MaritalDesc=='Widowed':
             MaritalDesc_Widowed = 1 
             
    Sex_M=0      
            
    Sex = st.selectbox("Select the sex", Gender) 
    if Sex == 'F':
        Sex_M = 1
    else:
        Sex_M = 0
      
      
    Department_Executive_Office = 0
    Department_IT_IS = 0
    Department_Production=0
    Department_Sales=0
    Department_Software_Engineering=0
    
    Department = st.selectbox("Select the Department", Department_E)
    
    if Department=='Executive Office':
        Department_Executive_Office = 1
    elif Department=='IT/IS':
        Department_IT_IS = 1
    elif Department=='Production':
        Department_Production = 1
    elif Department=='Sales':
        Department_Sales = 1    
    elif Department=='Software Engineering':
        Department_Software_Engineering = 1 
    
    ManagerName_Amy_Dunn = 0
    ManagerName_Board_of_Directors = 0
    ManagerName_Brandon_R_LeBlanc = 0
    ManagerName_Brannon_Miller = 0
    ManagerName_Brian_Champaigne = 0
    ManagerName_David_Stanley = 0
    ManagerName_Debra_Houlihan = 0
    ManagerName_Elijiah_Gray = 0
    ManagerName_Eric_Dougall = 0
    ManagerName_Janet_King = 0
    ManagerName_Jennifer_Zamora = 0
    ManagerName_John_Smith = 0
    ManagerName_Kelley_Spirea = 0
    ManagerName_Ketsia_Liebig = 0
    ManagerName_Kissy_Sullivan = 0
    ManagerName_Lynn_Daneault = 0
    ManagerName_Michael_Albert = 0
    ManagerName_Peter_Monroe = 0
    ManagerName_Simon_Roup = 0
    ManagerName_Webster_Butler = 0    
    
    Managername = st.selectbox("Select the Manager Name", Manager_name)
    
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
    
    def crear(lista_variables):
       diccionario = {variable: 0 for variable in lista_variables}
       
    #Recruitmen Channel
    
    
    RecruitmentSource_Diversity_Job_Fair = 0 
    RecruitmentSource_Employee_Referral = 0
    RecruitmentSource_Google_Search = 0
    RecruitmentSource_Indeed = 0
    RecruitmentSource_LinkedIn = 0
    RecruitmentSource_On_line_Web_application = 0
    RecruitmentSource_Other = 0
    RecruitmentSource_Website = 0
    
    
    RecruitmentSource = st.selectbox("Select the Recruitment Source", Recruitment_Source)

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
    
    #Position of Employee
        
    Position_Administrative_Assistant=0
    Position_Area_Sales_Manager = 0
    Position_BI_Developer = 0
    Position_BI_Director = 0
    Position_CIO = 0 
    Position_Data_Analyst = 0
    Position_Data_Architect = 0
    Position_Data_Analyst_II = 0
    Position_Database_Administrator = 0
    Position_Director_of_Operations = 0
    Position_Director_of_Sales = 0 
    Position_Enterprise_Architect = 0
    Position_IT_Director = 0
    Position_IT_Manager_DB = 0
    Position_IT_Manager_Infra = 0
    Position_IT_Manager_Support = 0
    Position_IT_Support = 0
    Position_Network_Engineer = 0
    Position_President_and_CEO = 0
    Position_Principal_Data_Architect = 0
    Position_Production_Manager = 0
    Position_Production_Technician_I = 0
    Position_Production_Technician_II = 0
    Position_Sales_Manager = 0
    Position_Senior_BI_Developer = 0
    Position_Shared_Services_Manager = 0
    Position_Software_Engineer = 0
    Position_Software_Engineering_Manager = 0
    Position_Sr_Accountant = 0
    Position_Sr_DBA = 0
    Position_Sr_Network_Engineer = 0      
        
    Position = st.selectbox("Select the Position", Positions) 
    
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
    elif Position == 'Manager Support':
       Position_IT_Manager_Support  = 1
    elif Position == 'IT Support':     
       Position_IT_Support  = 1
    elif Position == 'Network Engineer':  
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
       
   #Performance Score        
    
    PerformanceScore_Fully_Meets = 0
    PerformanceScore_Needs_Improvement = 0
    PerformanceScore_PIP = 0
        
        
    PerformanceScore = st.selectbox("Select the Performance Score", Performance_Score)
    
    if PerformanceScore == 'Fully Meets':
       PerformanceScore_Fully_Meets = 1 
    if PerformanceScore == 'Needs Improvement':                
     PerformanceScore_Needs_Improvement = 1
    if PerformanceScore == 'PIP':      
     PerformanceScore_PIP = 1 
     
     
      
    #Employee Satisfaction 
       
    EmpSatisfaction  = st.selectbox('Select the Satisfaction of the Employee', [0,1,2,3,4,5])
    
    #Special Poject Counts
    
    SpecialProjectsCount = st.selectbox('Select the number of projects of the Employee', [0,1,2,3,4,5])
    
    #Termd Target , not needed obtain its value
    
    Termd=0
   

    x= [EmpSatisfaction,
        SpecialProjectsCount,
        Termd,
        MaritalDesc_Married,
        MaritalDesc_Separated,
        MaritalDesc_Single,
        MaritalDesc_Widowed,
        Sex_M,
        Department_Production,
        Department_Sales,
        Department_Software_Engineering,
        PerformanceScore_Fully_Meets,
        PerformanceScore_Needs_Improvement,
        PerformanceScore_PIP,
        Position_Administrative_Assistant,
        Position_Area_Sales_Manager,
        Position_BI_Developer,
        Position_BI_Director,
        Position_CIO,
        Position_Data_Analyst,
        Position_Data_Architect,
        Position_Data_Analyst_II,
        Position_Database_Administrator,
        Position_Director_of_Operations,
        Position_Director_of_Sales,
        Position_Enterprise_Architect,
        Position_IT_Director,
        Position_IT_Manager_DB,
        Position_IT_Manager_Infra,
        Position_IT_Manager_Support,
        Position_IT_Support,
        Position_Network_Engineer,
        Position_President_and_CEO,
        Position_Principal_Data_Architect,
        Position_Production_Manager,
        Position_Production_Technician_I,
        Position_Production_Technician_II,
        Position_Sales_Manager,
        Position_Senior_BI_Developer,
        Position_Shared_Services_Manager,
        Position_Software_Engineer,
        Position_Software_Engineering_Manager,
        Position_Sr_Accountant,
        Position_Sr_DBA,
        Position_Sr_Network_Engineer,
        ManagerName_Amy_Dunn,
        ManagerName_Board_of_Directors,
        ManagerName_Brandon_R_LeBlanc,
        ManagerName_Brannon_Miller,
        ManagerName_Brian_Champaigne,
        ManagerName_David_Stanley,
        ManagerName_Debra_Houlihan,
        ManagerName_Elijiah_Gray,
        ManagerName_Eric_Dougall,
        ManagerName_Janet_King,
        ManagerName_Jennifer_Zamora,
        ManagerName_John_Smith,
        ManagerName_Kelley_Spirea,
        ManagerName_Ketsia_Liebig,
        ManagerName_Kissy_Sullivan,
        ManagerName_Lynn_Daneault,
        ManagerName_Michael_Albert,
        ManagerName_Peter_Monroe,
        ManagerName_Simon_Roup,
        ManagerName_Webster_Butler,
        RecruitmentSource_Diversity_Job_Fair,
        RecruitmentSource_Employee_Referral,
        RecruitmentSource_Google_Search,
        RecruitmentSource_Indeed,
        RecruitmentSource_LinkedIn,
        RecruitmentSource_On_line_Web_application,
        RecruitmentSource_Other,
        RecruitmentSource_Website]    
     
        
             
    def modelpred():
        modelocargado=pickle.load(open('modelop1.pkl','rb'))
        pred=modelocargado.predict([variable])
        return variable        
        
    if st.button("Predecir"):
        resultado = modelpred
        st.subheader(x)


  

       
        
       
        
    