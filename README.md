# Worker Performance Exploration Analysis Data

In this project, we will project employees who are likely to be terminated based on a set of features
## Description

An EDAD project in wich  employees who are likely to be terminated based on a set of features we will project employees who are likely to be laid off based on a set of characteristics.

**Employee Name :** Employee’s full name

**EmpID :** Employee ID is unique to each employee

**MarriedID :** Is the person married (1 or 0 for yes or no)

**MaritalStatusID :** Marital status code that matches the text field MaritalDesc

**EmpStatusID :** Employment status code that matches text field EmploymentStatus

**DeptID :** Department ID code that matches the department the employee works in

**PerfScoreID :** Performance Score code that matches the employee’s most recent performance score

**FromDiversityJobFairID :** Was the employee sourced from the Diversity job fair? 1 or 0 for yes or no

**PayRate :** The person’s hourly pay rate. All salaries are converted to hourly pay rate

**Termd :** Has this employee been terminated - 1 or 0

**PositionID :** An integer indicating the person’s position

**Position :** The text name/title of the position the person has

**State :** The state that the person lives in

**Zip :** The zip code for the employee

**DOB :** Date of Birth for the employee

**Sex :** Sex - M or F

**MaritalDesc :** The marital status of the person (divorced, single, widowed, separated, etc)

**CitizenDesc :** Label for whether the person is a Citizen or Eligible NonCitizen

**HispanicLatino :** Yes or No field for whether the employee is Hispanic/Latino

**RaceDesc :** Description/text of the race the person identifies with

**DateofHire :** Date the person was hired

**DateofTermination :** Date the person was terminated, only populated if, in fact, Termd = 1

**TermReason :** A text reason / description for why the person was terminated

**EmploymentStatus :** A description/category of the person’s employment status. Anyone currently working full time = Active

**Department :** Name of the department that the person works in

**ManagerName :** The name of the person’s immediate manager

**ManagerID :** A unique identifier for each manager

**RecruitmentSource :** The name of the recruitment source where the employee was recruited from

**PerformanceScore :** Performance Score text/category (Fully Meets, Partially Meets, PIP, Exceeds)

**EngagementSurvey :**  Results from the last engagement survey, managed by our external partner

**EmpSatisfaction :** A basic satisfaction score between 1 and 5, as reported on a recent employee 
satisfaction survey

**SpecialProjectsCount :** The number of special projects that the employee worked on during the last 6 months

**LastPerformanceReviewDate :** The most recent date of the person’s last performance review.

**DaysLateLast30 :** The number of times that the employee was late to work during the last 30 days


## Code and Resources Used

Python Version: 3.7 Packages: pandas, numpy, datetime, scipy, sklearn, matplotlib, seaborn.

## 1. Data cleaning and feature engineering:

Fix the Day Of Birth dates of the original %m/%d/%y format, which Pandas might not convert correctly.
Remove the extra white spaces and bring the Employee_Name and ManagerName columns to the same format.
Remove extra white spaces in the Department column and transform the values of the TermReason and HispanicLatino columns to lowercase.
Divide the employees into active and left (terminated). 
Add employee age and length of service (the number of years an employee has been working for the company).
Missing values in the DateofTermination column mean that 207 persons are still employed
Select most important columns to make the model prediction
Apply get dummies to the categorical columns selected
Termd is the target vector to the prediction model
Made a balanced dataset by using SMOTE technique.
Column scaling was done

## 2. Exploratory Data Analysis:

solving the following questions

Which department suffers the most from absences?

(/images/abscences_by_department.png)


What relationship exists between the employee's department and his or her level of satisfaction?


