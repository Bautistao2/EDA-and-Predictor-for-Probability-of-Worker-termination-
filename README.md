# Worker Performance Exploration Analysis Data :100:
<div class=text-justify>
I performed HR analytics using the 'HR Dataset'. As a first step, we imported the necessary libraries, and then we imported the dataset in which we need to perform EDA. Before beginning with the analysis, I did some data manipulation as well

I performed various queries for analysing the data for insights and also to create visualization for appealing appearance and better understanding.Thus, I successfully did the HR Data Analytics and find the insights that could extremely help the company to take decisions as per the insights obtain, for the sake of their company.

I will project employees who are likely to be terminated based on a set of features
</div>

## Description

An EDAD project in wich have beed projected employees who are likely to be laid off based on a set of characteristics.

- **Employee Name :** Employee’s full name
- **EmpID :** Employee ID is unique to each employee
- **MarriedID :** Is the person married (1 or 0 for yes or no)
- **MaritalStatusID :** Marital status code that matches the text field MaritalDesc
- **EmpStatusID :** Employment status code that matches text field EmploymentStatus
- **DeptID :** Department ID code that matches the department the employee works in
- **PerfScoreID :** Performance Score code that matches the employee’s most recent performance score
- **FromDiversityJobFairID :** Was the employee sourced from the Diversity job fair? 1 or 0 for yes or no
- **PayRate :** The person’s hourly pay rate. All salaries are converted to hourly pay rate
- **Termd :** Has this employee been terminated - 1 or 0
- **PositionID :** An integer indicating the person’s position
- **Position :** The text name/title of the position the person has
- **State :** The state that the person lives in
- **Zip :** The zip code for the employee
- **DOB :** Date of Birth for the employee
- **Sex :** Sex - M or F
- **MaritalDesc :** The marital status of the person (divorced, single, widowed, separated, etc)
- **CitizenDesc :** Label for whether the person is a Citizen or Eligible NonCitizen
- **HispanicLatino :** Yes or No field for whether the employee is Hispanic/Latino
- **RaceDesc :** Description/text of the race the person identifies with
- **DateofHire :** Date the person was hired
- **DateofTermination :** Date the person was terminated, only populated if, in fact, Termd = 1
- **TermReason :** A text reason / description for why the person was terminated
- **EmploymentStatus :** A description/category of the person’s employment status. Anyone currently working full time = Active
- **Department :** Name of the department that the person works in
- **ManagerName :** The name of the person’s immediate manager
- **ManagerID :** A unique identifier for each manager
- **RecruitmentSource :** The name of the recruitment source where the employee was recruited from
- **PeRformanceScore :** Performance Score text/category (Fully Meets, Partially Meets, PIP, Exceeds)
- **EngagementSurvey :**  Results from the last engagement survey, managed by our external partner
- **EmpSatisfaction :** A basic satisfaction score between 1 and 5, as reported on a recent employee 
satisfaction survey
- **SpecialProjectsCount :** The number of special projects that the employee worked on during the last 6 months
- **LastPerformanceReviewDate :** The most recent date of the person’s last performance review.
- **DaysLateLast30 :** The number of times that the employee was late to work during the last 30 days


## Code and Resources Used

Python Version: 3.7 Packages: pandas, numpy, datetime, scipy, sklearn, matplotlib, seaborn.

## 1. Data cleaning and feature engineering:

- Fix the Day Of Birth dates of the original %m/%d/%y format, which Pandas might not convert correctly.
- Remove the extra white spaces and bring the Employee_Name and ManagerName columns to the same format.
- Remove extra white spaces in the Department column and transform the values of the TermReason and - - - HispanicLatino columns to lowercase.
- Divide the employees into active and left (terminated). 
- Add employee age and length of service (the number of years an employee has been working for the company).
- Missing values in the DateofTermination column mean that 207 persons are still employed

-

## 2. Exploratory Data Analysis:

**Solving the following questions**

### Which department suffers the most from absences?

![x](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/absences_by_department.png)

### Which gender has the most absences?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/absences_by_gender.png)

### Which department has the most days late in the last 30 days?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/DaysLate_by_department.png)

### What is the level of employee satisfaction by department?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/DepartmentandSatisfaction.png)

### What is the employees performance score?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/PerformanceScore.png)

### Is there some relationship between the recruitment resource and worker performance?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/RecruitmentSource_PerformanceScore.png)

### Is there any difference in payment by gender?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/SalarybyGender.png)

### What has been the lowest and highest dismissal rate in the company

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/newplot.png)




## 3. Build and training prediction model

I made the prediction to know which employee is more likely to resign, based on certain characteristics of the employee.
To do this I follow the usual steps:

### Data preprocessing 

- Select most important columns to make the model prediction
- Apply get dummies to the categorical columns selected
- Termd is the target vector to the prediction model
- Made a balanced dataset by using SMOTE technique.
- Columns scaling was done
- Data scaling
### Build a model
- Exploring model performance of different ML regression algorithms
- The most promising algorithm (NeuralNetwork) was adjusted.
- The characteristics that have the greatest impact on the prediction were calculated.

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/important.png)


## 4. Insights

<div class=text-justify>

- 60% of the employees are highly satisfied with the company and 34% are satisfied, this means that only 6% of the employees are not satisfied, so it can be said that the company has good motivation strategies, and employee retention as well as corporate culture.
- There are only two completely dissatisfied employees and they are in the production and sales department, although production is the department with the largest number of workers, it would be good to review the sales department to know what factors can influence
- The production department reports 8 times more employees who terminate their contract by their own decision than by cause, this means that there are disagreements among employees in the department that cause them to leave the company.
- Indeed and LinkedIn are the top platforms from where the company hired most of the candidates
- Although LinkedIn and Indeed are the recruiting resources that have brought the most active workers to the company, they are not the most retired workers from the company, the workers recruited through Google search have been fired more times for cause
- At least 56% of employees terminated for cause had a high performance score, this means that any employee can be terminated for cause even when their performance is good or excellent.
- Women have an average of 31% more absences than men. It is striking that although the sales department has only 14% of the number of employees than the production department, it has a similar absence ratio.
- At least 90% of the company's employees have a high performance score, it can be said that in general the staff is productive
- The employees recruited by the web application are the only candidates who do not have a low performance score
- The special projects that an employee is developing do not indicate that they cannot have a low performance score.
- In workers with a low degree of job satisfaction, a low performance score is found.
- The departments where there are no workers with low performance scores are the executives and the office administration.
- The organization pays females $2843 less than men. However, this metric is unadjusted for various factors that are known to affect salary, including job level, tenure, previous work experience, and more
- No matter where the hire is coming from for the HR department, all employees are able to get a performance score of "Fully Meets". Top sources being Diversity Job Fair, Linked and Indeed.

- ~75% of the candidates who get a performance score of "Fully Meets" are coming from Indeed followed by Linkedin.

- Hires from Indeed and Linkedin will help the overall organization be productive.
- Accounting hires are strong when they come from employee referrals
- All job sources for human resources are strong.
- Sales hires are strong when they come from Indeed and Website.
- Employee Referral job source is considered to the strongest.
- Salary directly influences employee satisfaction, because dissatisfied employees are those with the lowest salary.
</div>



