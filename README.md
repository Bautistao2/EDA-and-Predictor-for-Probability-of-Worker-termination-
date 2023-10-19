# Probability of worker termination : Exploration Analysis Data, and predictor model :100:
<div class=text-justify>
HR analytics were performed using the data of a company. As a first step, the necessary libraries and dataset were imported to perform the EDA. Before beginning with the analysis, some data formatting was applying, as well.

Various queries were performed to analyse the data for insights, and also to create an appealing and easily assimilated visualization. The insights thus acquired could be  very  helpful for the company to make decisions.

**The analysis projects those employees who are likely to be terminated based on a set of factors.**
</div>

## Description

An EDA project that predicts employees who are likely to be laid off based on a set of characteristics.

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

Python Version: 3.8 Packages: pandas, numpy, sklearn, matplotlib, seaborn, plotly.

## 1. Data cleaning and feature engineering:

- Fix the Day Of Birth dates from the original %m/%d/%y format, which Pandas might not convert correctly.
- Remove the extra white spaces and convert the Employee_Name and ManagerName columns to the same format.
- Remove extra white spaces in the Department column and transform the values of the TermReason and HispanicLatino columns to lowercase.
- Divide the employees into active and terminated. 
- Add employee age and length of service (the number of years an employee has been working for the company).
- Missing values in the "Date of Termination" column mean that 207 persons are still employed

## 2. Exploratory Data Analysis:

**Answering the following questions**
### Which department suffers the most from absences?

![x](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/absences_by_department.png)

### Which gender has the most absences?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/absences_by_gender.png)

### Which department has the most days late in the last 30 days?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/DaysLate_by_department.png)

### What is the level of employee satisfaction by department?

![](https://github.com/Bautistao2/EDA-and-Predictor-for-Probability-of-Worker-termination-/blob/main/images/Deparments%20by%20Employment%20Satisfaction.png)

### What is the performance score for each employee?

![](https://github.com/Bautistao2/EDA-and-Predictor-for-Probability-of-Worker-termination-/blob/main/images/Performance%20Score.png)

### Is there a relationship between the recruitment resource and worker performance?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/RecruitmentSource_PerformanceScore.png)

### Is there a difference in remuneration by gender?

![](https://github.com/Bautistao2/EDA-and-Predictor-for-Probability-of-Worker-termination-/blob/main/images/SalaryvsFrequency.png)

### What has been the lowest and highest dismissal rate per year?

![](https://github.com/Bautistao2/Worker-performance-EDA/blob/main/images/newplot.png)

## 3. Build and training prediction model

Building and training the model involved the following steps:
### Data pre-processing 

- Select the most important columns for the prediction model.
- Apply get dummies to the categoric columns selected.
- Termd is the target vector for the prediction model.
- Make a balanced dataset by using the SMOTE technique.
- Perform data scaling.
  
### Build the model
- Explore the model performance of different Machine Learning classification algorithms.
- The most promising algorithm (Decision Tree Classifier) was adjusted.
- The characteristics that have the greatest impact on the prediction were calculated.

![](https://github.com/Bautistao2/EDA-and-Predictor-for-Probability-of-Worker-termination-/blob/main/images/FactorvsScore.png)


## 4. Insights

-	60% of the employees are highly satisfied with the company, and 34% are satisfied. This means that only 6% of the employees are not satisfied. It can thus be inferred that the company has good motivation strategies and employee retention, as well as corporate culture.
-	There are only two completely dissatisfied employees, and they are in the production and sales department. Although production is the department with the largest number of workers, it would be good to review the sales department to know what factors may be at play.
-   The production department reports 8 times more employees who end their contract voluntarily rather than being terminated for cause. This finding implies possible sources of dissatisfaction within the department that cause them to leave the company.
-	Indeed and LinkedIn are the top platforms from which the company sourced its candidates.
-   Although LinkedIn and Indeed have provided the largest portion of active workers, the recruiting source with the highest number of workers subsequently terminated for cause was Google search.
-	At least 56% of employees terminated for cause had high performance scores, This finding implies that an employee can be terminated for cause even when their performance is good or excellent. This result is concerning, since it could indicate two possible problems. Either 1) The company’s performance measurement system is flawed, or 2) The company’s termination decisions are inappropriate.
-	Women exhibit an average of 31% more absences than men.
-	It is noteworthy that, although the sales department has only 14% of the number of employees that the production department does, both departments present nearly the same number of total absences.
-	At least 90% of the employees have high performance scores; implying that, in general, the staff is productive; although the aforementioned finding may cast doubt upon the reliability of the company’s performance metrics.
-	The employees recruited via the web application are the only candidates who do not exhibit any low performance scores, whatsoever.
-	The fact that an employee is working on special projects does not preclude their having a low performance score.
-	Workers with low satisfaction scores also tend to have low performance scores. This could imply a positive causational relationship between the variables in either direction (i.e., low performance leads to low satisfaction or vice versa); or that another variable impacts both satisfaction and performance together (e.g., the quality of management in a given area impacts both satisfaction and performance).
-	Only the executives and office administration departments had no workers with low performance scores.
-	The organization pays women $2,843 less than men. However, this metric is unadjusted for various factors that are known to affect salary; including job level, tenure, previous work experience, among others.
-	Regardless of the source of the hire, all employees are able to achieve a performance score of "Fully Meets"; although top-scoring employees are more frequently sourced from the Diversity Job Fair, Linked and Indeed.
-	~75% of the candidates who get a performance score of "Fully Meets" were sourced from Indeed, followed by Linkedin, implying that hires from Indeed and Linkedin will help the overall organization be productive.
-	Salary correlates highly with employee satisfaction; however, other variables also play a role in determining satisfaction scores.


## 5. Recomendations

After analyzing the most common causes for voluntary employee departure, and considering that an employee is 5 times more likely to voluntarily depart as opposed to being dismissed for cause, it is recommended that the company analyze and address the following: 

1. Review the pay scales across various positions to reduce the loss of employees to other companies that offer higher remuneration. Design non-monetary motivation strategies to increase employee satisfaction as a complement for employee salary.
2. Implement enjoyable team-building activities to build stronger inter-employee relationships and teamwork.
3. Create a professional growth program to train high-performing employees for future career opportunities, and to provide employees with opportunities to demonstrate their skills via the assignment of new responsibilities. 
4. Considering that women represent both the most frequent voluntarily withdrawals, as well as those most often terminated for cause; strategies should be developed to support female employees in the performance of their jobs; and ensure gender equality in terms of salaries, responsibilities, and positions.
5. Redesign employee satisfaction metrics to more accurately reflect employee feelings; given that currently, employees with high satisfaction scores are voluntarily leaving the company in equal measure.
6. Implement programs to drive employee engagement in creating, innovating, researching, and carrying out special projects; considering that the great majority of employees terminated for cause are not participating in these undertakings.
7. Generate brand-marketing strategies to increase the number of candidates who fill out an online web application, and incentivize current employees to refer more candidates; as both these recruitment channels are those from which the fewest employees are eventually dismissed. 
8. Improve employee selection criteria used to select employees for production technician, production manager, and sales roles.
9. Analyze workloads, in order to permit married employees to achieve a healthy work/life balance.


