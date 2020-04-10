# --------------
# Import packages
import pandas as pd
import numpy as np

# code starts here

# loading and reading data
bank = pd.read_csv(path)

# checking the datatype of cloumns
categorical_var = bank.select_dtypes(include='object')
numerical_var = bank.select_dtypes(include='number')

print(categorical_var)
print(numerical_var)

# load the dataset and drop the Loan_ID
banks = bank.drop('Loan_ID', axis=1)


# Checking all missing values
print(banks.isnull().sum())

# apply mode to dataframe
bank_mode = banks.mode().to_dict(orient='records')

# Filling in the missing values
banks.fillna(bank_mode[0], inplace=True)

# Rechecking the missing values
print(banks.isnull().sum())

# creating the pivot table
avg_loan_amount = pd.pivot_table(banks, index=['Gender', 'Married', 'Self_Employed'], values='LoanAmount', aggfunc=np.mean)


print(avg_loan_amount)

# calculating percentage of loans approved for self employed and non self employed

loan_approved_se = len(banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status'] == 'Y')])

loan_approved_nse = len(banks[(banks['Self_Employed']=='No') & (banks['Loan_Status'] == 'Y')])

percentage_se = loan_approved_se/614*100
percentage_nse = loan_approved_nse/614*100

# Transform the loan tenure from months to years for big audit
months_to_years = lambda x: int(x//12)
loan_term = banks['Loan_Amount_Term'].apply(months_to_years)
big_loan_term = len(loan_term[loan_term >=25])

# Income/ Credit History vs Loan Amount
loan_groupby = banks.groupby('Loan_Status')['ApplicantIncome', 'Credit_History']
mean_values = loan_groupby.mean()



# --------------
banks = bank.drop('Loan_ID',axis=1)
#print(banks)

print(banks.isnull().sum())

bank_mode = banks.mode


banks = banks.fillna(bank_mode)
print(banks.isnull().sum())


# --------------
# code starts here

# check the avg_loan_amount
avg_loan_amount = banks.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)


print (avg_loan_amount)
# code ends here


# --------------





loan_approved_se=banks[(banks.Self_Employed=='Yes')& (banks.Loan_Status == 'Y')].Loan_Status.value_counts()[0]
loan_approved_nse=banks[(banks.Self_Employed=='No')& (banks.Loan_Status == 'Y')].Loan_Status.value_counts()[0]
percentage_se=(loan_approved_se/614)*100
percentage_nse=(loan_approved_nse/614)*100
print(percentage_se)
print(percentage_nse)


# --------------
# code starts here


loan_term = banks['Loan_Amount_Term'].apply(lambda x : x/12)
big_loan_term = len(banks[loan_term>=25])
print(big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby(['Loan_Status'])
loan_groupby = loan_groupby['ApplicantIncome','Credit_History']
mean_values = loan_groupby.mean()

print(mean_values)




# code ends here


