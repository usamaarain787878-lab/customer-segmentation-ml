import pandas as pd 

# Sample customer data
data = {
	'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	'Age': [19, 21, 20, 23, 31, 22, 35, 25, 30, 40],
	'Annual_Income': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
	'Spending_Score': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

df = pd.DataFrame(data)
df.to_csv('cleaned_data.csv', index=False)
print("cleaned_data.csv kamyabi se ban gayi hai!")
