import matplotlib.pyplot as plt
from load_data import load_data 

data = load_data("data.csv")
new_data = load_data("cleaned_data.csv")

print(data.shape)
print(new_data.shape)

plt.scatter(data['experience'], data['salary'])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary for raw data")
plt.show()

plt.scatter(data["certifications"], data["salary"])
plt.xlabel("Certifications")
plt.ylabel("Salary")
plt.title("Certifications vs Salary for raw data")
plt.show()

plt.scatter(data["working_hours"], data["salary"])
plt.xlabel("Working Hours")
plt.ylabel("Salary")
plt.title("Working Hours vs Salary for raw data")
plt.show()

plt.scatter(data["age"], data["salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary for raw data")
plt.show()

plt.hist(data["salary"], bins=10)
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution for raw data")
plt.show()

plt.scatter(new_data['experience'], new_data['salary'])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary for cleaned data")
plt.show()

plt.scatter(new_data["certifications"], new_data["salary"])
plt.xlabel("Certifications")
plt.ylabel("Salary")
plt.title("Certifications vs Salary for cleaned data")
plt.show()

plt.scatter(new_data["working_hours"], new_data["salary"])
plt.xlabel("Working Hours")
plt.ylabel("Salary")
plt.title("Working Hours vs Salary for cleaned data")
plt.show()

plt.scatter(new_data["age"], new_data["salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary for cleaned data")
plt.show()

plt.hist(new_data["salary"], bins=10)
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution for cleaned data")
plt.show()