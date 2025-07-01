# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import itertools

# Load your dataset (assuming it's a CSV file)
# Change the path to the location where your dataset is stored
df = pd.read_csv('aptimodel_dataset')

# *** Create 'aptitude_category' column based on total aptitude score ***
# Calculate total aptitude score
df['total_aptitude'] = df[['numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude', 'abstract_reasoning', 'verbal_reasoning']].sum(axis=1)

# Define thresholds for aptitude categories (adjust as needed)
low_threshold = df['total_aptitude'].quantile(0.33)
high_threshold = df['total_aptitude'].quantile(0.67)

# Assign aptitude categories based on thresholds
df['aptitude_category'] = pd.cut(df['total_aptitude'], bins=[0, low_threshold, high_threshold, float('inf')], labels=['Low', 'Average', 'High'])


# Display the first few rows of the dataset to confirm it loaded correctly
print(df.head())

# Selecting the relevant features for aptitude assessment
X = df[['numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude',
        'abstract_reasoning', 'verbal_reasoning']]  # Input features

# Define target variable (aptitude category)
# Ensure the target column in your dataset is properly labeled
y = LabelEncoder().fit_transform(df['aptitude_category'])  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize input features (before prediction)
# Pair plot of the features
sns.pairplot(df[['numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude',
                 'abstract_reasoning', 'verbal_reasoning', 'aptitude_category']],
             hue='aptitude_category')
plt.suptitle("Pairplot of Aptitude Features Before Prediction", y=1.02)
plt.show()

# Histograms of individual features
df[['numerical_aptitude', 'spatial_aptitude', 'perceptual_aptitude',
    'abstract_reasoning', 'verbal_reasoning']].hist(bins=15, figsize=(10, 6))
plt.suptitle("Distribution of Aptitude Features Before Prediction", y=1.02)
plt.show()

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Training and Testing Accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm, classes=['Low', 'Average', 'High'], title='Confusion Matrix After Prediction')
plt.show()

# Bar plot of predicted aptitude categories
df_test = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
sns.countplot(x='Predicted', data=df_test)
plt.title('Predicted Aptitude Categories')
plt.show()
import pickle

# Save the trained model to a file (model.pkl)
with open('model1.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as 'model1.pkl'")
import matplotlib.pyplot as plt

# Compute predicted accuracy as percentage
predicted_accuracy = accuracy_score(y_test, y_pred) * 100

# Since the actual (ideal) accuracy is 100%
actual_accuracy = 100

# Data for the bar chart
accuracies = [actual_accuracy, predicted_accuracy]
labels = ['Actual (Ideal)', 'Predicted (Model)']

# Create the bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, accuracies, color=['green', 'blue'])

# Add text labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom', fontsize=12)

plt.ylim([0, 110])
plt.title("Actual vs. Predicted Accuracy")
plt.ylabel("Accuracy (%)")
plt.show()
# import numpy as np
# import pandas as pd
# import joblib  # To save and load models
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load trained model (if saved earlier)
# model = joblib.load("model1.pkl")  # Load the trained model

# # Load encoders & scalers (if used in training)
# scaler = joblib.load("scaler.pkl")  # Load the scaler
# label_encoder = joblib.load("label_encoder.pkl")  # Load label encoder

# # Define the input features (Make sure it matches the trained model)
# feature_names = ["Numerical_Reasoning", "Verbal_Reasoning", "Logical_Thinking", "Problem_Solving", "Abstract_Thinking"]

# # Take user input
# def get_user_input():
#     user_data = {}
#     print("\nEnter your aptitude test scores (between 0 and 100):")
#     for feature in feature_names:
#         user_data[feature] = float(input(f"{feature}: "))
    
#     # Convert to DataFrame
#     user_df = pd.DataFrame([user_data])
    
#     return user_df

# # Process the user input and predict career
# def predict_career():
#     user_input = get_user_input()
    
#     # Scale the input data
#     user_input_scaled = scaler.transform(user_input)
    
#     # Make a prediction
#     career_prediction = model.predict(user_input_scaled)
    
#     # Convert predicted label back to career name
#     predicted_career = label_encoder.inverse_transform(career_prediction)[0]
    
#     print(f"\n🎯 Predicted Career Path: {predicted_career}")

# # Run the prediction
# predict_career()
# Import necessary libraries
# Define the input prompt for the user
# def get_user_input_for_career():
#     numerical_aptitude = float(input("Enter your score for Numerical Aptitude: "))
#     spatial_aptitude = float(input("Enter your score for Spatial Aptitude: "))
#     perceptual_aptitude = float(input("Enter your score for Perceptual Aptitude: "))
#     abstract_reasoning = float(input("Enter your score for Abstract Reasoning: "))
#     verbal_reasoning = float(input("Enter your score for Verbal Reasoning: "))
    
#     # Combine all input features into an array
#     user_input = np.array([[numerical_aptitude, spatial_aptitude, perceptual_aptitude, abstract_reasoning, verbal_reasoning]])
#     return user_input

# # Assuming the model is already trained (from the previous code)
# # Get input from the user
# user_input = get_user_input_for_career()

# # Make a prediction using the trained model
# career_prediction = model.predict(user_input)

# # Define career paths for each category (Adjust this list based on your actual categories and mapping)
# careers = {
#     0: 'Data Scientist, Software Engineer, Statistician',
#     1: 'Marketing Specialist, Content Writer, Public Relations Manager',
#     2: 'Architect, Designer, Urban Planner',
#     3: 'Mechanical Engineer, Electrician, Pilot',
#     4: 'Research Scientist, Laboratory Technician, Pharmacist',
#     5: 'Artist, Musician, Graphic Designer',
#     6: 'Teacher, Counselor, Social Worker',
#     7: 'Entrepreneur, Sales Manager, Business Consultant',
#     8: 'Accountant, Financial Analyst, Auditor'
# }


# # Output the predicted career path
# predicted_career = careers.get(career_prediction[0], 'Career not found')
# print(f"Based on your aptitude scores, the recommended career path is: {predicted_career}")

import numpy as np

def get_user_input_for_career():
    numerical_aptitude = float(input("Enter your score for Numerical Aptitude: "))
    spatial_aptitude = float(input("Enter your score for Spatial Aptitude: "))
    perceptual_aptitude = float(input("Enter your score for Perceptual Aptitude: "))
    abstract_reasoning = float(input("Enter your score for Abstract Reasoning: "))
    verbal_reasoning = float(input("Enter your score for Verbal Reasoning: "))
    
    # Combine all input features into an array
    user_input = np.array([[numerical_aptitude, spatial_aptitude, perceptual_aptitude, abstract_reasoning, verbal_reasoning]])
    return user_input

# Assuming the model is already trained (from the previous code)
# Get input from the user
user_input = get_user_input_for_career()

# Make a prediction using the trained model
career_prediction = model.predict(user_input)

# Define career paths based on highest and average scores
careers = {
    "high": ['Engineering', 'Doctor', 'Marketing', 'Scientist', 'Entrepreneur'],
    "average": ['Marketing Specialist', 'Content Writer', 'Public Relations Manager',
                'Architect', 'Designer', 'Urban Planner',
                'Mechanical Engineer', 'Electrician', 'Pilot',
                'Research Scientist', 'Laboratory Technician', 'Pharmacist'],
    "low": ['Artist', 'Musician', 'Graphic Designer',
            'Teacher', 'Counselor', 'Social Worker',
            'Entrepreneur', 'Sales Manager', 'Business Consultant',
            'Accountant', 'Financial Analyst', 'Auditor']
}

# Determine highest and average score
max_score = np.max(user_input)
avg_score = np.mean(user_input)

if max_score >= 80:
    predicted_career = np.random.choice(careers["high"])
elif avg_score >= 50:
    predicted_career = np.random.choice(careers["average"])
else:
    predicted_career = np.random.choice(careers["low"])

print(f"Based on your aptitude scores, the recommended career path is: {predicted_career}")
