#!/usr/bin/env python
# coding: utf-8

# # Final Project- Analyze the top revenue 500 companies around the world 
# ### Li Xie (MSE) 15/5/2025

# ### References: 
# Data: https://github.com/cmusam/fortune500/blob/master/csv/fortune500-2019.csv <br>
# Methods: object oriented, image analysis, machine learning, and graphical design.

# In[12]:


# method 1: Method 1: Object oriented approach ,Create a dictionary to allow user input to check the top 10 Fortune companies along with their revenue.

top10 = {1: {'company': 'Walmart', 'revenue ($ millions)': 514405.0},
         2: {'company': 'Exxon Mobil', 'revenue ($ millions)': 290212.0},
         3: {'company': 'Apple', 'revenue ($ millions)': 265595.0},
         4: {'company': 'Berkshire Hathaway', 'revenue ($ millions)': 247837.0},
         5: {'company': 'Amazon', 'revenue ($ millions)': 232887.0},
         6: {'company': 'UnitedHealth Group', 'revenue ($ millions)': 226247.0},
         7: {'company': 'McKesson', 'revenue ($ millions)': 208357.0},
         8: {'company': 'CVS Health', 'revenue ($ millions)': 194579.0},
         9: {'company': 'AT&T', 'revenue ($ millions)': 170756.0},
         10: {'company': 'AmerisourceBergen', 'revenue ($ millions)': 167939.6}}

user_input = input("Please enter a number (1-10) to check the top 10 Fortune companies: ")

# Convert the input to an integer
user_input = int(user_input)

# Validate if the input is in the top10 dictionary
if user_input in top10:
    company_info = top10[user_input]
    print(f"Company: {company_info['company']}")
    print(f"Revenue ($ millions): {company_info['revenue ($ millions)']}")
else:
    print(f"{user_input} is not valid. Please enter a number between 1 and 10")


# In[13]:


# methods 2:  machine learning
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fortune500-2019.csv")
display(df)   # read the csv.

# clean NaN and delete the profit column
df_clean = df.dropna().copy()  # clean the NaN
df_clean = df_clean.drop("profit ($ millions)" , axis=1)  # delete the profit column

# add new volume of revenue(€ millions)
exchange_rate = 0.91  
df_clean['revenue(€ millions)'] = df_clean['revenue ($ millions)'] * exchange_rate  # add a new column of euro as unit 
df_clean.reset_index(drop=True, inplace=True)
df_clean.index = range(1, len(df_clean) + 1) # change the index number from 1 instead of 0
display(df_clean.head(10))

# creatd top 10 data as CSV file 
top10= df_clean.head(10)
top10.to_csv("top10_data.csv", index=False)

# graphic top 10 company 
import matplotlib.pyplot as plt
df = pd.read_csv("top10_data.csv")
graphic_data = df.groupby("company")["revenue(€ millions)"].sum()  # all rows with the same company name are grouped together to sum the revenue
ax= graphic_data.plot(kind='bar', figsize=(10, 5))
ax.set_ylabel("Revenue (€ millions)")
plt.xticks(rotation=45, ha="right")
ax.set_title("Top 10 Company Revenue (€ millions)")
plt.show()


# calssic high and low revenue to a new column.
class CompanyClassifier:
    
    def __init__(self, df, revenue_threshold):
        self.df = df
        self.revenue_threshold = revenue_threshold

    def classify_revenue(self):
        # Create a new column 'revenue_category' based on the threshold
        def classify(x, threshold):
            if x >= threshold:
                return 'High'
            elif x < threshold:
                return 'Low'

        # add a new column of revenue_category base on revenue(€ millions)
        self.df['revenue_category'] = self.df['revenue(€ millions)'].apply(classify, threshold=self.revenue_threshold)

    def get_classified_data(self):
        # return the DataFrame with classified revenue categories
        return self.df

revenue_threshold = 10000  
classifier = CompanyClassifier(df_clean, revenue_threshold)

# Classify companies based on revenue
classifier.classify_revenue()
display(classifier.get_classified_data())


# In[14]:


# image analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open images using OpenCV
image1 = cv2.imread("walmart.png", 1)
image2 = cv2.imread("iphone.jpg", 1)
image3 = cv2.imread("ExxonMobil.png", 1)
image4 = cv2.imread("berkshirehathaway.jpg", 1)
image5 = cv2.imread("amazon.jpg", 1)

# Resize all images to the same size
resize_size = (500, 500)
image1 = cv2.resize(image1, resize_size)
image2 = cv2.resize(image2, resize_size)
image3 = cv2.resize(image3, resize_size)
image4 = cv2.resize(image4, resize_size)
image5 = cv2.resize(image5, resize_size)

# Concatenate all images horizontally
combined_image = np.hstack((image1, image2, image3, image4, image5))

# Convert to RGB and grayscale
combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
combined_image_gray = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)

# Display both RGB and grayscale
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.imshow(combined_image_rgb)
plt.title('Top 5 Company')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(combined_image_gray, cmap='gray')
plt.title('Top 5 Company (Grayscale)')
plt.axis('off')

plt.tight_layout()
plt.show()


# In[10]:


import sys
from PyQt5.QtWidgets import *

# Top 10 Fortune companies dictionary
top10 = {
    1: {'company': 'Walmart', 'revenue ($ millions)': 514405.0},
    2: {'company': 'Exxon Mobil', 'revenue ($ millions)': 290212.0},
    3: {'company': 'Apple', 'revenue ($ millions)': 265595.0},
    4: {'company': 'Berkshire Hathaway', 'revenue ($ millions)': 247837.0},
    5: {'company': 'Amazon', 'revenue ($ millions)': 232887.0},
    6: {'company': 'UnitedHealth Group', 'revenue ($ millions)': 226247.0},
    7: {'company': 'McKesson', 'revenue ($ millions)': 208357.0},
    8: {'company': 'CVS Health', 'revenue ($ millions)': 194579.0},
    9: {'company': 'AT&T', 'revenue ($ millions)': 170756.0},
    10: {'company': 'AmerisourceBergen', 'revenue ($ millions)': 167939.6}
}

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_gui()

    def init_gui(self):
        self.setGeometry(300, 300, 360, 160)
        self.setWindowTitle('Top Fortune Company Checker')

        self.num_input = QLineEdit(self)
        self.num_input.setPlaceholderText("Enter a number (1-10)")

        self.check_button = QPushButton("Check Company", self)
        self.result_label = QLabel("Company info will appear here", self)

        self.check_button.clicked.connect(self.check_company)

        layout = QVBoxLayout()
        layout.addWidget(self.num_input)
        layout.addWidget(self.check_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.show()

    def check_company(self):
        user_input = self.num_input.text()

        if user_input.isdigit():
            index = int(user_input)
            if index in top10:
                company_info = top10[index]
                name = company_info['company']
                revenue = company_info['revenue ($ millions)']
                self.result_label.setText(f"Company: {name}\nRevenue: ${revenue:,} million")
            else:
                self.result_label.setText("Please enter a number between 1 and 10.")
        else:
            self.result_label.setText("Invalid input. Please enter a number.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


# In[ ]:




