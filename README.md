# Java Spam Classifier

A Spam Email Classifier built entirely from the ground up in pure Java, without any external machine learning libraries. This project demonstrates a deep, fundamental understanding of the Naive Bayes algorithm and the core principles of text classification.

---

## How It Works

The classifier is based on **Bayes' Theorem**. In simple terms, it calculates the probability of an email being "spam" or "ham" based on the words it contains.

1.  **Training:** The model first learns from a labeled dataset. It counts the frequency of every word in both spam and ham emails, building two separate probability models.
2.  **Prediction:** When a new email arrives, the classifier calculates two scores:
    -   The probability that this email is **spam**, given its words.
    -   The probability that this email is **ham**, given its words.
3.  **Classification:** The email is assigned the label with the higher probability score.

---

## Key Features

-   **Pure Java Implementation:** The entire Naive Bayes algorithm and its components are built using only standard Java libraries.
-   **Advanced Text Preprocessing:** Includes tokenization, conversion to lowercase, and removal of common "stop words" to focus on meaningful terms.
-   **Laplace (Add-1) Smoothing:** A crucial technique implemented to prevent zero-probability errors when encountering words not seen during training, making the classifier more robust.
-   **Custom CSV Parser:** A robust, from-scratch parser built to handle the complexities of the dataset, including multi-line fields, commas, and quotes within the email text.

---

## Getting Started

Follow these steps to run the classifier on your own machine.

### **Prerequisites**
-   Java JDK 11 or higher
-   Git for cloning the repository

### **1. Clone the Repository**
Open your terminal and run the following command:
```bash
git clone [https://github.com/Tanmay-hue/Email_Spam_Classifier](https://github.com/Tanmay-hue/Email_Spam_Classifier)
cd Email_Spam_Classifier
```
*(Remember to replace `YOUR_USERNAME` with your actual GitHub username!)*

### **2. Download the Dataset**
This project is trained on the "Spam Ham Dataset" from Kaggle.
-   **Download the file here:** [https://www.kaggle.com/datasets/venky73/spam-mails-dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
-   After downloading, rename the file to `spam_ham_dataset.csv`.
-   Place the `spam_ham_dataset.csv` file inside the `Email_Spam_Classifier` folder you just cloned.

### **3. Compile and Run**
With the terminal open in the project's root directory, execute the following commands:

```bash
# Compile the Java source code
javac SpamClassifier.java

# Run the classifier
java SpamClassifier
```
The program will then load the data, train the model, and print the final accuracy on the test set.

---
