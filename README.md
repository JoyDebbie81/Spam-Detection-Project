Spam Detection using Machine Learning
Project Overview:
This project focuses on classifying emails as spam or ham using machine learning. The dataset is the SpamAssassin Public Corpus, containing labeled spam and ham emails.



The project demonstrates the entire data science workflow:
Data loading and preprocessing

Feature extraction with TF-IDF vectorization

Training baseline models: Decision Tree, SVM, MLP

Applying ensemble methods: Voting and Stacking Classifiers

Model evaluation using accuracy, precision, recall, F1-score, and confusion matrices



Objectives: 
Build baseline models for spam detection

Improve performance using ensemble methods

Compare models using clear metrics and visualizations

Provide insights into which techniques are most effective for spam filtering




Technologies Used:
Python

Scikit-learn (ML models, vectorization, evaluation)

Matplotlib & Seaborn (visualization)

Pandas & NumPy (data handling)




Results - 
Baseline Models: Decision Tree, SVM, and MLP achieved good accuracy but varied in precision and recall.

Ensemble Models: Both Voting and Stacking outperformed individual models, with Stacking providing the best overall F1-score for spam detection.

Visualization: Confusion matrices and F1-score bar plots clearly show model performance differences.




Key Insights: 
MLP was the strongest single model, achieving the highest F1-score for spam detection.

Ensemble methods improved generalization and reduced misclassifications.

Stacking outperformed Voting, showing the benefit of a meta-learner.




How to Run:
Clone this repository

Install required libraries:

pip install -r requirements.txt




Open the Jupyter Notebook:
jupyter notebook spam_detection.ipynb

Run each cell sequentially


Future Work
Apply hyperparameter tuning for better accuracy
Explore deep learning approaches (RNN, LSTM, Transformers)

Deploy the model as a simple web app or API
