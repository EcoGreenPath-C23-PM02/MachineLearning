# Machine Learning Recommendation System

This project is a machine learning-based recommendation system that utilizes two algorithms: Content Based Filtering and Collaborative Filtering. The goal of the system is to provide personalized recommendations for activities based on user preferences.

## Content Based Filtering

The Content Based Filtering algorithm generates recommendations by calculating the cosine similarity between the activity_level and activity_category features. The system identifies activities with similar attribute values and recommends them to users who have shown interest in similar activities in the past.

## Collaborative Filtering

The Collaborative Filtering algorithm uses the Singular Value Decomposition (SVD) model to predict user preferences based on similarities among users. By analyzing user behavior and preferences, the system identifies patterns and makes recommendations based on the preferences of similar users.

## API Development

After analyzing the data and building the recommendation system, we have created a machine learning API using Python Flask. The API allows users to interact with the recommendation system through HTTP requests. It provides endpoints for users to input their preferences and receive personalized recommendations based on the selected algorithm.

## Repository Structure

- `ContentBased_CosSim.ipynb`: Contains the implementation of the Content Based Filtering algorithm.
- `cf2.py`: Contains the implementation of the Collaborative Filtering algorithm.
- `ml_api.py`: Implements the Flask API for interacting with the recommendation system.
- `trial model/`: Directory containing the dataset used for training and testing the recommendation system.
- `requirements.txt`: Lists the required dependencies for running the project.

## Getting Started

To replicate this project, follow these steps:

1. Clone the repository: `git clone https://github.com/EcoGreenPath-C23-PM02/MachineLearning.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the API: `ml_api.py`
4. Send HTTP requests to the API endpoints to interact with the recommendation system.

Please refer to the code files and the API documentation for detailed instructions on how to use and customize the recommendation system.

