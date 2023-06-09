#api
from flask import Flask, jsonify
import time
#data
import pandas as pd
import pymysql
#model
import pickle
from surprise import Prediction
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


file_path = 'model/model_new.pkl'
with open(file_path, 'rb') as file:
    model = pickle.load(file)


connection = pymysql.connect(
    host= host,
    user = user,
    password = password,
    database= db_name
)
cursor = connection.cursor()

previous_predictions_cf = {}

##scheduling and clear cache every 1 week

cache_expiration_time = 7 * 24 * 60 * 60 
last_cache_clear_time = time.time()

def clear_cache():
    global last_cache_clear_time
    last_cache_clear_time = time.time()
    previous_predictions_cf.clear()

def is_cache_expired():
    current_time = time.time()
    time_since_last_clear = current_time - last_cache_clear_time
    return time_since_last_clear >= cache_expiration_time

############################################################## Load data #################################################################################################

####### CF #####
ratings = 'SELECT pr.user_id, pr.pack_id, pr.user_rating, p.package_name, p.budget FROM package_rating pr JOIN detail_package p ON pr.pack_id = p.pack_id'
cursor.execute(ratings)

##### CBF ####
def load_data_from_sql(connection):
    detail_activity_query = "SELECT * FROM detail_activity"
    activity_rating_query = "SELECT * FROM activity_rating"

    cursor.execute(detail_activity_query)
    cursor.execute(activity_rating_query)

    detail_activity = pd.read_sql(detail_activity_query, connection)
    activity_rating = pd.read_sql(activity_rating_query, connection)

    return detail_activity, activity_rating


#prep data
####### CF #####
rating_df = pd.read_sql(ratings, connection)
ratings = rating_df[['user_id','pack_id','user_rating']]
avg_ratings = rating_df.groupby(['pack_id','package_name','budget'])['user_rating'].mean().reset_index()
top_rated_packs = avg_ratings.sort_values(by='user_rating', ascending=False)
top_rated_packs.head()

reader = Reader()
data = Dataset.load_from_df(ratings[['user_id','pack_id','user_rating']], reader)
trainset, testset = train_test_split(data, test_size=0.20, random_state=50)
predictions = model.test(testset)

def is_uid_valid(user_id):
    query = f"SELECT user_id FROM user_table WHERE user_id = '{user_id}'"
    result = cursor.execute(query)
    if result :
        return True
    else:
        return False


##### CBF #####
def calculate_average_ratings(detail_activity, activity_rating):
    average_ratings = activity_rating.groupby('activity_id')['user_rating'].mean().reset_index()
    merged_data1 = pd.merge(detail_activity, average_ratings, on='activity_id', how='inner')
    return merged_data1

def calculate_similarity_matrices(merged_data1):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(merged_data1['activity_name'] + ' ' + merged_data1['activity_category'] + ' ' + merged_data1['activity_level'])
    cos_sim_matrix = cosine_similarity(tfidf_matrix)
    return cos_sim_matrix

app = Flask(__name__)

#### RecSys #####

#cf
def get_top_predictions(predictions, user_id, top_rated_packs, rating_df):
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    if len(user_predictions) >= 2:
        top_predictions = user_predictions[:2]
        pack_count = 2
    elif len(user_predictions) == 1:
        top_predictions = user_predictions[:1]
        pack_count = 3
    else:
        top_predictions = []
        pack_count = 4
    
    predicted_pack_ids = set([pred.iid for pred in top_predictions])
    
    for _, row in top_rated_packs.iterrows():
        pack_id = row['pack_id']
        if pack_id not in predicted_pack_ids:
            package_name = rating_df.loc[rating_df['pack_id'] == pack_id, 'package_name'].iloc[0]
            budget = rating_df.loc[rating_df['pack_id'] == pack_id, 'budget'].iloc[0]
            top_predictions.append(Prediction(uid=user_id, iid=pack_id, r_ui=None, est=row['user_rating'], details={'was_impossible': False}))
            predicted_pack_ids.add(pack_id)
            pack_count -= 1
            if pack_count == 0:
                break
    
    if len(top_predictions) == 0:
        for _, row in top_rated_packs.iterrows():
            pack_id = row['pack_id']
            if pack_id not in predicted_pack_ids:
                package_name = rating_df.loc[rating_df['pack_id'] == pack_id, 'package_name'].iloc[0]
                budget = rating_df.loc[rating_df['pack_id'] == pack_id, 'budget'].iloc[0]
                
                top_predictions.append(Prediction(uid=user_id, iid=pack_id, r_ui=None, est=row['user_rating'], details={'was_impossible': False}))
                predicted_pack_ids.add(pack_id)
                pack_count -= 1
                if pack_count == 0:
                    break
    return top_predictions

def store_db_cf(user_id, predictions):
    for pred in predictions:
        pack_id = pred.iid

        package_name = rating_df.loc[rating_df['pack_id'] == pack_id, 'package_name'].iloc[0]
        user_rating = top_rated_packs.loc[top_rated_packs['pack_id'] == pack_id, 'user_rating'].iloc[0]
        budget = rating_df.loc[rating_df['pack_id'] == pack_id, 'budget'].iloc[0]
        user_rating = float(user_rating)

        query = "SELECT * FROM package_recommendations WHERE user_id = %s AND pack_id = %s"
        values = (user_id, pack_id)
        cursor.execute(query, values)
        result = cursor.fetchone()

        if result is None:
            query = "INSERT INTO package_recommendations (user_id, pack_id, avg_rating, package_name, budget) VALUES (%s, %s, %s, %s, %s)"
            values = (user_id, pack_id, round(pred.est,1), package_name, budget)

            cursor.execute(query, values)
    connection.commit()

#cbf
detail_activity, activity_rating = load_data_from_sql(connection)
merged_data1 = calculate_average_ratings(detail_activity, activity_rating)
cos_sim_matrix = calculate_similarity_matrices(merged_data1)

def get_recommend_activities(activity_name, activity_category, activity_level, cos_sim_matrix, top_n=3):
    activity_indices = merged_data1[(merged_data1['activity_name'] == activity_name) &
                                    (merged_data1['activity_category'] == activity_category) &
                                    (merged_data1['activity_level'] == activity_level)].index
    if len(activity_indices) > 0:
        activity_index = activity_indices[0]
        similarity_scores = cos_sim_matrix[activity_index]
        similar_activities_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
        similar_activities = merged_data1.loc[similar_activities_indices, 'activity_name']

        similar_activities = [activity_name] + similar_activities.tolist()
        similar_activities = list(dict.fromkeys(similar_activities))

        return similar_activities
    else:
        print("Activity not found in dataset.")
        return []
    
@app.route('/homepage/<id>/cf_recommendation', methods=['GET'])
def CF_RecSys(id):
    user_id = id

    if is_cache_expired():
        clear_cache()
     # Check if the user ID exists in the user table
    if not is_uid_valid(user_id):
        response = {'message': 'Invalid user ID'}
        return jsonify(response), 400
    
    if user_id in previous_predictions_cf:
            top_predictions = previous_predictions_cf[user_id]
    else:
        top_predictions = get_top_predictions(predictions, user_id, top_rated_packs, rating_df)
        previous_predictions_cf[user_id] = top_predictions
        store_db_cf(user_id, top_predictions)

    predictions_list = []
    for pred in top_predictions:
        pack_id = pred.iid
        package_name = rating_df.loc[rating_df['pack_id'] == pack_id, 'package_name'].iloc[0]
        user_rating = top_rated_packs.loc[top_rated_packs['pack_id'] == pack_id, 'user_rating'].iloc[0]
        budget = rating_df.loc[rating_df['pack_id'] == pack_id, 'budget'].iloc[0]
        budget = "Rp {:,}".format(int(budget)).replace(',', '.')
        prediction_dict = {
            'pack_id': pack_id,
            'package_name': package_name,
            'user_rating': user_rating,
            'budget': budget
        }
        predictions_list.append(prediction_dict)

    response = predictions_list
    return jsonify(response), 200
    
@app.route('/homepage/<id>/cbf_recommendation', methods=['GET'])
def get_cbf_recommendations(id):
    activity_query = f"SELECT activity_name FROM questionnaire_responses WHERE user_id = '{id}'"
    cursor.execute(activity_query)
    result = cursor.fetchone()

    if result is not None:
        activity_name = result[0]  
        activity_data = merged_data1[merged_data1['activity_name'] == activity_name].iloc[0]
        activity_id = activity_data['activity_id']
        activity_category = activity_data['activity_category']
        activity_level = activity_data['activity_level']
        budget = "Rp {:,}".format(int(activity_data['budget'])).replace(',', '.')

        similar_activities = get_recommend_activities(activity_name, activity_category, activity_level, cos_sim_matrix, top_n=3)
        if similar_activities:
            recommendations = []
            for i, activity in enumerate(similar_activities[:4]): 
                activity_data = merged_data1[merged_data1['activity_name'] == activity].iloc[0]
                activity_id = activity_data['activity_id']
                ratings = activity_data['user_rating']
                aver_rating = ratings.mean()
                budget =  activity_data['budget'] 

                activity_id = int(activity_id)
                aver_rating = float(aver_rating) 
                recommendations.append({
                    'activity_id':  activity_id,
                    'activity_name': activity,
                    'avg_rating': round(aver_rating, 1),
                    'budget': "Rp {:,}".format(int(activity_data['budget'])).replace(',', '.')
                })

                insert_query = "INSERT INTO activity_recom(activity_id, activity_name, avg_rating, budget) VALUES (%s, %s, %s,%s)"
                cursor.execute(insert_query, (activity_id, activity, round(aver_rating, 1),budget))
                connection.commit()

            response = {
                'recommendations': recommendations
            }
            return jsonify(response)
        else:
            return jsonify({'message': 'Activity not found in the database.'})
    else:
        return jsonify({'message': 'No activity name found for the given user ID.'})

if __name__ == '__main__':
    
    app.run(debug=True)