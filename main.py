import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder


def count_ads(group):
    return np.count_nonzero(group == "adShow")


def purchase(group):
    return np.count_nonzero(group == "in_app_purchase") > 0


def interstitial_count(group):
    return np.count_nonzero(group == "INTERSTITIAL")


def banners_count(group):
    return np.count_nonzero(group == "BANNERS")


def rewarded_video_count(group):
    return np.count_nonzero(group == "REWARDED_VIDEO")


def is_unlikely_to_purchase(group):
    return not any(group)  # Returns True if user has made no purchases in any session (user is unlikely to purchase)


dataset = pd.read_csv("ds_test_data.csv")

grouped_by_sessions = dataset.groupby(by=["user_pseudo_id", "session_id"], as_index=False).aggregate({
    "event_name": [count_ads, purchase],
    "time_in_session": "max",
    "ad_type": [interstitial_count, banners_count, rewarded_video_count],
    "mobile_brand_name": "first",
})
# print(f"Result: {grouped_by_sessions.to_string()}")
sessions = pd.DataFrame({
    "total_time_in_session": grouped_by_sessions["time_in_session"]["max"].values,
    "total_ads": grouped_by_sessions["event_name"]["count_ads"].values,
    "total_interstitial": grouped_by_sessions["ad_type"]["interstitial_count"].values,
    "total_banners": grouped_by_sessions["ad_type"]["banners_count"].values,
    "total_video_rewards": grouped_by_sessions["ad_type"]["rewarded_video_count"].values,
    "mobile_brand": LabelEncoder().fit_transform(grouped_by_sessions["mobile_brand_name"]["first"].values),
    "user": grouped_by_sessions["user_pseudo_id"],
    "purchase_occurred": grouped_by_sessions["event_name"]["purchase"].values
})
# print(f"\nSessions Dataset: \n{sessions.to_string()}")

grouped_by_users = sessions.groupby(by="user").aggregate({
    "total_time_in_session": "mean",
    "total_ads": "mean",
    "total_interstitial": "mean",
    "total_banners": "mean",
    "total_video_rewards": "mean",
    "mobile_brand": "first",
    "purchase_occurred": is_unlikely_to_purchase
})
# print(f"\nGrouped by users: \n{grouped_by_users.to_string()}")

users_input_dataset = pd.DataFrame({
    "avg_time_in_session": grouped_by_users["total_time_in_session"].values,
    "avg_ads": grouped_by_users["total_ads"].values,
    "avg_interstitials": grouped_by_users["total_interstitial"].values,
    "avg_banners": grouped_by_users["total_banners"].values,
    "avg_video_rewards": grouped_by_users["total_video_rewards"].values,
    "mobile_brand": grouped_by_users["mobile_brand"].values,
})
users_output_dataset = pd.DataFrame({
    "is_unlikely_to_purchase": grouped_by_users["purchase_occurred"].values
})
# print(f"\nUsers input dataset: \n{users_input_dataset.to_string()}")
# print(f"\nUsers output dataset: \n{users_output_dataset.to_string()}")

x_train, x_test, y_train, y_test = train_test_split(users_input_dataset, users_output_dataset, test_size=0.2, random_state=42)

print(f"Train length {len(x_train)}, test length: {len(x_test)}")

classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(x_train, y_train)

print(f"Tree depth: {classifier.get_depth()}")
print(f"Accuracy: {classifier.score(x_test, y_test)*100}%")
y_predicted = classifier.predict(x_test)
print(f"Precision: {precision_score(y_test, y_predicted)*100}%")
