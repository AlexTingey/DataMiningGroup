import requests
import datetime
from datetime import timezone
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
#load 
tokenizer = AutoTokenizer.from_pretrained
sentiment_classes = ["positive", "negative", "neutral"]
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
						
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

class Post:
    date = None
    permalink = None
    is_video = None
    num_comments = None
    score = None
    gilded = None
    title = None
    comments = None
    body = None
    id = None

    def __init__(self):
        num_comments = 0
def create_feature_vec_for_post(post):
    post_body = post.body

    if len(post_body) > 512:
        post_body = post_body[:512]
    post_score = post.score
    
    split_body = post_body.split(' ')

    btc_counter = 0

    for word in split_body:
        if word.lower() == 'btc':
            btc_counter += 1

    tokenize_input = tokenizer.encode_plus(post_body, return_tensors="pt")
    classification_logits = model(**tokenize_input)[0]
    model_evaluation = torch.softmax(classification_logits, dim=1).tolist()

    number_words = len(split_body)

    model_eval_to_list = list(model_evaluation)

    feature_vector_as_list = [number_words, post_score, btc_counter]

    for data in model_eval_to_list[0]:
        feature_vector_as_list.append(data)

    feature_vector_to_numpy = np.array(feature_vector_as_list)
    
    return feature_vector_to_numpy

    
def create_feature_vec(input_data):
    comment_body = input_data['body']

    if len(comment_body) > 512:
        comment_body = comment_body[:512]

    words_in_comment = comment_body.split(' ')

    btc_counter = 0
    for word in words_in_comment:
        if word.lower() == 'btc':
            btc_counter += 1
    
    score = input_data['score']

    tokenize_input = tokenizer.encode_plus(comment_body, return_tensors="pt")
    classification_logits = model(**tokenize_input)[0]
    model_evaluation = torch.softmax(classification_logits, dim=1).tolist()
    number_words = len(words_in_comment)
    
    model_eval_to_list = list(model_evaluation)


    feature_vector_as_list = [number_words, score, btc_counter]

    for data in model_eval_to_list[0]:
        feature_vector_as_list.append(data)

    feature_vector_to_numpy = np.array(feature_vector_as_list)
    
    return feature_vector_to_numpy

subreddit = "bitcoin"
postList = []
start_date = datetime.date(2016, 4, 2)
end_date = datetime.date(2018, 5, 3)
delta = datetime.timedelta(days=1)
data = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

while start_date <= end_date:
    next_date = start_date + delta
    #TODO: Find a solution to the potential problem that api only returns 100
    #      results per request. May be problematic for both comments&posts
    reqStr = ("https://api.pushshift.io/reddit/search/submission/?subreddit=" + 
    subreddit + "&sort=desc&sort_type=created_utc&after=" + str(start_date) + 
    "&before=" + str(next_date) +"&size=100&aggs=link_id")
    resp = requests.get(reqStr)
    if resp.status_code ==200:
        start_date = next_date
        json = resp.json()

        avg_vec = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        count = 0

        if len(json["data"])!=0:
            for p in range(len(json["data"])):
                if count > 100:
                    break
                try:
                    postInfo = json["data"][p]
                    post = Post()
                    post.date = start_date
                    post.permalink = postInfo["permalink"]
                    post.num_comments = postInfo["num_comments"]
                    post.score = postInfo["score"]
                    post.title = postInfo["title"]
                    post.body = postInfo["selftext"]
                    post.comments = []
                    post.id = postInfo["id"]
                    #make requests for comments here,add comments to list
                    commentsRetrieved = False
                    post_feature_vec = create_feature_vec_for_post(post)
                    avg_vec += post_feature_vec
                    count += 1
                except Exception:
                    continue
                while commentsRetrieved == False:
                    commentIdReq ="https://api.pushshift.io/reddit/submission/comment_ids/" + post.id
                    resp = requests.get(commentIdReq)
                    #Make sure we try again until we recieve the response to query
                    if(resp.status_code == 200):
                        json2 = resp.json()
                        if(len(json2["data"]) != 0):
                            commentIDS = json2["data"]
                            #Loop through each comment id, making a request for
                            #each comment and storing its data
                            thisCRet = False

                            for id in commentIDS:

                                while thisCRet ==False:
                                  commentReq = "https://api.pushshift.io/reddit/comment/search?ids=" + id
                                  cResp = requests.get(commentReq)
                                  if cResp.status_code ==200:
                                      try:
                                        commentData = cResp.json()['data'][0]
                                        score = commentData['score']
                                        body = commentData['body']
                                        thisCRet = True
                                        #Create the feature vec
                                        comment_features = create_feature_vec(commentData)
                                        avg_vec += comment_features
                                        count += 1
                                        if count > 100:
                                            commentsRetrieved = True
                                            thisCRet = True
                                            break
                                      except Exception:
                                          continue
                            commentsRetrieved = True
                        else:
                          commentsRetrieved = True
                    elif(resp.status_code == 500):
                        commentsRetrieved = True
        
        avg_vec = avg_vec/count
        avg_vec = np.insert(avg_vec, 0, time.mktime(start_date.timetuple()))
        data = np.vstack((data, avg_vec))
        np.savetxt('data.csv', data, delimiter=",")

        #If  we don't get a good response code we try again