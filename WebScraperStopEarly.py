import requests
import datetime
from datetime import timezone
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import threading
#load 
						
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
proxyTxt = np.genfromtxt("webshare_100_proxies.txt",delimiter = ':',dtype = str)
proxyList = []
user = "czrglanp-dest"
passW = "poj6zxytidya"
for idx in range(len(proxyTxt[:,0])):
    proxyList.append("http://" + user + ":" + passW + "@" + proxyTxt[idx,0] + ":" + proxyTxt[idx,1])
    np.random.shuffle(proxyList)

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

#Can't figure out easy way to find difference in end_date to start_date and divide
# that number into num_Threads bins
"""
Method to start the scrubbing of reddit and make feature vectors
uses a proxy list divided into numThreads bins to query faster

inputs:
    numThreads - How many threads to run
    proxyList - Proxies to run all the threads
    start_date - DateTime argument representing the start date
    num_months - number of months you wish to see after the start date
"""
def start_scrubbing_threaded(numThreads,proxyList,start_date,num_months):
    monthsDif = int(num_months/numThreads)#Want the floor, simplest way for now
    
    proxDif = int(100/numThreads)
    startProx = 0
    for i in range(numThreads):
        oldStart = start_date
        endProx = startProx+proxDif
        pList = proxyList[startProx:endProx]
        startProx = endProx
        #Don't know behavior if day >=30, Febuary may fuck everything up if the initial day is >28
        if start_date.month ==12:
            end_date = start_date.replace(year=start_date.year+1,month = 1, day = start_date.day)
        else:
            end_date = start_date.replace(year=start_date.year,month = start_date.month+1, day = start_date.day)
        thr = threading.Thread(target=scrape_and_make_vectors, args=[start_date,end_date,pList])
        thr.start()
        start_date = end_date
    print("threads out")


def create_feature_vec_for_post(post,tokenizer):
    try:
        post_body = post.body

        if len(post_body) > 512:
            post_body = post_body[:512]
    except:
        post_body = ""
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

    
def create_feature_vec(input_data,tokenizer):
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

def scrape_and_make_vectors(start_date,end_date,proxyList):
    #Declare our tokenizers within this method, this way they can be thread safe
    tokenizer = AutoTokenizer.from_pretrained
    sentiment_classes = ["positive", "negative", "neutral"]
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


    subreddit = "bitcoin"
    postList = []

    delta = datetime.timedelta(days=1)
    data = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    proxyIdx = 0
    proxy = {"http":proxyList[proxyIdx]}
    while start_date <= end_date:
        next_date = start_date + delta
        #TODO: Find a solution to the potential problem that api only returns 100
        #      results per request. May be problematic for both comments&posts
        reqStr = ("https://api.pushshift.io/reddit/search/submission/?subreddit=" + 
        subreddit + "&sort=desc&sort_type=created_utc&after=" + str(start_date) + 
        "&before=" + str(next_date) +"&size=100&aggs=link_id")
        resp = requests.get(reqStr,proxies = proxy)
        if resp.status_code ==200:
            if(proxyIdx == len(proxyList)-1):
                proxyIdx = 0
                proxy = {'http':proxyList[proxyIdx]}
            else:
                proxyIdx = proxyIdx + 1
                proxy = {'http':proxyList[proxyIdx]}
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
                        if "selftext" in postInfo:
                            post.body = postInfo["selftext"]
                        post.comments = []
                        post.id = postInfo["id"]
                        #make requests for comments here,add comments to list
                        commentsRetrieved = False
                        post_feature_vec = create_feature_vec_for_post(post,tokenizer)
                        avg_vec += post_feature_vec
                        count += 1
                    except:
                        print("Error trying to process post: " + json["data"][p])
                        continue
                    commentCount = 0
                    while commentsRetrieved == False:
                        commentIdReq ="https://api.pushshift.io/reddit/submission/comment_ids/" + post.id
                        resp = requests.get(commentIdReq,proxies = proxy)
                        #Make sure we try again until we recieve the response to query
                        if(resp.status_code == 200):
                            if(proxyIdx == len(proxyList)-1):
                                proxyIdx = 0
                                proxy = {'http':proxyList[proxyIdx]}
                            else:
                                proxyIdx = proxyIdx + 1
                                proxy = {'http':proxyList[proxyIdx]}
                            json2 = resp.json()
                            if(len(json2["data"]) != 0):
                                commentIDS = json2["data"]
                                #Loop through each comment id, making a request for
                                #each comment and storing its data
                                thisCRet = False

                                for id in commentIDS:
                                    commentReq = "https://api.pushshift.io/reddit/comment/search?ids=" + id
                                    cResp = requests.get(commentReq,proxies = proxy)
                                    if cResp.status_code ==200:
                                        if proxyIdx == len(proxyList) - 1:
                                            proxyIdx = 0
                                            proxy = {'http':proxyList[proxyIdx]}
                                        else:
                                            proxyIdx += 1
                                            proxy = {'http':proxyList[proxyIdx]}
                                        try:
                                            commentCount += 1
                                            commentData = cResp.json()['data'][0]
                                            score = commentData['score']
                                            body = commentData['body']
                                            thisCRet = True
                                            #Create the feature vec
                                            comment_features = create_feature_vec(commentData,tokenizer)
                                            avg_vec += comment_features
                                            if commentCount > 100:
                                                commentsRetrieved = True
                                                break
                                        except:
                                            print("Error occured while attempting to process comment " + str(cResp.json()['data'][0]))
                                            if commentCount == len(commentIDS):
                                                commentsRetrieved = True
                                                
                                            continue
            avg_vec = avg_vec/count
            avg_vec = np.insert(avg_vec, 0, time.mktime(start_date.timetuple()))
            data = np.vstack((data, avg_vec))
    print("saving data")
    np.savetxt('data' + str(start_date) + '.csv', data, delimiter=",")


################################################
##############  MAIN METHOD HERE  ##############
################################################
start_date = datetime.date(2017,1,1)
start_scrubbing_threaded(12,proxyList,start_date,12)