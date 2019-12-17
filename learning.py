import pandas as pd
import pickle
import pymongo
from bson.objectid import ObjectId

from funk_svd import SVD

'''
Intra Outfit Similarity: How well outfits work within themselves
    * look into concepts, + possibly learn what are popular outfit combos based on upper half of voted fits
    * look into color combinations

Inter Outfit Similarity: How well outfits work with a user's taste
    * Rank a user's preferred concepts + styles, rank a fit higher based on its items + concepts
'''
URI = 'mongodb+srv://user_1:pass_1@cluster0-eqftc.mongodb.net/test'


def pull_all_ratings():
    c = pymongo.MongoClient(URI)
    db = c.cloutfits
    users = db.users
    ratings = {}
    for i in list(users.find()):
        if 'votedItems' not in i:
            continue
        ratings[i['username']] = i['votedItems']
    return ratings


def generate_matrix(ratings):
    dfObj = pd.DataFrame(columns=['u_id', 'i_id', 'rating'])
    counter = 0
    for u_id in ratings:
        for i_id in ratings[u_id]:
            dfObj.loc[counter] = [u_id, i_id, ratings[u_id][i_id]]
            counter += 1
    return dfObj


def train_model(df):
    train = df.sample(frac=0.8, random_state=7)
    val = df.drop(train.index.tolist()).sample(frac=1.0, random_state=8)

    svd = SVD(learning_rate=0.1, regularization=0.005, n_epochs=10,
              n_factors=10, min_rating=1, max_rating=10)

    svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

    outfile = open('svd_model', 'wb')
    pickle.dump(svd, outfile)
    outfile.close()


def make_item_prediction(u_id, i_id, model):
    test = pd.DataFrame({
        'u_id': [u_id],
        'i_id': [i_id]
    })
    return model.predict(test)


def pull_user_outfits(u_id):
    c = pymongo.MongoClient(URI)
    db = c.cloutfits
    users = db.users
    current_user = users.find_one({"username": u_id})
    gender = current_user["gender"]

    # Don't give a user their own outfits, or outfits they've seen
    outfitIds = [ObjectId(x) for x in list(current_user["outfitsIds"])]
    upvotedIds = [ObjectId(x) for x in list(current_user["upvotedOutfits"])] + outfitIds
    outfitIds.append(upvotedIds)
    outfitIds.append([ObjectId(x) for x in list(current_user["downvotedOutfits"])])

    outfits = db.outfits
    result = []
    favorites = []
    for i in list(outfits.find()):
        if gender and (i["gender"] != gender):
            continue
        if i['_id'] not in outfitIds:
            # append {outfitId: [itemid1, itemid2, ... ]}
            result.append({i['_id']: [x for x in list(i.values())[2:] if x is not None]})
        if i['_id'] in upvotedIds:
            favorites.append({i['_id']: [x for x in list(i.values())[2:] if x is not None]})
    return result, favorites


def users_preferences(favorites, items):
    unique_items = []
    concept_dict = {}
    for i in favorites:
        for j in list(i.values())[0]:
            if j not in unique_items:
                unique_items.append(j)
    for i in unique_items:
        current_concept = items[f'{i}']['concepts']
        if current_concept not in concept_dict:
            concept_dict[current_concept] = 1
        concept_dict[current_concept] += 1
    sorted_concepts = [k for k, v in sorted(concept_dict.items(), key=lambda item: item[1])]
    # rank the concepts, return them
    # re-write outfit prediction to add it in
    result = {}
    for i in range(len(sorted_concepts)):
        if i == 4:  # return top 4
            break
        result[sorted_concepts[i]] = 5 - i
    return result


def make_outfit_prediction(u_id, model, limit):
    c = pymongo.MongoClient(URI)
    db = c.cloutfits
    all_items = {x['id']: x for x in list(db.items.find())}
    unique_items = []
    outfits, favorites = pull_user_outfits(u_id)
    preferences = users_preferences(favorites, all_items)

    # Run through model
    for i in outfits:
        for j in list(i.values())[0]:
            if j not in unique_items:
                unique_items.append(j)
    test = pd.DataFrame({
        'u_id': [u_id for x in unique_items],
        'i_id': unique_items
    })
    _pred = model.predict(test)
    predictions = {unique_items[i]: _pred[i] for i in range(len(_pred))}

    # Create an outfit ranking using those predictions
    outfit_values = {list(i.keys())[0]: 0 for i in outfits}
    for i in outfits:
        cnt = 0
        for j in list(i.values())[0]:
            outfit_values[list(i.keys())[0]] += predictions[j]
            if all_items[f'{j}']['concepts'] in preferences:
                outfit_values[list(i.keys())[0]] += (preferences[all_items[f'{j}']['concepts']]) / 10
            cnt += 1
        outfit_values[list(i.keys())[0]] = outfit_values[list(i.keys())[0]] / cnt
    result = [str(k[0]) for k in sorted(outfit_values.items(), key=lambda item: item[1])][:int(limit)]
    # Connect to mongo and get that outfit
    return result
