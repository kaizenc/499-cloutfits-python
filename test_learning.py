from learning import generate_matrix, train_model,\
    make_item_prediction, make_outfit_prediction
import requests
import pymongo


def fake_ratings():
    return {
        '123': {
            '1': 8,
            '2': 7,
            '3': 5,
            '4': 5,
            '5': 5,
            '6': 5,
            '8': 1
        },
        '126': {
            '7': 8,
            '2': 7,
            '3': 5,
            '4': 5,
            '5': 5,
            '6': 5,
            '8': 1
        }
    }


def fake_outfits():
    return {
        # '234': ['1', '2', '3'],
        # '235': ['1', '2', '4'],
        # '236': ['1', '5', '6'],
        # '334': ['7', '2', '3'],
        # '335': ['7', '2', '4'],
        # '336': ['7', '5', '6'],
        '431': ['1', '2', '6'],
        '432': ['8', '5', '3']
    }


if __name__ == "__main__":
    if False:
        df = generate_matrix(fake_ratings())
        model = train_model(df)
        print(make_item_prediction('123', '7', model))
        print(make_item_prediction('126', '1', model))
        make_outfit_prediction('123', fake_outfits(), model)
        make_outfit_prediction('126', fake_outfits(), model)
    if False:
        response = requests.get("localhost:4000/items?limit=10")
        print(response.json())
    if False:
        c = pymongo.MongoClient()
        db = c.cloutfits
        items = db.items
        for i in items.find():
            print(i['id'])
