#!/usr/bin/env python3
"""
inserts a new document in a collection based on kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """
    - Python function that changes all topics of a school
document based on the name:
    + Prototype: def update_topics(mongo_collection, name, topics):
    + mongo_collection will be the pymongo collection object
    + name (string) will be the school name to update
    + topics (list of strings) will be the list of topics
approached in the school
    + Returns the new _id
    """

    return mongo_collection.insert_one(kwargs).inserted_id
