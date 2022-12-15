#!/usr/bin/env python3
'''inserts a new document in a collection based on kwargs'''


def insert_school(mongo_collection, **kwargs):
    """
    - Write a Python function that inserts a new document in
a collection based on kwargs:
      + Prototype: def insert_school(mongo_collection, **kwargs):
      + mongo_collection will be the pymongo collection object
      + Returns the new _id
    """

    return mongo_collection.insert_one(kwargs).inserted_id
