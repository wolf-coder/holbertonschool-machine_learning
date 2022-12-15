#!/usr/bin/env python3
"""script that provides some stats about Nginx logs stored in MongoDB"""
from pymongo import MongoClient


if __name__ == "__main__":
    """ script that provides some stats abfdjjjjjjjjjjjjjjjjjout Nginx logs stored in MongoDB"""
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    counts = logs.count_documents({})
    print("{} logs".format(counts))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        nmethod = logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, nmethod))
    filter_path = {"method": "GET", "path": "/status"}
    npath = logs.count_documents(filter_path)
    print("{} status check".format(npath))
