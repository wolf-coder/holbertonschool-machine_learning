#!/usr/bin/env python3
""" og stats - new version """
from pymongo import MongoClient


if __name__ == "__main__":
    """script that provides some stats about Nginx logs stored in MongoDB"""
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    num_doc = logs.count_documents({})
    print("{} logs".format(num_doc))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        nmethod = logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, nmethod))
    filter_path = {"method": "GET", "path": "/status"}
    npath = logs.count_documents(filter_path)
    print("{} status check".format(npath))
    print("IPs:")
    pipeline = [{"$group": {"_id": "$ip", "count": {"$sum": 1}}}]
    ips = logs.aggregate(pipeline)
    ips_list = []
    for ip in ips:
        ips_list.append(ip)
    sorted_ips = sorted(ips_list, key=lambda i: i["count"], reverse=True)
    i = 0
    limit = 10
    if len(sorted_ips) < 10:
        limit = len(sorted_ips)
    while i < limit:
        ip = sorted_ips[i]["_id"]
        count = sorted_ips[i]["count"]
        print("\t{}: {}".format(ip, count))
        i += 1
