
# coding: utf-8

#manually create folders for each queue

from azure.storage.queue import QueueService
import base64
from ast import literal_eval
import json
import sys
import random
import re


# function to set filenemaes based on message data
def get_file_name(data, queue_name):
    # queues with different names and same information
    art_list = ['artifact', 'artifact-poison']
    dev_list = ['device','device-poison']
    mea_list = ['measure', 'measure-poison', 'measures-poison']
    per_list = ['person', 'person-poison', 'persons-poison']
    if queue_name in art_list:
        filename = r'{0}-{1}-{2}-{3}'.format(data['createdBy'].split("@")[0],queue_name[:3],data['createDate'],random.randint(1000, 9999))
        return filename, 'artifact' 
    elif queue_name in dev_list:
        filename = r'{0}-{1}-{2}-{3}'.format(data['owner'].split("@")[0],queue_name[:3],data['create_timestamp'],random.randint(1000, 9999))
        return filename, 'device'
    elif queue_name in mea_list:
        filename = r'{0}-{1}-{2}-{3}'.format(data['createdBy'].split("@")[0],queue_name[:3],data['timestamp'],random.randint(1000, 9999))
        return filename, 'measure'
    elif queue_name in per_list:
        filename = r'{0}-{1}-{2}-{3}'.format(data['createdBy'].split("@")[0],queue_name[:3],data['timestamp'],random.randint(1000, 9999))
        return filename, 'person'


def main():    
    queue_name = str(sys.argv[1])
    account_name = str(sys.argv[2])
    
    #checking if message is base64 encoded
    pattern = re.compile("^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$")  
    
    #account_keys to be stored in specific storage_account_name folders as "acc_key.txt"
    with open("{0}/acc_key.txt".format(account_name), "r") as f:
        account_key = f.read()

    queue_service = QueueService(account_name, account_key)

    metadata = queue_service.get_queue_metadata(queue_name)
    count = metadata.approximate_message_count

    #count = 50

    while count > 0:
        count -= 32
        messages = queue_service.get_messages(queue_name, num_messages=32)
        for message in messages:
            if pattern.match(message.content):
                msg = base64.b64decode(message.content)     # Received message is encoded in base 64
                data = json.loads(msg.decode('utf8'))
            else:
                msg = message.content
                data = json.loads(msg)
            filename, folder = get_file_name(data, queue_name)
            with open(r'/data/home/smahale/notebooks/localstorage/db/{0}/{1}/{2}.json'.format(account_name,folder,filename), 'w') as f:
                json.dump(data, f, indent=2)
            queue_service.delete_message(queue_name, message.id, message.pop_receipt)    
        

        
if __name__ == "__main__":
    main()        
