
# coding: utf-8

#create directories 

from azure.storage.queue import QueueService
import base64
from ast import literal_eval
import json
import sys
import random
import re
import time
import os

def get_file_name(storage_account_name, queue_name, destination_folder):
    filename = '{0}-{1}-{2}-{3}.json'.format(storage_account_name, queue_name, int(time.time()),random.randint(10000, 99999))
    folder = '{0}/{1}/{2}/'.format(destination_folder, storage_account_name, queue_name)
    return filename, folder
   
def main():    
    storage_account_name = str(sys.argv[1])
    queue_name = str(sys.argv[2])
    destination_folder = str(sys.argv[3])
    
    if not storage_account_name or not queue_name or not destination_folder:
        print("usage: get_message.py storage_account_name queue_name destination_folder")
        sys.exit(1)
    
    base64_regex = re.compile("^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$")  
    
    key_file = "{0}/{1}/acc_key.txt".format(destination_folder, storage_account_name)
    try :
        with open(key_file, "r") as f:
            account_key = f.read()
    except IOError:
        print("file not found or empty ", key_file)
    
    queue_service = QueueService(storage_account_name, account_key)

    metadata = queue_service.get_queue_metadata(queue_name)
    approximate_message_count = metadata.approximate_message_count
    print('fetching approximately {0} messages'.format(approximate_message_count))
    messages = queue_service.get_messages(queue_name)
    count = len(messages)
    total = 0
    while count > 0:
        for message in messages:
            if base64_regex.match(message.content):
                msg = base64.b64decode(message.content)     # Received message is encoded in base 64
                data = json.loads(msg.decode('utf8'))
            else:
                msg = message.content
                data = json.loads(msg)
                
            filename, folder = get_file_name(storage_account_name, queue_name, destination_folder)
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(r'{0}/{1}'.format(folder,filename), 'w') as f:
                json.dump(data, f, indent=2)
            queue_service.delete_message(queue_name, message.id, message.pop_receipt) 
            count -= 1
            total += 1
            print('successfully processed {0} messages of ~{1} from {2} queue {3}'.format(total,approximate_message_count,storage_account_name,queue_name))

        messages = queue_service.get_messages(queue_name)
        count = len(messages)
            

        
if __name__ == "__main__":
    main()        
