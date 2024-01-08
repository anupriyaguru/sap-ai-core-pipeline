from hana_ml import dataframe
from flask import Flask, request, json
import pandas as pd
import requests
import re
import uuid
import os


KEY_FILE_GPT = "/app/credentials/key.json"
KEY_FILE_HANA = "/app/credentials/hana-connection.json"

TRANSACTION_SOURCE = "Transaction"
TRANSACTION_TARGET = "NEW_TRANSACTION_CATEGORY"

# KEY_FILE_GPT = "credentials/key.json"
# KEY_FILE_HANA = "credentials/hana-connection.json"

class GPT:
    def __init__(self, model="gpt-4-32k"):
        self.model = model
        with open(KEY_FILE_GPT, "r") as key_file:
            svc_key = json.load(key_file)
        self.svc_key = svc_key
        self.svc_url = svc_key["url"]
        self.client_id = svc_key["uaa"]["clientid"]
        self.client_secret = svc_key["uaa"]["clientsecret"]
        self._get_token()

    def _get_token(self):
        uaa_url = self.svc_key["uaa"]["url"]
        params = {"grant_type": "client_credentials" }
        resp = requests.post(
            f"{uaa_url}/oauth/token",
            auth=(self.client_id, self.client_secret),
            params=params
        )
        self.token = resp.json()["access_token"]
        self.headers = {
            "Authorization":  f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def createPromptBatch(self, targetCategories, transactions):
        transactionString = '\n'.join([f"| {index} | {transaction} |" for transaction, index in transactions.items()])
        prompt = f'''Categorize the numerized descriptions below into the {len(targetCategories)} categories {targetCategories}.

Display the result in a table with the format "| INDEX OF DESCRIPTION | CATEGORY |".
Descriptions:
{transactionString}'''
        return prompt
        
    def get_response(self, prompt):
        data = {
            "deployment_id": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 1.0,
            "n": 1
        }
        response = requests.post(
            f"{self.svc_url}/api/v1/completions",
            headers=self.headers,
            json=data
        )
        try:
            response = str(response.json()['choices'][0]['message']['content'])
        except:
            self._get_token()
            response = requests.post(
                f"{self.svc_url}/api/v1/completions",
                headers=self.headers,
                json=data
            )
            response = str(response.json()['choices'][0]['message']['content'])
        return response    

class App(Flask):
    def __init__(self):
        super().__init__(__name__)

        with open(KEY_FILE_HANA) as file:
            self.db_key = json.load(file)

        self.add_url_rule("/v2/inference", "inference", self.inference, methods=["POST"])
    
    def connectToHANA(self):
        # connect with HANA Databse
        return dataframe.ConnectionContext(
            address  = self.db_key["ADDRESS"],
            port     = self.db_key["PORT"],
            user     = self.db_key["USER"],
            password = self.db_key["PASSWORD"])

    def inference(self):
         # generate unique uuid for this process
        processID = uuid.uuid4()

        # get HANA connection
        conn = self.connectToHANA()

        # collect target categories
        # categories = pd.DataFrame(conn.table(targetCategories, self.db_key['USER']).select('ID', 'DESCRIPTION').collect()).iloc[:,1].values

        categories = request.json["categories"]

        if(categories == None or type(categories) != list or len(categories) == 0):
            return {"error" : "invalid body data"}, 404

        # collect transactions
        transactions = pd.DataFrame(conn.table(TRANSACTION_SOURCE, self.db_key['USER']).select('ID', 'DESCRIPTION').collect())

        # get distinct description from all transactions to reduce token count of the chatGPT prompt
        transactionDescriptions = { description : index for index, description in enumerate(transactions.filter(["DESCRIPTION"]).drop_duplicates().values.flatten()) }

        # create ChatCPT instance
        gpt = GPT()

        # create prompt with categories
        chatGPTPromptBatch = gpt.createPromptBatch(categories, transactionDescriptions)
        with open(f"logs/prompt-{processID}.txt", "w+") as file:
            print("Writing prompt to file")
            file.write(chatGPTPromptBatch)

        # get chatGPT completion for given prompt
        completion = gpt.get_response(chatGPTPromptBatch)
        with open(f"logs/completion-{processID}.txt", "w+") as file:
            print("Writing completion to file")
            file.write(completion)

        # get all table rows
        allFindings = re.findall("^\| ?(\d*) ?\| ?(.*?) ?\|$", completion, re.MULTILINE)

        # turn findings into dictionary for easy access 
        descriptionLookUp = dict(allFindings)

        # get Category from transaction - description
        def lookUp(x):
            return descriptionLookUp.get(str(transactionDescriptions.get(x, 0)), "OTHER")

        # create new coloumn with the new category from chatGPT for each   
        transactions["CATEGORY"] = transactions["DESCRIPTION"].apply(lookUp)

        # format transaction data for result HANA table
        new_transactions = transactions.rename(columns = {"ID" : "TRANSACTION_ID"}).assign(ID = str(processID)).filter(["ID", "TRANSACTION_ID", "CATEGORY"])

        # Insert new data into result HANA table

        insertionResult = dataframe.create_dataframe_from_pandas(
            conn, # HANA Connection
            new_transactions, # formatted result data 
            TRANSACTION_TARGET, # result table
            self.db_key['USER'], # db schema matches db user
            upsert=True, # update if key exists, insert otherwise
            replace=True
        )

        # log if db operation was successful
        print(insertionResult.collect())
        return str(processID)

app = App()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"App started. Serving on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)