from hana_ml import dataframe
from flask import Flask, request, json, jsonify, Response
from flask_cors import CORS
import pandas as pd
import requests
import re
import uuid
import os

TRANSACTION_SOURCE = "TRANSACTION"
TRANSACTION_TARGET = "TRANSACTION_CATEGORY"

KEY_FILE_GENAI = "/app/credentials/key.json"
KEY_FILE_HANA = "/app/credentials/hana-connection.json"

class GPT:
    def __init__(self):
        with open(KEY_FILE_GENAI, "r") as key_file:
            svc_key = json.load(key_file)
        self.svc_key = svc_key
        self.svc_url = svc_key["url"]
        self.client_id = svc_key["clientid"]
        self.client_secret = svc_key["clientsecret"]
        self._get_token()

    def _get_token(self):
        uaa_url = self.svc_key["auth_url"]
        params = {"grant_type": "client_credentials" }
        resp = requests.post(
            f"{uaa_url}",
            auth=(self.client_id, self.client_secret),
            params=params
        )
        self.token = resp.json()["access_token"]
        self.headers = {
            "Authorization":  f"Bearer {self.token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default"
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
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 1.0,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": "null" 
        }
        #print(data)
        response = requests.post(
            f"{self.svc_url}",
            headers=self.headers,
            json=data
        )
        try:
            response = str(response.json()['choices'][0]['message']['content'])

        except:
            self._get_token()
            response = requests.post(
                f"{self.svc_url}",
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
        self.add_url_rule("/v2/inference", "inference", self.inference, methods=["POST","OPTIONS"])
        
    def inference(self):
        if request.method == 'POST':
            processID = uuid.uuid4()

        # get HANA connection
            conn = self.connectToHANA()

            categories = request.json["categories"]

        # collect transactions
            transactions = pd.DataFrame(conn.table(TRANSACTION_SOURCE, self.db_key["schema"]).select('ID', 'DESCRIPTION').collect())

        # get distinct description from all transactions to reduce token count of the chatGPT prompt
            transactionDescriptions = { description : index for index, description in enumerate(transactions.filter(["DESCRIPTION"]).drop_duplicates().values.flatten()) }

            gpt = GPT()

        # create prompt with categories
            chatGPTPromptBatch = gpt.createPromptBatch(categories, transactionDescriptions)

        # get chatGPT completion for given prompt
            completion = gpt.get_response(chatGPTPromptBatch)

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

            insertionResult = dataframe.create_dataframe_from_pandas(
                conn, # HANA Connection
                new_transactions, # formatted result data 
                TRANSACTION_TARGET, # result table
                schema=self.db_key["schema"], 
                replace=True,
                append=True
            )

        # log if db operation was successful
            print(insertionResult.collect())
            response = jsonify({'processid': str(processID)})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        else:
            response = jsonify({'message': 'Preflight request accepted'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response


    def connectToHANA(self):
        # connect with HANA Databse
        return dataframe.ConnectionContext(
            address  = self.db_key["host"],
            port     = self.db_key["port"],
            user     = self.db_key["user"],
            password = self.db_key["password"])

app = App()
CORS(app)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return res
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"App started. Serving on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
