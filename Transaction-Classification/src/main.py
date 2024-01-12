from hana_ml import dataframe
from flask import Flask, request, json
from flask_cors import CORS
import pandas as pd
import requests
import re
import uuid
import os

TRANSACTION_SOURCE = "TRANSACTION"
TRANSACTION_TARGET = "TRANSACTION_CATEGORY"


class GPT:
    def __init__(self):

        #self.svc_key = svc_key
        self.svc_url = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dedcee24541dbd38/chat/completions?api-version=2023-05-15"
        self.client_id = "sb-93c2a0f2-916d-4347-b7ee-ebd95e0d2ca0!b148595|aicore!b540"
        self.client_secret = "8103a851-ab3c-49ce-b0ed-dccf48051c0b$_ICYNwC_pttFOmR8xZOKiBJB-1MM0mTk_jEI7CQJu8Q="
        self._get_token()

    def _get_token(self):
        uaa_url = "https://hackxperience-analytics-insight.authentication.eu10.hana.ondemand.com/oauth/token"
        params = {"grant_type": "client_credentials" }
        resp = requests.post(
            f"https://hackxperience-analytics-insight.authentication.eu10.hana.ondemand.com/oauth/token",
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
            #response = str(response.json()['choices'][0]['message']['content'])
        return response      


class App(Flask):
    def __init__(self):
        super().__init__(__name__)
        self.add_url_rule("/v2/inference", "inference", self.inference, methods=["POST"])
        
    def inference(self):
        processID = uuid.uuid4()

        # get HANA connection
        conn = self.connectToHANA()

        categories = request.json["categories"]

        # collect transactions
        transactions = pd.DataFrame(conn.table(TRANSACTION_SOURCE, "USR_CIA6SS2GPMU6V8KX48KTLMFCS").select('ID', 'DESCRIPTION').collect())

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
            schema="USR_CIA6SS2GPMU6V8KX48KTLMFCS", # db schema matches db user
            replace=True,
			append=True
        )

        # log if db operation was successful
        print(insertionResult.collect())
        return str(processID)


    def connectToHANA(self):
        # connect with HANA Databse
        return dataframe.ConnectionContext(
            address  = "18c742bb-9f70-4a10-88f7-4fa1f5cf3ed8.hna1.prod-eu10.hanacloud.ondemand.com",
            port     = 443,
            user     = "USR_CIA6SS2GPMU6V8KX48KTLMFCS",
            password = "Xi2HZ-wVLt.TXUAawCe9pRwXUKii4nLsZ4mBtIlJrnY6m6xleE1.zHy-8VlHkAJAcXxlmuIiHgvaHdJoeHG8K9dUdsF7ELkR-NAySgOcpkMnLxgkT7Tgzv7m.ikOY03k")

app = App()
cors = CORS(app, resources={r"/*": {"origins": "*"}})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"App started. Serving on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
