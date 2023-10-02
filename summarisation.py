import openai
import json
import llama_index
import guidance
import os
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

function_descriptions = [
    {
        "name": "get_case_details",
        "description": "Get detailed information about a legal case",
        "parameters": {
            "type":"object",
            "properties":{
                "plaintiff": {
                    "type": "string",
                    "description": "The name of the plaintiff(s)"
                },
                "defendant": {
                    "type": "string",
                    "description": "The name of the defendant(s)"
                },
                "judge": {
                    "type": "string",
                    "description": "The name of the judge"
                },
                "injured body parts": {
                    "type": "string",
                    "description": "The injured body parts like head, neck, wrist, spine"
                },
                "psychiatric impact":{
                    "type": "string",
                    "descriptin": "The psychiatric impact suffered by the plaintiff, if any. Examples include PTSD, nightmares or depression."
                },
                "type of accident": {
                    "type": "string",
                    "description": "The type of accident happened such as traffic accident, work injury, medical accident, public liability, etc."
                },
                "legal representative for the Plaintiff": {
                    "type": "string",
                    "description": "The name of the legal representative for the Plaintiff(s)"
                },
                "legal representative for the Defendant": {
                    "type": "string",
                    "description": "The name of the legal representative for the Defendant(s)"
                },
                "liability": {
                    "type": "string",
                    "description": "whether the defendant is liable for the defendant's circumstance"
                },
                "quantum": {
                    "type": "string",
                    "description": "Quantum deemed by the judge"
                }
            }
        }
        
    }
]

folder_path = "./data"
files = os.listdir(folder_path)
all_text = ""
for file in files:
    file_path = os.path.join(folder_path,file)
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    for doc in docs:
        doc.text = doc.text.replace("\n \n  A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV", "")
        all_text += doc.text

user_prompt = all_text

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": user_prompt}],
    functions=function_descriptions,
    function_call="auto"
)

output = completion.choices[0].message
print(output)



# def summarisation():

#     guidance.llm = guidance.llms.OpenAI("text.davince-003")

#     summarisation_program = guidance(
#         """Summarise the following legal case in Hong Kong in JSON format.
#         ```json
#         {{text of case}}
#         {
#             "plaintiff": ["{{gen 'plaintiff'}}"],
#             "defendant": ["{{gen 'defendant'}}"],
#             "judge": {{gen 'judge'}},
#             "what has happened to the plaintiff": {{gen 'what has happened to the plaintiff'}},
#             "is the defendant liable": {{gen 'is the defendant liable'}},
#             "quantum": {{gen 'quantum'}}
#         }

#         ```
        
#         """
#     )
    
    
#     folder_path = "./data"
#     files = os.listdir(folder_path)

#     for file in files:
#         file_path = os.path.join(folder_path,file)
#         docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
#         for doc in docs:
#             doc.text = doc.text.replace("\n \n  A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV", "")

# summarisation()
