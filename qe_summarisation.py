import os
import openai
from dotenv import load_dotenv
from pathlib import Path
from llama_index import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, ServiceContext, StorageContext, GPTVectorStoreIndex, Document
from llama_index.node_parser import SimpleNodeParser
from llama_index import get_response_synthesizer
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
import nltk
import pinecone

# openai.log = "debug"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_folder_of_summaries():

    files = os.listdir("./data")

    for file in files:
        summaries_path = Path("./summaries")
        if not summaries_path.exists():
            summaries_path.mkdir()
        case_number = os.path.splitext(file)[0]
        summary_path = Path(summaries_path)/f"{case_number}.txt"
        if not summary_path.exists():
            print("summary for ", case_number," does not exist yet. Creating summary.")        
            summary = ''
            docs = SimpleDirectoryReader(input_files=[os.path.join("./data",file)]).load_data()
            for doc in docs:
                doc.text = doc.text.replace("A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV", "")
            
            llm = OpenAI(temperature=0, model="gpt-4")
            service_context = ServiceContext.from_defaults(llm=llm)

            vector_index = VectorStoreIndex.from_documents(docs)
            service_context = ServiceContext.from_defaults(llm=llm)
            query_engine = vector_index.as_query_engine(similarity_top_k=3)    

            date_of_hearing = query_engine.query("What is the date of Hearing? The Date of Hearing is usually located before the Date of Judgement")
            date_of_judgement = query_engine.query("What is the date of Judgment? The Date of Judgment is usually located before the Date of Hearing")
            judge = query_engine.query("What is the name of the judge. It is usually at the beginning of the case.")
            background_of_plaintiffs = query_engine.query("There might be one more more than one plaintiffs. Identify all of them and list their gender, age during the accident, age during the trial or assessment of damages and occupation before the accident.")
            background_facts_of_case = query_engine.query("Identify the facts of the case about what has happened and answer in bullet point forms. If there are different versions of facts from the plaintiff and the defendant, identify both the plaintiff and the defendant's versions and list out their differences as detailed as possible.")
            legal_issues = query_engine.query("Identify what legal issues and the judge and the lawyers discussed and the rationale behind those arguments. Based on each legal issue identified, what are the respective ruling? Answer in bullet point forms.")
            rulings = query_engine.query("What are the rulings made by the judge?")
            injury = query_engine.query("Identify the injuries suffered by the plaintiffs, both physical injuries or pschological ones. Examples include persistent low back pain ,stiff back, tenderness at the lumbosarcal junction and both sacro-iliac joints, nightmares, depression, PTSD, etc.")
            loss_of_amenities = query_engine.query("If the plaintiff has any loss of amenities, such as loss of the ability to engage in social activities, sports and hobbies, please list them out for each plaintiff.")
            medical_evidence = query_engine.query("List out the medical evidence for each plaintiff.")
            earnings_of_plaintiff = query_engine.query("List out the income or earning abilities of each plaintiff.")
            quantum = query_engine.query("List out the amount of damages or compensation to be awarded to each plaintiff. Please list out the respective heads of damages and their respective amoutns. Also provide the total amount rewarded to each of the Plaintiff.")
            issues_related_to_quantum = query_engine.query("List out the court’s rulings and reasons and basis for each of the heads of damages.")
            costs = query_engine.query("Summaries the court's ruling on costs, identify who should bear the costs and which party was rewarded costs.")
            legal_represenation = query_engine.query("List out the legal represenative or the lawyer(s) for the plaintiff(s) and the defendant(s) respectively.")
            plaintff_and_defendant = query_engine.query("This case is between who (plaintiff) and who (defendant)? Please list out the names of all plaintiffs and defendants.")
            accident_circumstances = query_engine.query("Please describe the accident involved, which is usually mentioned in the introduction. How did the accident happened? What occured before the moment of accident? What exactly cause the injury? For example, the plaintiff is hit by a vehicle and that caused the accident. Another example is the plaintiff sprained his back while lifting some heavy goods.")
            
            # print("Date of Hearing: ",date_of_hearing)
            # print("Date of Judgement: ",date_of_judgement)
            # print("judge: ",judge)
            # print("background_of_plaintiffs: ", background_of_plaintiffs)
            # print("background_facts_of_case: ", background_facts_of_case)
            # print(background_facts_of_case.source_nodes)
            # print("legal_issues: ", legal_issues)
            # print("ruilings: ", ruilings)
            # print("injury:", injury)
            # print("loss_of_amenities: ",loss_of_amenities)
            # print("medical_evidence: ", medical_evidence)
            # print("earnings_of_plaintiff: ",earnings_of_plaintiff)
            # print("quantum:", quantum)
            # print("issues_related_to_quantum: ",issues_related_to_quantum)
            # print("costs: ",costs)
            # print("legal_representation: ",legal_represenation)
            # print("hahahahah")
            # print("plaintff_and_defendant: ", plaintff_and_defendant)
            # print("accident_circumstances: ", accident_circumstances.response)
            # print(accident_circumstances.source_nodes)

            summary = (
                "Date of Hearing: ", date_of_hearing.response,
                "\n\nDate of Judgement: ", date_of_judgement.response,
                "\n\nPlaintiff and Defendant: ", plaintff_and_defendant.response,
                "\n\nJudge: ", judge.response,
                "\n\nBackground of Plaintiffs: ", background_of_plaintiffs.response,
                "\n\nBackground facts of case: ", background_facts_of_case.response,
                "\n\nLegal Issues: ", legal_issues.response,
                "\n\nRulings: ", rulings.response,
                "\n\ninjury:", injury.response,
                "\n\nLoss of amenities: ",loss_of_amenities.response,
                "\n\nMedical evidence: ", medical_evidence.response,
                "\n\nEarnings of plaintiff: ",earnings_of_plaintiff.response,
                "\n\nQuantum:", quantum.response,
                "\n\nIssues related to quantum: ",issues_related_to_quantum.response,
                "\n\nCosts: ",costs.response,
                "\n\nLegal representation: ",legal_represenation.response,
                "\n\nAccident circumstances: ", accident_circumstances.response
                )
            
            with open(summaries_path/f"{case_number}.txt", "w") as f:
                f.write("\n".join(summary))
                print("hi. Summary for ",case_number," is done.")

        
        else:
            print("summary for ", case_number, " already exists. Skipping to next case.")
            continue


if __name__ == "__main__":
    create_folder_of_summaries()








    
    
    
  


