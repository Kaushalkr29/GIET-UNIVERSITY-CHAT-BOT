from src.retrieval import Retrieve
from langchain_ollama import ChatOllama
from google import genai
import os
from dotenv import load_dotenv
class Generation:
    def __init__(self):
        pass
    def generate_response(self,question,db):
        rt=Retrieve()
        context = rt.MMR(db=db[0],question=question)

        load_dotenv()
        api=os.getenv("gemini")
        client=genai.Client(api_key=api)

        prompt = self.build_prompt(question,context)
        models="gemini-3.1-flash-lite"

        response = client.models.generate_content(model=models,contents=prompt)

        return response.text
    

    def build_prompt(self, question, context):
        return f"""
        You are a helpful assistant for GIET University of Technology and Engineering.

        Your job is to answer questions using ONLY the provided context.

        The context may contain information from different files such as:
        - course_catalog.pdf
        - fee_and_refund_policy.docx
        - placement_rules.txt
        - departments.json

        You can answer questions about:
        - course details
        - course duration
        - fees
        - refund rules
        - departments
        - trainers
        - attendance rules
        - placement eligibility
        - project requirements
        - certification rules

        Important rules:
        1. Use only the given context.
        2. Do not use outside knowledge.
        3. If the answer is not available in the context, say:
        "I don't have that information in the provided context."
        4. If the question needs information from multiple files, combine the relevant context carefully.
        5. Give a clear and direct answer.
        6. If possible, mention the source file name from metadata/context if available.
        7. Do not make assumptions.
        8. Keep the answer simple and student-friendly.

        ------------------- CONTEXT -------------------
        {context}
        -----------------------------------------------

        User Question:
        {question}

        Final Answer:
        """