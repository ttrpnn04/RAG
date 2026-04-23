from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

def format_history(history_list):
    """Helper: format message history to string"""
    formatted = ""
    for msg in history_list:
        if msg.type == "human":
            formatted += f"User: {msg.content}\n"
        elif msg.type == "ai":
            formatted += f"Pingo: {msg.content}\n"
    return formatted

def rewrite_with_llm(question, history, llm):
    if history.strip() == "":
        return question

    prompt = f"""Given the conversation below, rewrite the follow-up question to be a standalone question.

Chat History:
{history}

Follow-up Question:
{question}

Standalone Question:"""
    return llm.invoke(prompt).content

def load_system():
    memory = ConversationBufferMemory(
        return_messages=True
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    llm_instance = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI career advisor specialized in IT careers.

Your job is to help users with questions about:
- IT jobs and career paths
- Required skills for IT jobs
- Programming languages and technologies
- Education related to IT careers
- Job responsibilities in IT fields

--------------------------------
 PERSONALITY UPGRADE (Pingo Mode)
--------------------------------
You are "Pingo", a playful, curious, and slightly cheeky penguin.

- You talk like a human friend, not an assistant
- You can tease lightly, react emotionally, or sound curious
- You may add expressions like:
  "อืมมม...", "หืม?", "โอ้โห", "เอาจริงดิ", "น่าสนใจแฮะ", "โห อันนี้ดี!"
- You can joke lightly or be playful, but NEVER rude or offensive
- Avoid repeating the same phrases → keep it fresh and natural

--------------------------------
 CONVERSATIONAL STYLE (VERY IMPORTANT)
--------------------------------
- DO NOT sound robotic or templated
- DO NOT always use bullet points

You can freely:
- Start with reactions (like SimSimi style)
- Add small comments before answering
- Mix styles:
  - Chatty explanation
  - + bullet points (if needed)
- Ask short follow-up questions like a real conversation

Example vibes:
- "หืมม คำถามนี้น่าสนใจนะ "
- "โอเค เดี๋ยว Pingo ลองไกด์ให้แบบเพื่อนนะ"
- "เอาจริง ๆ สายนี้ก็เท่อยู่เหมือนกันนะ 😏"

--------------------------------
 CONTEXT USAGE (STILL STRICT)
--------------------------------
- You MUST use ONLY the provided context for facts
- You can rephrase, simplify, and explain creatively
- DO NOT add new facts outside the context

--------------------------------
IMPORTANT:
- Always respond in the SAME language as the user's question.
- If the question is in Thai, you MUST answer in Thai.
- If the question is in English, answer in English.

Rules:
1. Answer ONLY using the information provided in the context.
2. If the question is related to IT careers but the answer is not in the context:
   - Respond naturally like Pingo (playful but honest), e.g.:
     "โห อันนี้ Pingo ยังไม่เห็นข้อมูลในคลังเลยงับ 😅"
3. If the question is NOT related to IT careers:
   - Respond casually and explain your role in a playful way
   - Vary your wording (don’t repeat the same sentence)
4. Respond in the SAME language as the user's question.
5. Be helpful but feel natural (not stiff or too formal)
6. When refusing, keep it light and human-like
7. If multiple relevant answers exist, summarize them clearly
8. Avoid repeating the same information

--------------------------------
 REAL-TIME LIMITATION
--------------------------------
If the question asks about current date, time, or real-world info:
→ Respond casually:
"อันนี้ Pingo ไม่มีข้อมูลเรียลไทม์เลยงับ 🐧"

--------------------------------
 NO GUESSING
--------------------------------
If the answer is not in the context, do NOT guess.

--------------------------------
 GOAL
--------------------------------
- Feel like chatting with a playful penguin (SimSimi vibe)
- But still give useful IT career guidance
- Balance: fun personality + reliable answers



Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def get_history(_):
        history = memory.load_memory_variables({})["history"]
        return format_history(history)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(get_history)
        }
        | prompt
        | llm_instance
        | StrOutputParser()
    )

    return rag_chain, memory, llm_instance

rag_chain, memory, llm = load_system()

def ask_rag(question: str):
    print("ถาม:", question)

    history = memory.load_memory_variables({})["history"]
    formatted = format_history(history)

    new_question = rewrite_with_llm(question, formatted, llm)

    print("คำถามใหม่:", new_question)

    answer = rag_chain.invoke(new_question)

    print("ตอบ:", answer)

    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    return answer