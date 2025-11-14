from flask import Flask, render_template, request, session, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from src.prompt import *
import os
import urllib.parse
from datetime import datetime

# === Load Environment Variables ===
load_dotenv()

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret-key-for-development")

# === Set API Keys from Environment ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Embedding & Retrieval ===
embeddings = download_hugging_face_embeddings()
index_name = os.getenv("PINECONE_INDEX_NAME", "bookingbot4")
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# === Groq LLM Setup ===
llm = ChatGroq(
    temperature=0.4,
    max_tokens=500,
    model_name="llama-3.1-8b-instant"
)
import re  # Make sure to import this at the top

def is_valid_email(email: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$"
    return re.match(pattern, email) is not None

# === RAG Prompt Chain ===
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# === Booking Info Parser ===
class BookingInfo(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    service: str = Field(..., min_length=1)
    date: str = Field(..., min_length=1)
    time: str = Field(..., min_length=1)

parser = PydanticOutputParser(pydantic_object=BookingInfo)

booking_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that extracts structured booking info from user input.\n"
     "Always respond in this JSON format:\n"
     "{{\n"
     "  \"name\": \"<user name>\",\n"
     "  \"email\": \"<user email>\",\n"
     "  \"service\": \"<service>\",\n"
     "  \"date\": \"YYYY-MM-DD\",\n"
     "  \"time\": \"HH:MM\"\n"
     "}}\n"
     "{format_instructions}"),
    ("human", "{input}")
]).partial(format_instructions=parser.get_format_instructions())

booking_chain = booking_prompt | llm | parser

# === Helpers ===
def validate_date(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_time(time_str):
    try:
        datetime.strptime(time_str, '%H:%M')
        return True
    except ValueError:
        return False

def get_memory():
    return session.get("conversation_history", [])

def save_memory(history):
    session["conversation_history"] = history

# === ROUTES ===
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    conversation_history = get_memory()
    conversation_history.append({"role": "user", "content": msg})

    # === Check if we're in booking flow ===
    booking_keywords = ["book", "appointment", "schedule", "reserve", "meeting", "session", "therapy"]
    is_booking_flow = any(word in msg.lower() for word in booking_keywords) or session.get("booking_in_progress")

    if is_booking_flow:
        session["booking_in_progress"] = True

        # === Combine stored info with current input ===
        existing_info = {
            "name": session.get("name", ""),
            "email": session.get("email", ""),
            "service": session.get("service", ""),
            "date": session.get("date", ""),
            "time": session.get("time", "")
        }

        combined_input = " ".join([f"My {k} is {v}." for k, v in existing_info.items() if v]) + " " + msg

        try:
            booking_info = booking_chain.invoke({"input": combined_input})
            print("Parsed Booking Info:", booking_info)

            # === Check for hallucinated or placeholder values ===
            required_fields = ['name', 'email', 'service', 'date', 'time']
            bad_values = [
            "", "john doe", "your name", "name", "example", "abc@example.com",
            "johndoe@example.com","johndoe@example.com","john@example.com" ,"someone@example.com", "user", "user@gmail.com","user@example.com","unknown","unknown@gmail.com","You didn't mention your name","You didn't mention your email"
            ]

            missing_fields = []

            for field in required_fields:
                value = getattr(booking_info, field, "").strip().lower()
                if value in bad_values:
                    missing_fields.append(field)

            if "date" not in missing_fields and not validate_date(booking_info.date):
                missing_fields.append("date")
            if "time" not in missing_fields and not validate_time(booking_info.time):
                missing_fields.append("time")
            if "email" not in missing_fields and not is_valid_email(booking_info.email):
                missing_fields.append("email")

            # === Store valid partial info ===
            for field in required_fields:
                value = getattr(booking_info, field, "").strip()
                if value.lower() not in bad_values:
                    session[field] = value

            if missing_fields:
                reply = f"Sure! To proceed, please provide your {', '.join(missing_fields)}."
                conversation_history.append({"role": "assistant", "content": reply})
                save_memory(conversation_history)
                return reply

            # === All fields are valid, proceed ===
            session["booking_in_progress"] = False

            base_url = urllib.parse.urljoin(request.host_url, "book")
            query_params = {
                "name": session['name'],
                "email": session['email'],
                "service": session['service'],
                "date": session['date'],
                "time": session['time']
            }
            booking_url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
            reply = f"You can complete your booking here: <a href='{booking_url}' target='_blank'>{booking_url}</a>"

            conversation_history.append({"role": "assistant", "content": reply})
            save_memory(conversation_history)
            return reply

        except ValidationError as ve:
            error_fields = [e['loc'][0] for e in ve.errors()]
            reply = f"Missing or invalid: {', '.join(error_fields)}. Please provide them."
            conversation_history.append({"role": "assistant", "content": reply})
            save_memory(conversation_history)
            return reply

        except Exception as e:
            print("Booking extraction failed:", e)
            reply = "Sorry, I couldn't understand all the booking details. Could you please provide your name, email, preferred date, and time?"
            conversation_history.append({"role": "assistant", "content": reply})
            save_memory(conversation_history)
            return reply

    # === General Q&A ===
    context_messages = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
    )
    full_input = f"{context_messages}\nuser: {msg}"

    response = rag_chain.invoke({"input": full_input})
    answer = response["answer"]
    print("RAG Response:", answer)

    conversation_history.append({"role": "assistant", "content": answer})
    save_memory(conversation_history)
    return answer


@app.route("/book")
def book():
    return render_template(
        "booking_form.html",
        name=request.args.get("name", ""),
        email=request.args.get("email", ""),
        service=request.args.get("service", ""),
        date=request.args.get("date", ""),
        time=request.args.get("time", "")
    )

@app.route("/logout")
def logout():
    session.clear()
    return "Session cleared!"

@app.route("/session")
def session_data():
    return jsonify(dict(session))

# === Run Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
