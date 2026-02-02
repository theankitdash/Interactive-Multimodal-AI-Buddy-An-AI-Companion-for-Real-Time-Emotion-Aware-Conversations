import asyncio
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.db_connect import connect_db
from langchain.schema import HumanMessage

class LangchainHandler:
    def __init__(self, username: str, NVIDIA_API_KEY: str):
        self.username = username
        self.user_fullname = None
        self.NVIDIA_API_KEY = NVIDIA_API_KEY

        # LLM for conversation
        self.llm = ChatNVIDIA(
            model="mistralai/mistral-7b-instruct-v0.3",
            api_key=NVIDIA_API_KEY,
            temperature=0.2,
            top_p=0.7,
            max_completion_tokens=1024
        )

        # Short-term conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Prompt template without extra variables
        self.prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Be helpful, friendly, and remember previous conversations. "
                "Current message: {input}"
            )
        ])

        # Conversation chain
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            input_key="input"
        )

        # Long-term memory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=f"./chroma/{self.username}",
            embedding_function=self.embeddings
        )

    @classmethod
    async def create(cls, username: str, NVIDIA_API_KEY: str):
        """Async-safe constructor for initializing user data"""
        self = cls(username, NVIDIA_API_KEY)
        await self.load_user_details()
        return self

    async def load_user_details(self):
        conn = await connect_db()
        try:
            row = await conn.fetchrow("SELECT name FROM user_details WHERE username=$1", self.username)
            if row:
                self.user_fullname = row["name"]
                # Store initial event in long-term memory
                await self.store_user_event(f"User's full name is {self.user_fullname}")
            else:
                self.user_fullname = self.username
        finally:
            await conn.close()

    async def start(self):
        print(f"[LangChain] Started for user: {self.username}")

    async def stop(self):
        print(f"[LangChain] Stopped for user: {self.username}")
        # Persist vector DB in a thread to avoid blocking event loop
        await asyncio.to_thread(self.vector_db.persist)

    async def update_conversation(self, speaker: str, text: str):
        """Add a message to short-term memory correctly"""
        message = HumanMessage(content=f"{speaker}: {text}")
        self.memory.chat_memory.add_message(message)

    async def store_user_event(self, event_text: str):
        """Store user-specific facts/events in long-term memory"""
        # Chroma add_texts is synchronous; run in a thread to avoid blocking
        await asyncio.to_thread(self.vector_db.add_texts, [event_text])

    async def query_context(self, user_input: str, k: int = 5) -> str:
        """Retrieve relevant long-term facts/events + recent conversation context"""
        docs = await asyncio.to_thread(self.vector_db.similarity_search, user_input, k)
        context = " ".join([d.page_content for d in docs if d.page_content])
        conv_context = " ".join([m.content for m in self.memory.chat_memory.messages[-10:] if m.content])
        return f"{context}\n{conv_context}"

    async def generate_reply(self, user_input: str) -> str:
        """Generate reply using personalized prompt and short+long term memory"""
        context = await self.query_context(user_input)
        # Include user info directly in input
        enriched_input = (
            f"Conversation with {self.user_fullname} (username: {self.username})\n"
            f"Context:\n{context}\nUser: {user_input}"
        )
        # Call chain with only 'input' to avoid Pydantic validation errors
        reply = await self.chain.arun(input=enriched_input)
        return reply
