from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class ChatMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def get_conversation_chain(self, retriever, llm):
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory
        )