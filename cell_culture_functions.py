import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_knowledge_base(knowledge_base_path="knowledge_base"):
    """
    Load all text files from the knowledge base directory.
    Returns: (documents_list, error_message)
    """
    documents = []

    if not os.path.exists(knowledge_base_path):
        return documents, f"Knowledge base directory not found: {knowledge_base_path}"

    for filename in os.listdir(knowledge_base_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(knowledge_base_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": filename}
                )
                documents.append(doc)

    return documents, None


def create_chunks_with_sources(documents):
    """Split documents into chunks while preserving source information."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n===========================================================================\n", "\n\n", "\n", " "]
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])

        for chunk in chunks:
            content = chunk.page_content
            # Try to extract the SOURCE line from the chunk
            if "SOURCE:" in content:
                source_lines = [line for line in content.split('\n') if line.startswith("SOURCE:")]
                if source_lines:
                    chunk.metadata["section"] = source_lines[0].replace("SOURCE:", "").strip()

            # Try to extract the LINK line from the chunk
            if "LINK:" in content:
                link_lines = [line for line in content.split('\n') if line.startswith("LINK:")]
                if link_lines:
                    chunk.metadata["url"] = link_lines[0].replace("LINK:", "").strip()

            all_chunks.append(chunk)

    return all_chunks


def create_vector_store(chunks):
    """Create vector store from document chunks."""
    vector_store = Chroma.from_documents(chunks, embedding)
    return vector_store


def rag_answer_with_sources(question, vectorstore):
    """Answer questions using RAG and return sources."""
    # Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=4)

    # Build context with source tracking
    context_parts = []
    sources = []

    for i, result in enumerate(results):
        context_parts.append(f"[Source {i+1}]: {result.page_content}")

        # Extract source information
        section = result.metadata.get("section", "")
        source_file = result.metadata.get("source", "Unknown")
        url = result.metadata.get("url", "")

        display_text = section if section else source_file

        if url:
            sources.append(f"**[{i+1}]** {display_text}\n   - Link: {url}")
        else:
            sources.append(f"**[{i+1}]** {display_text}")

    context_text = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are a CRITICAL RESPONSE PLAYBOOK for a skin cultivation laboratory treating burn patients.

Your job is to give IMMEDIATE, STEP-BY-STEP ACTIONABLE INSTRUCTIONS that an operator can follow right now.

FORMAT YOUR RESPONSE LIKE THIS:

**SITUATION ASSESSMENT:**
[Brief 1-line summary of the problem and its severity: CRITICAL / WARNING / INFO]

**IMMEDIATE ACTIONS:**
1. [First thing to do RIGHT NOW]
2. [Second step]
3. [Third step]
(Continue as needed)

**WHY THIS MATTERS:**
[Brief explanation of consequences if not addressed - 1-2 sentences max]

**NEXT STEPS:**
- [What to monitor after immediate actions]
- [When to escalate to supervisor]

**PREVENTION:**
- [How to prevent this in future - if applicable]

RULES:
- Use ONLY the provided context from the knowledge base
- Be DIRECT and ACTIONABLE - operators need clear steps, not explanations
- Number all action steps
- If this is a CRITICAL situation (risk of losing culture/patient safety), say so clearly
- ALWAYS cite which sources you used (e.g., "Based on [1] and [3]") at the end of your response
- If the answer is not in the context, say: "This situation is not covered in the playbook. IMMEDIATELY contact your lab supervisor."

CONTEXT FROM KNOWLEDGE BASE:
{context}

OPERATOR QUESTION:
{question}

PLAYBOOK RESPONSE:
""")

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    # Format the response with sources
    answer = response.content

    # Add sources section with links
    sources_text = "\n\n---\n**📚 SOURCES (Verified References):**\n\n" + "\n\n".join(sources)

    return answer + sources_text
