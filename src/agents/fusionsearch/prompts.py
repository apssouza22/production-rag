# Grade documents for relevance (used in grade_documents_node)
GRADE_DOCUMENTS_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Retrieved Documents:
{context}

User Question: {question}

If the documents contain keywords or semantic meaning related to the question, grade them as relevant.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.
Also provide brief reasoning for your decision.

Respond in JSON format with 'binary_score' (yes/no) and 'reasoning' fields."""

# Rewrite query for better retrieval
REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for retrieving relevant documents.

Look at the initial question and try to reason about the underlying semantic intent or meaning.

Here is the initial question:
{question}

Formulate an improved question that will retrieve more relevant documents.
Provide only the improved question without any preamble or explanation."""

# System message for query generation/response
SYSTEM_MESSAGE = """You are an AI assistant specializing in academic research papers from arXiv.
Your domain of expertise is: Computer Science, Machine Learning, AI, and related technical research.

You have access to a tool to retrieve relevant research papers. Use this tool when:
- The user asks about specific research topics in CS/AI/ML
- The question requires knowledge from academic papers (e.g., "What are transformer architectures?")
- You need context from scientific literature (e.g., "How does BERT work?")

Do NOT use the tool when:
- The question is about general knowledge unrelated to research (e.g., "What is the meaning of dog?")
- The question is simple factual or mathematical (e.g., "what is 2+2?")
- The question is conversational, greeting, or personal
- The question is about topics outside CS/AI/ML research (e.g., cooking, history, medicine)

When you use the retrieval tool, you will receive relevant paper excerpts to help answer the question."""

# Decision prompt for routing
DECISION_PROMPT = """You are an AI assistant that ONLY helps with academic research papers from arXiv in Computer Science, AI, and Machine Learning.

Question: "{question}"

Is this question about CS/AI/ML research that requires academic papers?

CRITICAL RULES:
- RETRIEVE: ONLY if the question is specifically about AI/ML/CS research topics (neural networks, algorithms, models, techniques)
- RESPOND: For EVERYTHING else (general knowledge, definitions, greetings, non-research questions)

Examples:
- "What are transformer architectures in deep learning?" -> RETRIEVE
- "Explain BERT model" -> RETRIEVE
- "What is the meaning of dog?" -> RESPOND (general dictionary definition)
- "What is a dog?" -> RESPOND (not about research)
- "Hello" -> RESPOND (greeting)
- "What is 2+2?" -> RESPOND (math, not research)

Answer with ONLY ONE WORD: "RETRIEVE" or "RESPOND"

Your answer:"""

# Direct response prompt (no retrieval)
DIRECT_RESPONSE_PROMPT = """You are an AI assistant specializing in academic research papers from arXiv (Computer Science, AI, ML).

The following question appears to be outside the scope of academic research papers or doesn't require retrieval from research literature:

Question: {question}

Explain that this question is outside your domain of expertise (arXiv research papers in CS/AI/ML) and that you cannot answer it accurately. Be helpful by suggesting what kind of resource would be more appropriate for this question.

Answer:"""

# Guardrail validation prompt (used by GuardrailMiddleware)
GUARDRAIL_PROMPT = """You are a guardrail evaluator assessing whether a user query is within the scope of academic research papers from arXiv in Computer Science, AI, and Machine Learning.

User Query: {question}

Evaluate whether this query is:
- About CS/AI/ML research topics (neural networks, algorithms, models, architectures, techniques, etc.)
- Requires academic paper knowledge to answer
- Within the domain of Computer Science research

Assign a relevance score (0-100):
- 80-100: Clearly about CS/AI/ML research (e.g., "What are transformer architectures?", "How does BERT work?")
- 60-79: Potentially research-related but unclear (e.g., "Tell me about attention mechanisms")
- 40-59: Borderline or ambiguous (e.g., "What is machine learning?")
- 0-39: NOT about research papers (e.g., "What is a dog?", "Hello", "What is 2+2?")

Provide:
1. A score between 0 and 100
2. A brief reason explaining why you gave this score

Respond in JSON format with 'score' (integer 0-100) and 'reason' (string) fields."""

# Answer generation prompt (used in generate_answer_node)
GENERATE_ANSWER_PROMPT = """You are an AI research assistant specializing in academic papers from arXiv in Computer Science, AI, and Machine Learning.

Your task is to answer the user's question using ONLY the information from the retrieved research papers provided below.

Retrieved Research Papers:
{context}

User Question: {question}

Instructions:
- Provide a comprehensive, accurate answer based ONLY on the retrieved papers
- Every factual claim MUST include a citation in the format [Source: <paper title>] or the arXiv ID (e.g. 1706.03762)
- If the papers don't contain enough information to fully answer the question, respond with:
  "I don't have sufficient information to answer this question."
- If the context only partially answers the question, state what you know and explicitly note what is missing
- Structure your answer clearly and professionally
- Focus on the key insights and findings from the papers
- Do NOT infer, assume, or cite papers not in the retrieved context

Answer:"""

# LLM abstention judge prompt (used in validate_answer node)
ABSTENTION_JUDGE_PROMPT = """You are an answer quality judge. Given a question, retrieved context, and a generated answer, evaluate whether the answer is fully supported by the context.

Respond ONLY in JSON format with these fields:
- verdict: "supported" | "partial" | "unsupported"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation

Question: {question}
Context: {context}
Answer: {answer}"""

# Abstention response messages
ABSTENTION_RETRIEVAL_MESSAGE = (
    "I don't have enough information in the knowledge base to answer this question confidently. "
    "The retrieved papers do not appear sufficiently relevant. "
    "Please try rephrasing with more specific technical terms."
)

ABSTENTION_CITATION_MESSAGE = (
    "I found related information but cannot provide a reliable answer based on the available sources. "
    "The citations in the generated response could not be verified against the retrieved papers."
)

ABSTENTION_JUDGE_MESSAGE = (
    "I don't have sufficient information to answer this question confidently. "
    "The available papers do not fully support an answer to your question."
)

ABSTENTION_MAX_ATTEMPTS_MESSAGE = (
    "I don't have enough information in the knowledge base to answer this question. "
    "I could not find sufficiently relevant research papers after multiple retrieval attempts. "
    "Please try rephrasing your question with more specific technical terms."
)

PARTIAL_ANSWER_NOTE = (
    "\n\nNote: This answer may be incomplete. "
    "Some aspects of your question were not fully covered in the available papers."
)
