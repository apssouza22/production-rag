CHUNK_CONTEXT_PROMPT = """You are an AI assistant specializing in scientific document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.

Here is the document:

<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document context.
3. If applicable, note how this information relates to the overall theme or purpose of the document.
4. Include any key figures, dates, or percentages that provide important context.
5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

Context:"""
