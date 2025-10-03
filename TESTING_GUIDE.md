# LangChain Refactoring - Testing Guide

## Installation

To test the refactored codebase, first install the dependencies:

```bash
cd /home/runner/work/text2sql-project/text2sql-project
pip install -r api/requirements.txt
```

## Quick Validation

Run the validation test script:

```bash
python test_langchain_refactoring.py
```

This will verify:
1. All LangChain imports work correctly
2. PromptTemplate functionality
3. QueryOrchestrator syntax is valid
4. All expected methods are defined
5. LangChain components are properly integrated
6. Backward compatibility is maintained
7. No breaking changes were introduced

## Manual Testing

### 1. Start the Services

Ensure all required services are running:

```bash
# PostgreSQL
docker-compose up -d postgres-db

# ChromaDB
docker-compose up -d chromadb

# Ollama
docker-compose up -d ollama
```

### 2. Start the API

```bash
cd api
uvicorn app.main:app --reload
```

### 3. Test Endpoints

#### Test Question Answering
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quel est le PIB nominal de l'\''UEMOA en 2022?"
  }'
```

Expected: SQL query generated, executed, and natural language response returned.

#### Test Query Indexing
```bash
curl -X POST http://localhost:8000/api/index-queries \
  -H "Content-Type: application/json"
```

Expected: Reference queries indexed successfully.

#### Test Model Pulling
```bash
curl -X POST http://localhost:8000/api/pull-model \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2"
  }'
```

Expected: Model downloaded successfully.

### 4. Verify Functionality

Check that:
- [ ] SQL queries are generated correctly
- [ ] Natural language responses maintain quality
- [ ] Security validations block harmful content
- [ ] Concurrency limits are respected
- [ ] Error handling works as before
- [ ] Performance is comparable

## Component-Specific Tests

### Test Embeddings
```python
from app.services.query_orchestrator import QueryOrchestrator

orchestrator = QueryOrchestrator()
queries = ["SELECT * FROM table WHERE date = '2022-01-01'"]

# This should use HuggingFaceEmbeddings.embed_documents
embeddings = orchestrator.embedding_model.embed_documents(queries)
print(f"Embeddings shape: {len(embeddings)}, {len(embeddings[0])}")
```

### Test Vector Store
```python
from app.services.query_orchestrator import QueryOrchestrator

orchestrator = QueryOrchestrator()

# This should use Chroma.similarity_search
similar_docs = orchestrator.vector_store.similarity_search(
    "Quel est le PIB en 2022?", 
    k=3
)
print(f"Found {len(similar_docs)} similar queries")
for doc in similar_docs:
    print(f"- {doc.page_content}")
```

### Test ChatOllama
```python
import asyncio
from app.services.query_orchestrator import QueryOrchestrator

async def test_llm():
    orchestrator = QueryOrchestrator()
    
    # This should use ChatOllama.ainvoke
    response = await orchestrator.llm.ainvoke("What is 2+2?")
    print(f"Response: {response.content}")

asyncio.run(test_llm())
```

### Test PromptTemplate
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["schema", "question"],
    template="Schema: {schema}\nQuestion: {question}\nSQL:"
)

prompt = template.format(
    schema="CREATE TABLE users (id INT, name TEXT)",
    question="Get all users"
)
print(prompt)
```

## Validation Checklist

### Functionality
- [ ] All API endpoints return expected responses
- [ ] SQL generation works correctly
- [ ] Natural language responses maintain quality
- [ ] Forecast narrative generation works
- [ ] Inflation interpretation works
- [ ] Model pulling succeeds

### Security
- [ ] Harmful content is blocked
- [ ] SQL injection attempts are prevented
- [ ] Only SELECT queries are allowed
- [ ] Read-only database access enforced

### Performance
- [ ] Response times are comparable to original
- [ ] Concurrency limits are respected
- [ ] Memory usage is reasonable
- [ ] No resource leaks

### Compatibility
- [ ] All existing environment variables work
- [ ] No configuration changes needed
- [ ] All request/response formats unchanged
- [ ] Error messages are consistent

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError` for LangChain packages:
```bash
pip install langchain>=0.3.27 langchain-community>=0.3.27 \
  langchain-chroma>=0.1.4 langchain-ollama>=0.2.2
```

### Ollama Connection Issues
If ChatOllama can't connect to Ollama:
1. Check `OLLAMA_HOST` and `OLLAMA_PORT` environment variables
2. Ensure Ollama is running: `docker-compose ps ollama`
3. Test connection: `curl http://ollama:11434/api/version`

### ChromaDB Issues
If vector store operations fail:
1. Check `CHROMA_HOST` and `CHROMA_PORT` environment variables
2. Ensure ChromaDB is running: `docker-compose ps chromadb`
3. Re-index queries: `curl -X POST http://localhost:8000/api/index-queries`

### Database Issues
If SQL operations fail:
1. Check `DATABASE_URL` and `ADMIN_DATABASE_URL`
2. Ensure PostgreSQL is running: `docker-compose ps postgres-db`
3. Verify database credentials

## Rollback Procedure

If issues are found and you need to rollback:

1. **Revert the code**:
   ```bash
   git revert <commit-hash>
   ```

2. **Downgrade dependencies**:
   ```bash
   pip install ollama sentence-transformers chromadb sqlalchemy
   pip uninstall -y langchain langchain-community langchain-chroma langchain-ollama
   ```

3. **Restart services**:
   ```bash
   docker-compose restart api
   ```

The original implementation used only direct library calls, so no data migration is needed for rollback.

## Success Criteria

The refactoring is successful when:

✅ All 7 validation tests pass
✅ All API endpoints work correctly
✅ Security validations function properly
✅ Performance is comparable to original
✅ No functionality regressions
✅ All existing configurations work

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChatOllama Reference](https://python.langchain.com/docs/integrations/chat/ollama)
- [Chroma Vector Store](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [HuggingFace Embeddings](https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub)
- [PromptTemplate Guide](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
