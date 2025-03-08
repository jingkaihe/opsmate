import pytest

from opsmate.tools.knowledge_retrieval import KnowledgeRetrieval
from opsmate.tests.base import BaseTestCase


class TestKnowledgeRetrieval(BaseTestCase):
    @pytest.mark.asyncio
    async def test_knowledge_retrieval(self):
        tool = KnowledgeRetrieval(
            query="What is the meaning of life?",
        )
        assert tool.output is None

        aconn = await tool.aconn()
        assert aconn is not None

        result = await tool.run(context={"with_reranking": False})
        assert result is not None
        assert result == tool.output

        assert tool.markdown().startswith("\n## Knowledge")
