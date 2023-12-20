from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type, Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from enum import Enum


class Category(Enum):
    Medicine = "Medicine"
    Chemistry = "Chemistry"
    Biology = "Biology"
    Humanities = "Humanities"
    Physics = "Physics"
    Engineering = "Engineering"
    Math = "Math"
    Ecology = "Ecology"
    Computer_Science = "Computer Science"
    Economics = "Economics"
    Geosci = "Geosci"


class SummaryArgsSchema(BaseModel):
    summary: str = Field(description="Summary of the document.")
    keywords: str = Field(description="Keywords of the document.")
    category: Category = Field(description="Category of the document.")


def summarize_doc(params):
    # Update file in here
    return params


class CustomSummaryTool(BaseTool):
    name = "summary_tool"
    description = "Summarize a document."
    args_schema: Type[SummaryArgsSchema] = SummaryArgsSchema

    def _run(
        self,
        keywords: str,
        summary: str,
        category: Category,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return summarize_doc(params={
            'keywords': keywords,
            'summary': summary,
            "category": category
        })

    async def _arun(
        self,
        keywords: str,
        summary: str,
        category: Category,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


summary_tool = CustomSummaryTool()
