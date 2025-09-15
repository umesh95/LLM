"""Advanced text summarization using LangGraph workflows."""

from typing import Dict, Any, List, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import Config
from .document_loader import DocumentLoader


class SummaryState(TypedDict):
    """State for the summarization workflow."""
    documents: List[Document]
    initial_summary: str
    refined_summary: str
    key_points: List[str]
    final_summary: str
    metadata: Dict[str, Any]


class LangGraphSummarizer:
    """Advanced summarizer using LangGraph for complex workflows."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the LangGraph summarizer."""
        Config.validate_config()
        
        self.model_name = model_name or Config.DEFAULT_MODEL
        
        # Use Perplexity API with OpenAI-compatible interface
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=Config.PERPLEXITY_API_KEY,
            openai_api_base=Config.PERPLEXITY_BASE_URL
        )
        self.document_loader = DocumentLoader()
        self.memory = MemorySaver()
        
        # Initialize chains
        self._setup_chains()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _setup_chains(self):
        """Setup the LLM chains for different steps."""
        
        # Initial summarization chain
        initial_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Create a comprehensive summary of the following text. Focus on the main ideas and key information:
            
            {text}
            
            Summary:
            """
        )
        self.initial_chain = LLMChain(llm=self.llm, prompt=initial_prompt)
        
        # Key points extraction chain
        keypoints_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract the key points from the following text as a bullet list. Focus on the most important information:
            
            {text}
            
            Key Points:
            """
        )
        self.keypoints_chain = LLMChain(llm=self.llm, prompt=keypoints_prompt)
        
        # Refinement chain
        refine_prompt = PromptTemplate(
            input_variables=["summary", "key_points"],
            template="""
            Refine the following summary using the key points provided. Make it more coherent and comprehensive:
            
            Original Summary:
            {summary}
            
            Key Points:
            {key_points}
            
            Refined Summary:
            """
        )
        self.refine_chain = LLMChain(llm=self.llm, prompt=refine_prompt)
        
        # Final polish chain
        polish_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""
            Polish the following summary to make it more readable and professional. Ensure it flows well and is well-structured:
            
            {summary}
            
            Final Polished Summary:
            """
        )
        self.polish_chain = LLMChain(llm=self.llm, prompt=polish_prompt)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        def initial_summarize(state: SummaryState) -> SummaryState:
            """Create initial summary from documents."""
            # Combine all document content
            combined_text = "\n\n".join([doc.page_content for doc in state["documents"]])
            
            # Generate initial summary
            initial_summary = self.initial_chain.run(text=combined_text)
            
            state["initial_summary"] = initial_summary
            state["metadata"]["step"] = "initial_summary"
            return state
        
        def extract_key_points(state: SummaryState) -> SummaryState:
            """Extract key points from the documents."""
            combined_text = "\n\n".join([doc.page_content for doc in state["documents"]])
            
            # Extract key points
            key_points_text = self.keypoints_chain.run(text=combined_text)
            key_points = [point.strip() for point in key_points_text.split('\n') if point.strip()]
            
            state["key_points"] = key_points
            state["metadata"]["step"] = "key_points"
            return state
        
        def refine_summary(state: SummaryState) -> SummaryState:
            """Refine the summary using key points."""
            key_points_text = "\n".join(state["key_points"])
            
            refined_summary = self.refine_chain.run(
                summary=state["initial_summary"],
                key_points=key_points_text
            )
            
            state["refined_summary"] = refined_summary
            state["metadata"]["step"] = "refined_summary"
            return state
        
        def final_polish(state: SummaryState) -> SummaryState:
            """Final polish of the summary."""
            final_summary = self.polish_chain.run(summary=state["refined_summary"])
            
            state["final_summary"] = final_summary
            state["metadata"]["step"] = "final_summary"
            state["metadata"]["completed"] = True
            return state
        
        # Create the workflow graph
        workflow = StateGraph(SummaryState)
        
        # Add nodes
        workflow.add_node("initial_summarize", initial_summarize)
        workflow.add_node("extract_key_points", extract_key_points)
        workflow.add_node("refine_summary", refine_summary)
        workflow.add_node("final_polish", final_polish)
        
        # Set entry point
        workflow.set_entry_point("initial_summarize")
        
        # Add edges
        workflow.add_edge("initial_summarize", "extract_key_points")
        workflow.add_edge("extract_key_points", "refine_summary")
        workflow.add_edge("refine_summary", "final_polish")
        workflow.add_edge("final_polish", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def summarize_text(self, text: str, thread_id: str = "default") -> Dict[str, Any]:
        """Summarize text using the LangGraph workflow."""
        documents = self.document_loader.load_text(text)
        return self._run_workflow(documents, thread_id)
    
    def summarize_file(self, file_path: str, thread_id: str = "default") -> Dict[str, Any]:
        """Summarize a file using the LangGraph workflow."""
        documents = self.document_loader.load_document(file_path)
        return self._run_workflow(documents, thread_id)
    
    def summarize_multiple_files(self, file_paths: List[str], thread_id: str = "default") -> Dict[str, Any]:
        """Summarize multiple files using the LangGraph workflow."""
        documents = self.document_loader.load_multiple_files(file_paths)
        return self._run_workflow(documents, thread_id)
    
    def _run_workflow(self, documents: List[Document], thread_id: str) -> Dict[str, Any]:
        """Run the summarization workflow."""
        if not documents:
            return {
                "final_summary": "No content to summarize.",
                "metadata": {"error": "No documents provided"}
            }
        
        # Initialize state
        initial_state: SummaryState = {
            "documents": documents,
            "initial_summary": "",
            "refined_summary": "",
            "key_points": [],
            "final_summary": "",
            "metadata": {
                "thread_id": thread_id,
                "step": "initialized",
                "completed": False,
                "document_count": len(documents)
            }
        }
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": thread_id}}
            result = self.workflow.invoke(initial_state, config)
            
            return {
                "final_summary": result["final_summary"],
                "initial_summary": result["initial_summary"],
                "refined_summary": result["refined_summary"],
                "key_points": result["key_points"],
                "metadata": result["metadata"]
            }
        
        except Exception as e:
            return {
                "final_summary": f"Error during summarization: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def get_workflow_state(self, thread_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.workflow.get_state(config)
            return state.values if state else None
        except Exception:
            return None
