import os
import warnings
from typing import Dict, List, Any, TypedDict, Annotated
warnings.filterwarnings('ignore')

from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.tools import Tool
from langchain.tools import GoogleSerperAPITool, GoogleSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.web_scraper import WebScraperTool

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, tools_condition

# Set environment variables (for any external APIs you might need)
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"

# Initialize Ollama model
llm = Ollama(model="llama3")

# Define Pydantic models for structured outputs
class TicketSummary(BaseModel):
    customer_name: str = Field(description="Name of the customer")
    issue_type: str = Field(description="Category of the issue")
    priority: str = Field(description="Priority level (low, medium, high, critical)")
    summary: str = Field(description="Brief summary of the issue")

class ActionItem(BaseModel):
    task: str = Field(description="Action item to be completed")
    team: str = Field(description="Team responsible for the task")
    due_date: str = Field(description="Due date for the task")

class Recommendation(BaseModel):
    resolution_steps: List[str] = Field(description="Step-by-step resolution instructions")
    linked_tickets: List[str] = Field(description="Related ticket IDs")
    confidence: float = Field(description="Confidence score (0-1)")

# Define the state that will be passed between nodes
class AgentState(TypedDict):
    chat_log: str
    ticket_data: str
    query_classification: Dict[str, Any]
    summary: str
    action_items: List[Dict[str, Any]]
    routing_info: Dict[str, Any]
    recommendations: Dict[str, Any]
    resolution_time: str
    performance_metrics: Dict[str, Any]
    feedback: Dict[str, Any]
    next_step: str

# Create tools
search_tool = GoogleSerperAPITool()
scrape_tool = WebScraperTool()

tools = [search_tool, scrape_tool]
tool_executor = ToolExecutor(tools)

# Create node functions for each agent
def query_comprehender(state: AgentState) -> AgentState:
    """Understand and classify customer queries"""
    chat_log = state["chat_log"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Query Comprehender who can understand and classify customer support queries.
    
    Customer Chat Log:
    {chat_log}
    
    Classify this query in terms of issue type, urgency, and relevant tags.
    Format your response as a valid JSON object with the following structure:
    
    {{
        "customer_name": "Extract from chat or use 'Unknown'",
        "issue_type": "Technical/Billing/Account/etc",
        "priority": "low/medium/high/critical",
        "summary": "Brief summary of the issue"
    }}
    """)
    
    parser = PydanticOutputParser(pydantic_object=TicketSummary)
    chain = prompt | llm | parser
    
    result = chain.invoke({"chat_log": chat_log})
    
    # Update state with classification results
    state["query_classification"] = result.dict()
    state["next_step"] = "summarize"
    
    return state

def summarizer(state: AgentState) -> AgentState:
    """Generate concise summaries of customer conversations"""
    chat_log = state["chat_log"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Conversation Summarizer who excels at distilling essential information from support tickets.
    
    Customer Chat Log:
    {chat_log}
    
    Provide a concise summary of this conversation, highlighting the key issues, customer needs, and context.
    Keep your summary under 100 words.
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({"chat_log": chat_log})
    
    # Update state with summary
    state["summary"] = result
    state["next_step"] = "extract_actions"
    
    return state

def action_extractor(state: AgentState) -> AgentState:
    """Extract tasks or escalations from conversations"""
    chat_log = state["chat_log"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are an Action Item Extractor who identifies follow-up tasks from customer support conversations.
    
    Customer Chat Log:
    {chat_log}
    
    Extract actionable tasks that need to be completed to resolve this issue.
    Format each task as a JSON object with the following structure:
    
    {{
        "task": "Description of the task",
        "team": "Team responsible (Technical/Billing/Management/etc)",
        "due_date": "Estimated due date"
    }}
    
    Return a list of these objects.
    """)
    
    parser = PydanticOutputParser(pydantic_object=ActionItem)
    chain = prompt | llm
    
    result = chain.invoke({"chat_log": chat_log})
    
    # Parse the action items (assuming the LLM returns a valid format)
    # In production, you'd want more robust parsing and error handling
    import json
    try:
        action_items = json.loads(result)
        if not isinstance(action_items, list):
            action_items = [action_items]
    except:
        action_items = [{"task": "Review customer issue", "team": "Support", "due_date": "ASAP"}]
    
    # Update state with action items
    state["action_items"] = action_items
    state["next_step"] = "route_tasks"
    
    return state

def task_dispatcher(state: AgentState) -> AgentState:
    """Assign issues to the correct teams efficiently"""
    action_items = state["action_items"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Task Dispatcher who efficiently routes support issues to the correct teams.
    
    Action Items:
    {action_items}
    
    For each action item, verify the team assignment and provide routing information.
    Return your analysis as a JSON object with team assignments and justification.
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({"action_items": action_items})
    
    # Update state with routing information
    state["routing_info"] = {"routing_analysis": result}
    state["next_step"] = "recommend_solutions"
    
    return state

def solution_advisor(state: AgentState) -> AgentState:
    """Recommend solutions from historical tickets and documentation"""
    chat_log = state["chat_log"]
    ticket_data = state["ticket_data"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Solution Advisor who recommends fixes for customer problems.
    
    Customer Issue:
    {chat_log}
    
    Historical Ticket Data:
    {ticket_data}
    
    Provide step-by-step resolution recommendations with confidence scores.
    Format your response as a valid JSON object with:
    - resolution_steps: a list of steps to resolve the issue
    - linked_tickets: a list of related ticket IDs (use placeholders if unknown)
    - confidence: a score between 0 and 1 indicating certainty in the recommendation
    """)
    
    parser = PydanticOutputParser(pydantic_object=Recommendation)
    chain = prompt | llm
    
    result = chain.invoke({"chat_log": chat_log, "ticket_data": ticket_data})
    
    # Update state with recommendations
    # In production, you'd want more robust parsing and error handling
    import json
    try:
        recommendations = json.loads(result)
    except:
        recommendations = {
            "resolution_steps": ["Investigate issue further", "Check system status", "Contact customer"],
            "linked_tickets": ["UNKNOWN"],
            "confidence": 0.7
        }
    
    state["recommendations"] = recommendations
    state["next_step"] = "predict_time"
    
    return state

def time_predictor(state: AgentState) -> AgentState:
    """Predict and minimize resolution time"""
    issue_type = state["query_classification"]["issue_type"]
    priority = state["query_classification"]["priority"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Resolution Time Estimator who predicts ticket resolution times.
    
    Issue Type: {issue_type}
    Priority: {priority}
    
    Based on this information, estimate the resolution time for this issue.
    Provide your estimate along with justification.
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({"issue_type": issue_type, "priority": priority})
    
    # Update state with resolution time estimate
    state["resolution_time"] = result
    state["next_step"] = "monitor_metrics"
    
    return state

def metrics_monitor(state: AgentState) -> AgentState:
    """Track performance and identify bottlenecks in support workflows"""
    ticket_data = state["ticket_data"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Support Metrics Monitor who analyzes operational performance.
    
    Ticket Data:
    {ticket_data}
    
    Analyze this data to identify bottlenecks, delays, and team performance issues.
    Provide insights for optimizing the support workflow.
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({"ticket_data": ticket_data})
    
    # Update state with performance metrics
    state["performance_metrics"] = {"analysis": result}
    state["next_step"] = "learn_from_feedback"
    
    return state

def knowledge_learner(state: AgentState) -> AgentState:
    """Improve over time by learning from closed tickets and feedback"""
    ticket_data = state["ticket_data"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Feedback and Learning Agent who improves support strategies.
    
    Ticket Data:
    {ticket_data}
    
    Based on this data, identify patterns and suggest improvements to the support process.
    Focus on learning from past resolutions and customer feedback.
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({"ticket_data": ticket_data})
    
    # Update state with feedback-based learning
    state["feedback"] = {"learnings": result}
    state["next_step"] = "end"
    
    return state

# Create the graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_comprehender", query_comprehender)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("action_extractor", action_extractor)
    workflow.add_node("task_dispatcher", task_dispatcher)
    workflow.add_node("solution_advisor", solution_advisor)
    workflow.add_node("time_predictor", time_predictor)
    workflow.add_node("metrics_monitor", metrics_monitor)
    workflow.add_node("knowledge_learner", knowledge_learner)
    
    # Define edges between nodes based on the next_step field
    workflow.add_conditional_edges(
        "query_comprehender",
        lambda state: state["next_step"],
        {
            "summarize": "summarizer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "summarizer",
        lambda state: state["next_step"],
        {
            "extract_actions": "action_extractor",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "action_extractor",
        lambda state: state["next_step"],
        {
            "route_tasks": "task_dispatcher",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "task_dispatcher",
        lambda state: state["next_step"],
        {
            "recommend_solutions": "solution_advisor",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "solution_advisor",
        lambda state: state["next_step"],
        {
            "predict_time": "time_predictor",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "time_predictor",
        lambda state: state["next_step"],
        {
            "monitor_metrics": "metrics_monitor",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "metrics_monitor",
        lambda state: state["next_step"],
        {
            "learn_from_feedback": "knowledge_learner",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "knowledge_learner",
        lambda state: state["next_step"],
        {
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("query_comprehender")
    
    return workflow

# Run the workflow
def run_support_system(chat_log: str, ticket_data: str):
    workflow = build_graph().compile()
    
    # Initialize the state
    initial_state = {
        "chat_log": chat_log,
        "ticket_data": ticket_data,
        "query_classification": {},
        "summary": "",
        "action_items": [],
        "routing_info": {},
        "recommendations": {},
        "resolution_time": "",
        "performance_metrics": {},
        "feedback": {},
        "next_step": ""
    }
    
    # Execute the workflow
    result = workflow.invoke(initial_state)
    
    # Save outputs to files
    import json
    
    with open("query_classification.json", "w") as f:
        json.dump(result["query_classification"], f, indent=2)
    
    with open("conversation_summary.md", "w") as f:
        f.write(result["summary"])
    
    with open("action_items.json", "w") as f:
        json.dump(result["action_items"], f, indent=2)
    
    with open("task_routing.json", "w") as f:
        json.dump(result["routing_info"], f, indent=2)
    
    with open("resolution_advice.json", "w") as f:
        json.dump(result["recommendations"], f, indent=2)
    
    with open("resolution_time.txt", "w") as f:
        f.write(result["resolution_time"])
    
    with open("performance_report.md", "w") as f:
        f.write(result["performance_metrics"]["analysis"])
    
    with open("feedback_learning.md", "w") as f:
        f.write(result["feedback"]["learnings"])
    
    return result

# Sample input (simulate a customer support ticket log)
if __name__ == "__main__":
    chat_log = """
    Customer: I'm having trouble with your app. It keeps disconnecting every few minutes.
    Support: I'm sorry to hear that. Can you tell me what device you're using?
    Customer: I'm using a Samsung Galaxy with Android 13.
    Support: Have you tried reinstalling the app?
    Customer: Yes, I've tried that twice already. Still getting error code ERR-42.
    Customer: This is the second time I've had to contact support about this issue.
    """
    
    ticket_data = """
    Data from 1000+ past tickets including resolution time, team assignment, success rate, and user feedback.
    Error code ERR-42 has previously been associated with API connectivity issues and background process termination.
    Average resolution time for connectivity issues is 2.3 days.
    """
    
    result = run_support_system(chat_log, ticket_data)
    print("Support workflow completed successfully.")
    print(f"Summary: {result['summary'][:100]}...")
