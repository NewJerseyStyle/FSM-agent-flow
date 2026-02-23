"""
Example: Customer Service Workflow using Google Agent Development Kit (ADK)

This example demonstrates:
- Using Google ADK agents within LLM-FSM
- ADK's built-in tool management and orchestration
- ADK's native pause/resume capabilities
- Multi-agent systems with ADK
- State-based workflow with ADK agent execution
"""

from statemachine import State
from llm_fsm import (
    LLMStateMachine,
    ExecutionBreak,
    PersistentMemory
)
from llm_fsm.adk_client import ADKClient, create_adk_agent_with_tools, ADKStateMixin


# Define tools for ADK agent
def search_knowledge_base(query: str) -> str:
    """
    Search the company knowledge base for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results from knowledge base
    """
    # Mock implementation
    knowledge = {
        "refund policy": "Refunds are available within 30 days of purchase with receipt.",
        "shipping": "Standard shipping takes 5-7 business days. Express available.",
        "warranty": "All products come with 1-year manufacturer warranty.",
        "returns": "Returns accepted within 30 days in original packaging."
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return f"Knowledge Base Result: {value}"
    
    return "No relevant information found in knowledge base."


def create_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "normal"
) -> str:
    """
    Create a support ticket for the customer.
    
    Args:
        customer_name: Customer's name
        issue_description: Description of the issue
        priority: Ticket priority (low, normal, high, urgent)
        
    Returns:
        Ticket ID and confirmation
    """
    import random
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    
    return f"""Support ticket created successfully:
- Ticket ID: {ticket_id}
- Customer: {customer_name}
- Issue: {issue_description}
- Priority: {priority}
- Status: Open

A support agent will contact you within 24 hours."""


def escalate_to_human(
    reason: str,
    customer_info: str
) -> str:
    """
    Escalate the conversation to a human agent.
    This is a BREAKING operation - pauses the workflow.
    
    Args:
        reason: Reason for escalation
        customer_info: Customer information
        
    Returns:
        Confirmation message
    """
    print(f"\n{'='*60}")
    print("ESCALATION TO HUMAN AGENT")
    print(f"{'='*60}")
    print(f"Reason: {reason}")
    print(f"Customer: {customer_info}")
    print("Workflow paused - waiting for human agent to handle...")
    print(f"{'='*60}\n")
    
    # In real implementation, this would trigger notification system
    return "[WAITING FOR HUMAN AGENT]"


def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Send an email to the customer.
    
    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body
        
    Returns:
        Confirmation of email sent
    """
    print(f"\n📧 Email sent to {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}...")
    
    return f"Email sent successfully to {recipient}"


# Customer Service Workflow using ADK
class CustomerServiceWorkflow(LLMStateMachine, ADKStateMixin):
    """
    Customer service workflow powered by Google ADK.
    
    Flow:
    1. Initial triage - understand customer issue
    2. Information gathering - collect details
    3. Resolution attempt - try to resolve with AI
    4. Human escalation - if needed, escalate to human
    5. Follow-up - send summary email
    """
    
    idle = State(initial=True)
    triage = State()
    gathering = State()
    resolution = State()
    escalation = State()
    followup = State()
    done = State(final=True)
    
    # Transitions
    start = idle.to(triage)
    gather_info = triage.to(gathering)
    attempt_resolution = gathering.to(resolution)
    need_escalation = resolution.to(escalation)
    send_followup = resolution.to(followup)
    complete = followup.to(done)
    escalation_done = escalation.to(followup)
    
    def on_enter_triage(self, state_input=None):
        """Triage state: Understand the customer issue."""
        customer_message = state_input.get("message") if isinstance(state_input, dict) else str(state_input)
        customer_name = state_input.get("customer_name", "Customer") if isinstance(state_input, dict) else "Customer"
        
        # Store customer info in memory
        self.memory.working_memory.set(0, f"Customer: {customer_name}")
        self.memory.working_memory.set(1, f"Initial message: {customer_message}")
        
        # Use ADK agent directly (it manages tools internally)
        query = f"""A customer named {customer_name} has contacted us with the following message:

"{customer_message}"

Analyze this message and determine:
1. What is the main issue?
2. What category does this fall into? (refund, shipping, warranty, technical, other)
3. What is the urgency level? (low, normal, high, urgent)
4. What information do we need to gather?

Use the search_knowledge_base tool if relevant information might exist."""
        
        response = self.run_with_adk(
            query=query,
            state_name="triage"
        )
        
        # Store analysis in memory
        self.memory.working_memory.set(2, f"Triage analysis: {response}")
        
        return {
            "analysis": response,
            "customer_name": customer_name,
            "message": customer_message
        }
    
    def on_enter_gathering(self, state_input=None):
        """Gathering state: Collect additional information if needed."""
        analysis = state_input.get("analysis", "")
        
        query = f"""Based on the triage analysis:

{analysis}

Determine what additional information we need from the customer to resolve their issue.
Prepare questions to ask, or if we have enough information, indicate we're ready to proceed with resolution.

Search the knowledge base if needed to understand what information is typically required."""
        
        response = self.run_with_adk(
            query=query,
            state_name="gathering"
        )
        
        self.memory.working_memory.set(3, f"Information gathering: {response}")
        
        return {
            **state_input,
            "gathering_result": response
        }
    
    def on_enter_resolution(self, state_input=None):
        """Resolution state: Attempt to resolve the issue."""
        context = self.memory.to_context()
        
        query = f"""Now let's resolve the customer's issue.

{context}

Based on all the information gathered:
1. Search the knowledge base for relevant policies and solutions
2. If you can resolve the issue, provide a clear solution
3. If the issue requires human intervention (e.g., complex refunds, technical issues beyond knowledge base), 
   use the escalate_to_human tool to escalate

If escalation is needed, use: escalate_to_human(reason="why escalation needed", customer_info="summary")

If a support ticket should be created, use: create_support_ticket(customer_name, issue_description, priority)"""
        
        try:
            response = self.run_with_adk(
                query=query,
                state_name="resolution"
            )
            
            # Check if escalation was triggered (ADK would handle the tool call)
            if "[WAITING FOR HUMAN AGENT]" in response:
                self.memory.working_memory.set(4, "Status: Escalated to human agent")
                raise ExecutionBreak("Escalated to human agent")
            
            self.memory.working_memory.set(4, f"Resolution: {response}")
            
            return {
                **state_input,
                "resolution": response,
                "escalated": False
            }
            
        except ExecutionBreak:
            # Re-raise to pause workflow
            raise
    
    def on_enter_escalation(self, state_input=None):
        """Escalation state: Handle human agent interaction."""
        # This state is entered after human agent handles the case
        # In practice, this would be triggered by external system
        
        # Get human agent resolution from new input
        human_resolution = state_input.get("human_resolution", "Issue resolved by human agent")
        
        self.memory.working_memory.set(5, f"Human agent resolution: {human_resolution}")
        
        return {
            **state_input,
            "resolution": human_resolution,
            "escalated": True
        }
    
    def on_enter_followup(self, state_input=None):
        """Follow-up state: Send summary email to customer."""
        customer_name = self.memory.working_memory.get(0).replace("Customer: ", "")
        resolution = state_input.get("resolution", "")
        escalated = state_input.get("escalated", False)
        
        query = f"""Prepare a professional follow-up email for the customer.

Customer name: {customer_name}
Resolution: {resolution}
Was escalated: {escalated}

Use the send_email tool to send a summary email.
Email should be professional, empathetic, and include:
1. Acknowledgment of their issue
2. Summary of resolution
3. Next steps if any
4. Contact information for further assistance

Use: send_email(recipient="{customer_name}@example.com", subject="...", body="...")"""
        
        response = self.run_with_adk(
            query=query,
            state_name="followup"
        )
        
        self.memory.working_memory.set(6, f"Follow-up complete: {response}")
        
        return {
            "followup_sent": True,
            "summary": response
        }


# Example usage
def run_customer_service_example():
    """Run the customer service workflow with ADK."""
    
    print("="*60)
    print("Customer Service Workflow with Google ADK")
    print("="*60)
    
    # Create ADK agent with tools
    # Note: In real usage, configure with your Google Cloud credentials
    client = create_adk_agent_with_tools(
        model="gemini-2.0-flash",
        name="customer_service_agent",
        instruction="""You are a helpful customer service agent.
        
You have access to tools to:
- Search the knowledge base
- Create support tickets
- Escalate to human agents when needed
- Send emails to customers

Always be professional, empathetic, and solution-oriented.
Use tools proactively to help customers efficiently.""",
        tools=[
            search_knowledge_base,
            create_support_ticket,
            escalate_to_human,
            send_email
        ]
    )
    
    # Create memory with background context
    memory = PersistentMemory()
    memory.background.vision_mission = "Provide exceptional customer service experiences"
    memory.background.goals = "Resolve customer issues quickly and professionally"
    
    # Create workflow
    workflow = CustomerServiceWorkflow(
        llm_client=client,
        memory=memory,
        enable_summarization=True
    )
    
    # Simulate customer interaction
    customer_input = {
        "customer_name": "John Doe",
        "message": "I received a damaged product and would like a refund. Order #12345."
    }
    
    print("\n📞 Customer Contact:")
    print(f"   Customer: {customer_input['customer_name']}")
    print(f"   Message: {customer_input['message']}")
    
    try:
        # Start workflow
        print("\n1️⃣  Triage Phase...")
        workflow.start()
        result = workflow.execute_state("triage", state_input=customer_input)
        print(f"   ✓ Triage complete")
        
        # Gather information
        print("\n2️⃣  Information Gathering...")
        workflow.gather_info()
        result = workflow.execute_state("gathering", state_input=result)
        print(f"   ✓ Information gathered")
        
        # Attempt resolution
        print("\n3️⃣  Resolution Attempt...")
        workflow.attempt_resolution()
        result = workflow.execute_state("resolution", state_input=result)
        
        if result.get("escalated"):
            print(f"   ⚠ Escalated to human agent")
        else:
            print(f"   ✓ Resolved automatically")
            
            # Send follow-up
            print("\n4️⃣  Sending Follow-up...")
            workflow.send_followup()
            result = workflow.execute_state("followup", state_input=result)
            print(f"   ✓ Follow-up sent")
            
            workflow.complete()
            print("\n✅ Workflow complete!")
        
    except ExecutionBreak:
        print("\n⏸️  Workflow paused - waiting for human agent")
        print("   Saving state...")
        memory.save_to_file("customer_service_paused.json")
        print("   ✓ State saved")
        return workflow, memory
    
    return workflow, memory


def resume_after_human_agent():
    """Resume workflow after human agent handles the case."""
    
    print("\n" + "="*60)
    print("Resuming Customer Service Workflow")
    print("="*60)
    
    # Load saved state
    memory = PersistentMemory.load_from_file("customer_service_paused.json")
    
    # Recreate client and workflow
    client = create_adk_agent_with_tools(
        model="gemini-2.0-flash",
        name="customer_service_agent",
        instruction="You are a helpful customer service agent.",
        tools=[search_knowledge_base, create_support_ticket, escalate_to_human, send_email]
    )
    
    workflow = CustomerServiceWorkflow(llm_client=client, memory=memory)
    
    # Simulate human agent resolution
    human_input = {
        "human_resolution": "Refund of $99.99 processed. Replacement item shipped via express (2-day delivery). Tracking: 1Z999AA1012345678"
    }
    
    print("\n👤 Human Agent Resolution:")
    print(f"   {human_input['human_resolution']}")
    
    # Continue workflow
    print("\n3️⃣  Processing Escalation...")
    workflow.escalation_done()
    result = workflow.execute_state("escalation", state_input=human_input)
    print(f"   ✓ Escalation processed")
    
    print("\n4️⃣  Sending Follow-up...")
    workflow.send_followup()
    result = workflow.execute_state("followup", state_input=result)
    print(f"   ✓ Follow-up sent")
    
    workflow.complete()
    print("\n✅ Workflow complete!")
    
    return workflow, memory


if __name__ == "__main__":
    print("Google ADK Integration Example")
    print("This example demonstrates using ADK agents within LLM-FSM workflows")
    print("\nNote: This example requires Google ADK installation and configuration:")
    print("  pip install google-adk")
    print("  Configure Google Cloud credentials or Gemini API key")
    print("\n" + "="*60 + "\n")
    
    # Uncomment to run with actual ADK installation
    # workflow, memory = run_customer_service_example()
    
    # Later, if escalated:
    # workflow, memory = resume_after_human_agent()
