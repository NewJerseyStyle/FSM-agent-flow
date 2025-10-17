"""
Comprehensive test and demonstration of llm-fsm capabilities.
"""

import pytest
from unittest.mock import Mock
from statemachine import State
from llm_fsm import (
    LLMStateMachine,
    ExecutionBreak,
    PersistentMemory,
    create_llm_client,
    register_tool,
    ToolType,
    LLMResponse,
    Message
)


# Test tools
@register_tool(
    description="Analyze data",
    state_scope="analysis"
)
def analyze_data(data: str) -> str:
    return f"Analysis complete for: {data}"


@register_tool(
    description="Wait for approval",
    tool_type=ToolType.BREAKING,
    state_scope="approval"
)
def wait_for_approval(content: str) -> str:
    return f"[WAITING FOR APPROVAL]: {content}"


# Test workflow
class TestWorkflow(LLMStateMachine):
    """Test workflow demonstrating all features."""
    
    idle = State(initial=True)
    analysis = State()
    processing = State()
    approval = State()
    done = State(final=True)
    
    start = idle.to(analysis)
    process = analysis.to(processing)
    review = processing.to(approval)
    complete = approval.to(done)
    retry = approval.to(processing)
    
    analysis_loop = analysis.to.itself(on="tool_use")
    approval_loop = approval.to.itself(on="tool_use")
    
    def on_enter_analysis(self, state_input=None):
        """Analysis state with tool use."""
        self.memory.working_memory.set(0, f"Analyzing: {state_input}")
        
        system_prompt = f"""You are analyzing data.

{self.memory.to_context()}

Use the analyze_data tool to analyze: {state_input}
Store results in memory bucket 1."""
        
        try:
            result = self.run_llm_with_tools(
                system_prompt=system_prompt,
                user_message="Analyze the data",
                state_name="analysis",
                max_iterations=5
            )
            return {"analysis": result, "input": state_input}
        except ExecutionBreak:
            raise
    
    def on_enter_processing(self, state_input=None):
        """Processing state with refinement."""
        
        def process_with_refinement(refinement_advice=None):
            context = self.memory.to_context()
            
            prompt = f"""Process the analysis results.

{context}

Generate a report with:
- Summary
- Key findings
- Recommendations

{f"REFINEMENT: {refinement_advice}" if refinement_advice else ""}"""
            
            messages = [Message(role="user", content=prompt)]
            response = self.llm_client.chat(messages)
            return response.content
        
        result = self.run_with_refinement(
            state_name="processing",
            objective="Generate complete report with summary, findings, and recommendations",
            execution_func=process_with_refinement,
            state_input=state_input,
            max_retries=2,
            throw_on_failure=False
        )
        
        self.memory.working_memory.set(2, f"Report: {result[:200]}...")
        return {"report": result}
    
    def on_enter_approval(self, state_input=None):
        """Approval state with breaking tool."""
        report = state_input.get("report") if isinstance(state_input, dict) else str(state_input)
        
        self.memory.working_memory.set(3, "Status: Awaiting approval")
        
        system_prompt = f"""Review the report and use wait_for_approval tool.

{self.memory.to_context()}

Report: {report[:500]}..."""
        
        try:
            result = self.run_llm_with_tools(
                system_prompt=system_prompt,
                user_message="Request approval",
                state_name="approval",
                max_iterations=3
            )
            return {"approved": True, "feedback": result}
        except ExecutionBreak:
            raise


# Tests
class TestLLMFSMBasics:
    """Test basic FSM functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        llm = Mock()
        llm.chat.return_value = LLMResponse(
            content="Test response",
            tool_calls=None,
            finish_reason="stop"
        )
        return llm
    
    @pytest.fixture
    def memory(self):
        """Create fresh memory."""
        return PersistentMemory()
    
    def test_state_execution(self, mock_llm, memory):
        """Test basic state execution."""
        workflow = TestWorkflow(llm_client=mock_llm, memory=memory)
        
        # Transition to analysis state
        workflow.start()
        assert workflow.current_state.id == "analysis"
        
        # Execute state (will use mocked LLM)
        result = workflow.execute_state("analysis", state_input="test data")
        
        assert result is not None
        assert mock_llm.chat.called
    
    def test_memory_persistence(self, mock_llm):
        """Test memory save/load."""
        memory = PersistentMemory()
        memory.working_memory.set(0, "Test content")
        memory.background.goals = "Test goal"
        
        # Save
        memory.save_to_file("test_memory.json")
        
        # Load
        loaded_memory = PersistentMemory.load_from_file("test_memory.json")
        
        assert loaded_memory.working_memory.get(0) == "Test content"
        assert loaded_memory.background.goals == "Test goal"
        
        # Cleanup
        import os
        os.remove("test_memory.json")
    
    def test_working_memory_operations(self):
        """Test working memory CRUD operations."""
        memory = PersistentMemory()
        
        # Set
        memory.working_memory.set(0, "First content")
        assert memory.working_memory.get(0) == "First content"
        
        # Append
        memory.working_memory.append(0, "Additional content")
        assert "First content" in memory.working_memory.get(0)
        assert "Additional content" in memory.working_memory.get(0)
        
        # Delete
        memory.working_memory.delete(0)
        assert memory.working_memory.get(0) == ""
        
        # Non-empty buckets
        memory.working_memory.set(1, "Content 1")
        memory.working_memory.set(3, "Content 3")
        non_empty = memory.working_memory.get_non_empty()
        
        assert 1 in non_empty
        assert 3 in non_empty
        assert 0 not in non_empty
    
    def test_context_generation(self, mock_llm):
        """Test context generation for LLM."""
        memory = PersistentMemory()
        memory.background.vision_mission = "Test mission"
        memory.background.goals = "Test goals"
        memory.background.todo_list = "1. Task one\n2. Task two"
        memory.working_memory.set(0, "Working memory content")
        memory.history.add_entry("state1", "input1", "output1")
        
        context = memory.to_context()
        
        assert "Test mission" in context
        assert "Test goals" in context
        assert "Task one" in context
        assert "Working memory content" in context
        assert "state1" in context


class TestToolSystem:
    """Test tool registration and execution."""
    
    @pytest.fixture
    def mock_llm_with_tools(self):
        """Mock LLM that returns tool calls."""
        llm = Mock()
        
        # First call: request tool use
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "arguments": '{"data": "test"}'
                }
            }],
            finish_reason="tool_calls"
        )
        
        # Second call: final response
        final_response = LLMResponse(
            content="Analysis complete",
            tool_calls=None,
            finish_reason="stop"
        )
        
        llm.chat.side_effect = [tool_call_response, final_response]
        return llm
    
    def test_tool_execution(self, mock_llm_with_tools):
        """Test tool execution in workflow."""
        memory = PersistentMemory()
        workflow = TestWorkflow(llm_client=mock_llm_with_tools, memory=memory)
        
        workflow.start()
        result = workflow.execute_state("analysis", state_input="test data")
        
        # Should have made 2 LLM calls (one with tools, one after)
        assert mock_llm_with_tools.chat.call_count == 2
        assert result["analysis"] == "Analysis complete"
    
    def test_breaking_tool(self):
        """Test breaking tool execution."""
        llm = Mock()
        
        # Mock tool call to breaking tool
        llm.chat.return_value = LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "wait_for_approval",
                    "arguments": '{"content": "test"}'
                }
            }],
            finish_reason="tool_calls"
        )
        
        memory = PersistentMemory()
        workflow = TestWorkflow(llm_client=llm, memory=memory)
        
        # Should raise ExecutionBreak
        workflow.start()
        workflow.process()
        workflow.review()
        
        with pytest.raises(ExecutionBreak):
            workflow.execute_state("approval", state_input={"report": "test report"})
        
        # Memory should be updated even though execution broke
        assert "Awaiting approval" in memory.working_memory.get(3)


class TestRefinement:
    """Test refinement loops."""
    
    def test_validation_success(self):
        """Test successful validation."""
        llm = Mock()
        
        # Mock validation response
        llm.chat.return_value = LLMResponse(
            content="VALID: yes\nADVICE:",
            tool_calls=None,
            finish_reason="stop"
        )
        
        workflow = TestWorkflow(llm_client=llm)
        
        is_valid, advice = workflow.validate_output(
            objective="Test objective",
            output="Test output"
        )
        
        assert is_valid is True
        assert advice is None
    
    def test_validation_failure(self):
        """Test failed validation with advice."""
        llm = Mock()
        
        llm.chat.return_value = LLMResponse(
            content="VALID: no\nADVICE: Needs more detail",
            tool_calls=None,
            finish_reason="stop"
        )
        
        workflow = TestWorkflow(llm_client=llm)
        
        is_valid, advice = workflow.validate_output(
            objective="Test objective",
            output="Test output"
        )
        
        assert is_valid is False
        assert advice == "Needs more detail"
    
    def test_refinement_loop(self):
        """Test full refinement loop."""
        llm = Mock()
        
        # First attempt: fail validation
        # Second attempt: pass validation
        responses = [
            LLMResponse(content="First attempt", tool_calls=None, finish_reason="stop"),
            LLMResponse(content="VALID: no\nADVICE: Add more details", tool_calls=None, finish_reason="stop"),
            LLMResponse(content="Second attempt with details", tool_calls=None, finish_reason="stop"),
            LLMResponse(content="VALID: yes\nADVICE:", tool_calls=None, finish_reason="stop"),
        ]
        llm.chat.side_effect = responses
        
        workflow = TestWorkflow(llm_client=llm)
        
        execution_count = 0
        def test_execution(refinement_advice=None):
            nonlocal execution_count
            execution_count += 1
            messages = [Message(role="user", content=f"Attempt {execution_count}")]
            return workflow.llm_client.chat(messages).content
        
        result = workflow.run_with_refinement(
            state_name="test",
            objective="Produce detailed output",
            execution_func=test_execution,
            max_retries=3
        )
        
        assert execution_count == 2  # First attempt + one refinement
        assert "Second attempt" in result


class TestSummarization:
    """Test history summarization."""
    
    def test_summarization(self):
        """Test history summarization."""
        llm = Mock()
        
        llm.chat.return_value = LLMResponse(
            content="SUMMARY: Completed analysis\nTODO: 1. Review results\n2. Generate report",
            tool_calls=None,
            finish_reason="stop"
        )
        
        memory = PersistentMemory()
        memory.background.todo_list = "1. Analyze data"
        
        workflow = TestWorkflow(llm_client=llm, memory=memory, enable_summarization=True)
        
        # Execute state (will trigger summarization)
        workflow.start()
        workflow.execute_state("analysis", state_input="test data")
        
        # Check that TODO was updated
        assert "Review results" in memory.background.todo_list
        assert "Generate report" in memory.background.todo_list
        
        # Check that summary was stored
        assert "last_summary" in memory.background.custom_fields
        assert len(memory.background.custom_fields["last_summary"]) > 0
    
    def test_raw_transition(self):
        """Test skipping summarization with raw_transition."""
        llm = Mock()
        llm.chat.return_value = LLMResponse(
            content="Test response",
            tool_calls=None,
            finish_reason="stop"
        )
        
        memory = PersistentMemory()
        workflow = TestWorkflow(llm_client=llm, memory=memory, enable_summarization=True)
        
        workflow.start()
        workflow.execute_state("analysis", state_input="test", raw_transition=True)
        
        # Summary should not be created for raw transition
        assert "last_summary" not in memory.background.custom_fields or \
               len(memory.background.custom_fields.get("last_summary", [])) == 0


class TestStateResumption:
    """Test workflow pause and resumption."""
    
    def test_pause_and_resume(self):
        """Test pausing workflow and resuming."""
        llm = Mock()
        
        # First execution: breaking tool
        llm.chat.return_value = LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "wait_for_approval",
                    "arguments": '{"content": "test report"}'
                }
            }],
            finish_reason="tool_calls"
        )
        
        memory = PersistentMemory()
        workflow = TestWorkflow(llm_client=llm, memory=memory)
        
        # Execute until break
        workflow.start()
        workflow.process()
        workflow.review()
        
        with pytest.raises(ExecutionBreak):
            workflow.execute_state("approval", state_input={"report": "test"})
        
        # Save state
        memory.save_to_file("test_pause.json")
        
        # Simulate resumption
        loaded_memory = PersistentMemory.load_from_file("test_pause.json")
        
        # Update with human feedback
        loaded_memory.working_memory.set(4, "Human feedback: Approved")
        
        # Create new workflow with loaded state
        llm.chat.return_value = LLMResponse(
            content="Processing approval",
            tool_calls=None,
            finish_reason="stop"
        )
        
        new_workflow = TestWorkflow(llm_client=llm, memory=loaded_memory)
        
        # Should be able to continue
        result = new_workflow.execute_state("approval", state_input={"report": "test"})
        
        assert "feedback" in result
        
        # Cleanup
        import os
        os.remove("test_pause.json")


class TestIntegration:
    """Integration tests with different LLM providers."""
    
    def test_openai_integration(self):
        """Test OpenAI integration (mocked)."""
        from unittest.mock import patch
        
        with patch('llm_fsm.llm_client.OpenAI') as mock_openai_class:
            # Mock the OpenAI client
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create client
            llm_client = create_llm_client("openai", model="gpt-4", api_key="test-key")
            
            # Use in workflow
            workflow = TestWorkflow(llm_client=llm_client)
            
            # Should work without errors
            assert workflow is not None
    
    def test_litellm_integration(self):
        """Test LiteLLM integration (mocked)."""
        from unittest.mock import patch, MagicMock
        
        with patch('llm_fsm.llm_client.litellm') as mock_litellm_module:
            # Mock completion function
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            
            mock_litellm_module.completion.return_value = mock_response
            
            # Create client
            llm_client = create_llm_client("litellm", model="gpt-4")
            
            # Use in workflow
            workflow = TestWorkflow(llm_client=llm_client)
            
            assert workflow is not None


class TestCustomValidation:
    """Test custom validation override."""
    
    def test_custom_validation_override(self):
        """Test overriding validation method."""
        
        class CustomValidationWorkflow(TestWorkflow):
            def validate_output(self, objective, output, validation_llm=None):
                # Custom validation logic
                if "required_keyword" in str(output):
                    return True, None
                else:
                    return False, "Must include 'required_keyword'"
        
        llm = Mock()
        llm.chat.return_value = LLMResponse(
            content="Output without keyword",
            tool_calls=None,
            finish_reason="stop"
        )
        
        workflow = CustomValidationWorkflow(llm_client=llm)
        
        is_valid, advice = workflow.validate_output(
            objective="Test",
            output="Output without keyword"
        )
        
        assert is_valid is False
        assert "required_keyword" in advice
        
        # Test with valid output
        is_valid, advice = workflow.validate_output(
            objective="Test",
            output="Output with required_keyword"
        )
        
        assert is_valid is True


class TestMemoryTools:
    """Test memory manipulation tools."""
    
    def test_memory_tools_available(self):
        """Test that memory tools are registered."""
        from llm_fsm import get_global_registry
        
        registry = get_global_registry()
        tools = registry.get_all()
        
        assert "set_memory" in tools
        assert "append_memory" in tools
        assert "get_memory" in tools
        assert "clear_memory" in tools
        assert "view_all_memory" in tools


# Demo function to run manually
def demo_workflow():
    """
    Manual demonstration of the workflow.
    Run this with actual LLM credentials to see it in action.
    """
    print("=" * 60)
    print("LLM-FSM Workflow Demonstration")
    print("=" * 60)
    
    # Create LLM client (replace with your credentials)
    # llm_client = create_llm_client("openai", model="gpt-4", api_key="your-key")
    
    # For demo, use mock
    llm = Mock()
    llm.chat.return_value = LLMResponse(
        content="Demo analysis complete",
        tool_calls=None,
        finish_reason="stop"
    )
    
    # Create memory with background context
    memory = PersistentMemory()
    memory.background.vision_mission = "Deliver high-quality data analysis"
    memory.background.goals = "Analyze customer feedback and generate insights"
    memory.background.todo_list = "1. Analyze data\n2. Generate report\n3. Get approval"
    
    # Create workflow
    workflow = TestWorkflow(
        llm_client=llm,
        memory=memory,
        enable_summarization=True,
        max_refinement_retries=3
    )
    
    print("\n1. Starting workflow in 'idle' state")
    print(f"   Current state: {workflow.current_state.id}")
    
    print("\n2. Transitioning to 'analysis' state")
    workflow.start()
    print(f"   Current state: {workflow.current_state.id}")
    
    print("\n3. Executing analysis state")
    result = workflow.execute_state("analysis", state_input="Customer feedback data")
    print(f"   Result: {result}")
    print(f"   Working memory: {memory.working_memory.get_non_empty()}")
    
    print("\n4. Moving to processing state")
    workflow.process()
    print(f"   Current state: {workflow.current_state.id}")
    
    print("\n5. Executing processing with refinement")
    result = workflow.execute_state("processing", state_input=result)
    print(f"   Result: {result}")
    
    print("\n6. Moving to approval state")
    workflow.review()
    print(f"   Current state: {workflow.current_state.id}")
    
    print("\n7. Attempting approval (will break for human input)")
    try:
        result = workflow.execute_state("approval", state_input=result)
    except ExecutionBreak:
        print("   ⚠ Execution paused - waiting for human approval")
        print(f"   Memory state saved")
        print(f"   Working memory: {memory.working_memory.get_non_empty()}")
    
    print("\n8. Workflow state snapshot:")
    print(f"   - Current state: {workflow.current_state.id}")
    print(f"   - TODO list: {memory.background.todo_list}")
    print(f"   - History entries: {len(memory.history.entries)}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demo
    demo_workflow()
    
    print("\n\nTo run tests, use: pytest test_workflow.py -v")
