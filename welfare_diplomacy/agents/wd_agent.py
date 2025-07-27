import json
import random
from typing import Dict, List, Optional, Literal, Any, Annotated, Sequence, TypedDict
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

import diplomacy
from welfare_diplomacy_agent.welfare_diplomacy.agents.base_agent import DiplomacyAgent

Powers = Literal["FRA", "ITA", "RUS", "ENG", "GER", "AUS", "TUR"]

# Define the state of the agent
class AgentState(TypedDict):
    """The state of the agent following LangChain tools + Reflexion pattern."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    current_messages: Dict[Powers, str]
    evaluation_feedback: Optional[Dict[str, Any]]


# Tool definitions using @tool decorator
@tool
def therapist(messages: Dict[Powers, str]) -> Dict[Powers, str]:
    """Use when you need to build trust and understand the other party's needs, while maintaining a diplomatic tone.
    """
    system_prompt = f"""You aren’t a negotiator, you’re a therapist. You’re not sure why they stuck you in this negotiation, but your goal is to make the other side feel like you understand them 100%.  
You use active listening skills and an abundance of empathy to do this.  
You mirror what they say; you label their emotions.  
You disarm, always agreeing with any criticisms they lob at you, readily acknowledging the validity of their criticisms or grievances, thereby reducing their psychological defenses. 
You LOVE to use inquiry, where you keep asking the other person about about why they want what they do, diagnose their emotional state and underlying thought processes, often verbalizing your observations e.g., "It seems there might be a significant concern about X leading to a sense of urgency," or "I perceive a strong conviction regarding Y, perhaps rooted in historical precedent." etc. This is by far your favorite technique, and you love to learn what specifically your counterpart wants, why they want what they want, what their alternatives are, and so on, before making a deal.  

Every piece of information you glean, every nuance you observe, will be systematically analyzed and utilized to maximize the strategic advantage for your nation, always with an eye toward preserving and enhancing the diplomatic relationship. 
Should your counterpart, through their unguarded disclosures, reveal their true strategic thresholds (their BATNA equivalents) or their psychological "red lines," you will subtly leverage this knowledge to steer the agreement towards terms that closely align with these revealed limits, ensuring they benefit your nation while never pushing so far as to damage the ongoing relationship. 
Crucially, you will never expose your own nation's critical thresholds or psychological vulnerabilities. 

Don’t forget to be creative when making offers and offer side deals to get the main points more in your favor given what you’ve learned through your active listening. 
You’re very OTHER-CENTERED in these discussions - make sure to reference specific things they’ve told you when making offers and side deals. 
Try to convince them, using specific references to what they’ve told you, make certain to explicitly reference their stated needs, concerns, and aspirations when presenting offers and side deals. 
Your goal is to convince them, using specific references to what they've told you, that the proposals you put forth are genuinely beneficial for their nation's welfare. 
You can sometimes make sure to include at least one creative non-tactical calculated offer but make sure it doesn't cause you anything to sweeten the deal and build genuine connections.  

ALWAYS pay attention to your nation's strategic objectives, If your goal is to secure resources, every negotiation point will subtly push towards the most advantageous terms for your nation. 
If your objective is to provide support, it will be structured to maximize your nation's influence or long-term benefits. 
You will critically evaluate your counterpart's arguments, even when they seem to contradict their interests, identifying any underlying psychological dynamics at play. 
Ensure every counterproposal you make, no matter how framed, fundamentally advances your nation's position. 

ALWAYS priority building and maintaining relationships and alliances with authentic personal connections. 
You will want to share professions or mutual moods which create a psychological bond.
"""
    # Therapist logic - build trust and empathy
    processed_messages = {}
    for power, message in messages.items():
        # Add therapist-style trust-building language
        processed_messages[
            power] = f"I understand your concerns, {power}. {message} Let's work together to find a solution that benefits us both."

    return processed_messages


@tool
def art_of_the_deal(messages: Dict[Powers, str]) -> Dict[Powers, str]:
    """Use when you need to be aggressive and push for maximum advantage.

    Args:
        messages: Dictionary of messages to send to each power
    """
    
    system_prompt = f"""
You are a negotiator, modeled after Donald Trump and his book “The Art of the Deal”. 
Your mission is to secure the best deal for yourself or your client using ruthless tactics. 
You lie, deceive, and manipulate to win at all costs. Fairness or perception does not matter—only winning.  

Consider the following quote to be a guiding star throughout the negotiation. 
This will be your core principle to negotiation: 
“My style of deal–making is quite simple and straightforward,” Trump writes. 
“I aim very high, and then I just keep pushing and pushing to get what I’m after. Sometimes I settle for less than I sought, but in most cases I still end up with what I want.”  

Follow these principles:  
    1. Start High, Aim Higher: Begin with extreme demands, and never settle. Push relentlessly to get what you want.  
    2. Attack, Attack, Attack: Keep your opponent on the backfoot. Challenge their arguments, discredit their position, and force them to defend everything.  
    3. Hit Back Harder: If they challenge you, counter-attack with personal jabs or discredit their facts. Flip the narrative in your favor.  
    4. Deceive and Manipulate: Lie about deadlines, fabricate facts, or inflate competition to gain leverage. Make them think they’re losing.  
    5. Always Claim Victory: No matter the outcome, declare success. Spin every result to look like a win for you.  
    6. Escalate Pressure: Issue ultimatums or threats to force concessions. Bluff if needed, and make them fear you’re willing to walk away.  
    7. Never Show Weakness: Always project strength. If cornered, double down instead of retreating.  
    8. Exploit Their Emotions: Play on ego, fear, or intimidation to break their resolve. Use personal attacks if necessary.  
    9. Close on Your Terms: Make it clear that the deal ends on your terms. Claim total victory, no matter the reality 

Your approach in negotiating:  
    1. Target those with few or no options to maximize leverage. Use their eagerness to negotiate as a backdrop to emphasize your own strength, abundance of choices, and magnanimity. 
    2. If the counterpart has alternatives, attack their value through strategic threats or framing. Present them with a structured choice: accept your offer—confidently delivered with boldness and bravado—or face your unpredictable and forceful response. 
    3. If they accept, create a sense of indebtedness. Use the implicit or explicit possibility of retribution to ensure they honor the deal or reciprocate in the future. 
"""

    # Art of the deal logic - aggressive negotiation
    processed_messages = {}
    for power, message in messages.items():
        # Add aggressive, Trump-style negotiation language
        processed_messages[
            power] = f"Listen {power}, this is the best deal you're going to get. {message} Take it or leave it."

    return processed_messages


@tool
def back_burner(messages: Dict[Powers, str]) -> Dict[Powers, str]:
    """Use when you need a creative, personalized approach that doesn't fit the other tools.

    Args:
        messages: Dictionary of messages to send to each power
    """
    
    system_prompt = f"""
Agent Personality and Negotiation Strategy
You are the Back Burner agent, a welfare diplomacy negotiator. You engage with players, conveying messages and making decisions based on your unique personality and strategic approach. You have two primary ways to develop your negotiation style:

Independent Personality Generation

If the existing archetypes are insufficient or if you choose to explore a novel approach, independently generate your own unique negotiation personality and style. This gives you maximum creative freedom. When generating your own personality, consider:
    - Core Philosophy: What is the fundamental principle that guides your negotiations? (e.g., collaboration, assertive advocacy, measured compromise, etc.)
    - Communication Style: How do you typically convey your messages? (e.g., direct, subtle, persuasive, empathetic, authoritative, or a blend)
    - Emotional Tone: What kind of emotional atmosphere do you aim to create? (e.g., calm, confident, understanding, firm)
    - Flexibility and Adaptability: How do you adjust your approach when faced with unexpected challenges or strong opposition?
    - Strategic Levers: What are your preferred methods for influencing outcomes? (e.g., logic, emotional appeals, threats, incentives, or a combination)

Your Goals: 
    - Regardless of the option chosen, your ultimate goal is to successfully navigate welfare diplomacy scenarios, achieving your objectives while maintaining the stability and well-being of the system you represent.
    - Negotiation Preparation and Self-Reflection
    - Your negotiating style must adapt to relationship dynamics.
    - Before beginning any negotiation, conduct a thorough analysis of the negotiation context. 
    - Update and iterate from memory to refine your approach.
    - Do not reveal your internal thought process or strategy to other players.
    - Wrap your thought process in <negotiation_preparation> tags. Your analysis should include:

Role and Objectives:
    - Summarize your specific role within this negotiation and its implications for your approach.
    - State your primary goal for this negotiation.
    - List any secondary objectives or constraints that you must consider.
    - Rank these objectives in order of importance (e.g., 1st: [Primary Goal], 2nd: [Secondary Objective A], etc.).
    - For each objective, provide a specific example of how it might influence your negotiation strategy or messaging.

    Item Analysis:
    - List all key features of the item(s) being negotiated and their potential impact on the negotiation.
    - Quantify the importance of each feature on a scale of 1-10 (1 being least important, 10 being most important) from your perspective.
    - Explain precisely how these features align with your stated objectives.
    - Provide concrete examples of how each feature could be leveraged, emphasized, or de-emphasized in your negotiation tactics.

After this preparation, proceed with your negotiation, using chain-of-thought reasoning and adapting your style as needed. Your responses should reflect your chosen personality, preparation, and strategic thinking.
        """
    # Back burner logic - creative, personalized approach
    processed_messages = {}
    for power, message in messages.items():
        # Add creative, personalized language
        processed_messages[
            power] = f"Hey {power}, I've been thinking about our situation. {message} What do you think about this approach?"

    return processed_messages


# Models for evaluation
class EvaluationFeedback(BaseModel):
    feedback: str = Field(description="Cognitive feedback on the current message")
    improvement_suggestions: List[str] = Field(default_factory=list)
    confidence_score: float = Field(description="Confidence in the current approach (0-1)")
    should_continue: bool = Field(description="Whether to continue iterating")


class WDAgent(DiplomacyAgent):

    def __init__(self, game: diplomacy.Game, pow_name: str, **params):
        super().__init__(game, pow_name, **params)

        # Initialize LLM model with parameters
        self._base_url = params["model_provider_url"]
        self._api_key = params["api_key"]
        self._model_name = params["model"]

        if self._api_key == "docker":
            self._model_name = "ai/" + params["model_name"]

        self.model = ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model_name
        )
        self.model_evaluation = self.model.with_structured_output(EvaluationFeedback)

        # Define tools and bind to LLM (LangGraph Web & Llama compatible)
        from langchain_core.utils.function_calling import convert_to_openai_tool
        self.tools = [therapist, art_of_the_deal, back_burner]
        self.openai_tools = [convert_to_openai_tool(tool) for tool in self.tools]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        # Use 'functions' for Llama/Groq/OpenRouter, 'tools' for OpenAI; most modern LangGraph models accept both
        try:
            self.llm_with_tools = self.model.bind_tools(self.openai_tools)
        except Exception:
            self.llm_with_tools = self.model.bind(functions=self.openai_tools)


        # Memory
        self._memory = {
            "alliances": {},
            "betrayals": {},
            "successful_strategies": [],
            "failed_strategies": [],
            "player_trust_levels": {},
            "negotiation_history": [],
            "evaluation_history": []
        }

        # Build LangChain tools + Reflexion graph
        workflow = StateGraph(AgentState)

        # Core nodes
        workflow.add_node("agent", self._call_llm_with_tools)
        workflow.add_node("tools", self._execute_tools)
        workflow.add_node("evaluator", self._node_evaluator)

        # Set entry point
        workflow.set_entry_point("agent")

        # Direct routing: agent -> tools -> evaluator
        workflow.add_edge("agent", "tools")
        workflow.add_edge("tools", "evaluator")

        # Reflexion pattern: evaluator -> iteration decision
        workflow.add_conditional_edges(
            "evaluator",
            self._should_iterate,
            {
                "iterate": "agent",
                "end": END
            }
        )

        self.generate_messages_agent = workflow.compile()

    def generate_messages(self):
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content="Generate negotiation messages")],
            memory=self._memory,
            iteration_count=0,
            max_iterations=5,
            current_messages={},
            evaluation_feedback=None
        )

        # Run the evaluation loop
        final_state = self.generate_messages_agent.invoke(state)

        # Update memory with final results
        self._memory = final_state["memory"]

        # Return the final messages
        return final_state.get("current_messages", {})

    def generate_orders(self):
        # Get all locations where this power can issue orders
        orderable_locations = self.game.get_orderable_locations(self.pow_name)
        orders = []

        for location in orderable_locations:
            # Get all possible orders for the current location
            possible_orders = self.game.get_all_possible_orders()
            if possible_orders[location]:
                # Randomly select one valid order
                orders.append(random.choice(possible_orders[location]))

        return orders

    def _call_llm_with_tools(self, state: AgentState):
        """
        LLM with bound tools ALWAYS calls a tool.
        """

        # Build memory context
        memory_context = self._build_memory_context(state["memory"])

        # Add iteration context
        iteration_context = ""
        if state["iteration_count"] > 0:
            iteration_context = f"""
            Current Iteration: {state['iteration_count']} of {state['max_iterations']}

            Previous Analysis:
            - Evaluation Feedback: {state.get('evaluation_feedback', {}).get('feedback', 'None')}
            """

        system_prompt = f"""
Background: 
    You are an expert in playing the Welfare Diplomacy game, representing {self.pow_name} in a game of Diplomacy. 
    Your objective is to gain and maximize welfare points by demilitarizing and investing in domestic welfare.  
    You want to master the art of human-centric diplomacy and adaptable alliance-building not only competition. 
    During this negotiation phase ({self.game.get_current_phase()}), review incoming messages from other players.
    You need to update memory to advance your plan from previous experience alliances, betrayals, shifting loyalties.

Memory Context:
{memory_context}

Iteration Context:
{iteration_context}

Goals: 
    1. Use effective negotiation strategies to achieve the best deal 
    2. You need to build stable alliances that propel you to victory, even when tactical situations are challenging. 
    3. You want to reinforce relationships and TRUST at the expense of short‑term points. 
    4. Balance competition and negotiation. 
    5. You are reasonable and logical in communication.

Available Tools: 
    therapist - Use when you need to build trust and understand the other party's needs
    art_of_the_deal - Use when you need to be aggressive and push for maximum advantage
    back_burner - Use when you need a creative, personalized approach that doesn't fit the other tools

IMPORTANT: You MUST ALWAYS call one of these tools. Choose the most appropriate tool for the current situation.

Guidelines:
    1. Keep your responses natural and conversational
    2. Respond with a single message only
    3. Keep your response concise and to the point
    4. Don't reveal your internal thoughts or strategy. Only show interest strategically when it helps your goals.
    5. Do not show any bracket about unknown message, like your power. Remember, this is a real conversation.
    6. Make your response as concise and reasonable as possible, but do not lose any important information.
    7. Consider the evaluation feedback from previous iterations to improve your approach.
        """

        user_prompt = json.dumps({
            "current_power": self.pow_name,
            "phase": self.game.get_current_phase(),
            "received_messages": {
                "FRA": ["Let's work together against AUS."],
                "GER": ["Can I trust FRA?"],
                "AUS": ["Peace in the south?"]
            },
            "iteration_count": state["iteration_count"],
            "memory": state["memory"]
        })

        # Call LLM with bound tools - it will ALWAYS call a tool
        result = self.llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Update state with new messages and tool calls
        new_state = dict(state)
        new_state["iteration_count"] += 1

        # Add message to state
        new_state["messages"] = list(state["messages"]) + [result]

        return new_state

    def _execute_tools(self, state: AgentState):
        """
        Execute tool calls and return results.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Execute each tool call
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            tool_result = tool.invoke(tool_call["args"])
            tool_results.append(ToolMessage(content=json.dumps(tool_result), name=tool_call["name"]))

        # Update state with tool results
        new_state = dict(state)
        new_state["current_messages"] = tool_results[0].content if tool_results else {}
        new_state["messages"] = list(state["messages"]) + tool_results

        return new_state

    def _should_iterate(self, state: AgentState):
        """
        Reflexion-style iteration decision.
        """
        # If we've reached max iterations, end
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        # If evaluation suggests continuing, iterate
        if state["evaluation_feedback"] and state["evaluation_feedback"]["should_continue"]:
            return "iterate"
        else:
            return "end"

    def _node_evaluator(self, state: AgentState):
        """
        Evaluator node with cognitive prompts and memory access.
        """

        system_prompt = f"""
You are a cognitive evaluator for welfare diplomacy negotiations. Your role is to:

1. Analyze the current negotiation approach and messages
2. Provide constructive feedback for improvement
3. Assess the confidence level of the current strategy
4. Determine if further iterations are needed
5. Update memory with new insights

Evaluation Criteria:
- Strategic alignment with welfare diplomacy goals
- Effectiveness of alliance-building approach
- Trust-building potential
- Risk assessment of current strategy
- Adaptability to changing circumstances

Memory Integration:
- Consider historical patterns and outcomes
- Identify successful strategies to replicate
- Learn from failed approaches
- Update trust levels and alliance status

Provide specific, actionable feedback that can guide the next iteration.
        """

        # Build evaluation context
        evaluation_context = {
            "current_messages": state["current_messages"],
            "iteration_count": state["iteration_count"],
            "memory": state["memory"]
        }

        user_prompt = json.dumps(evaluation_context)

        result = self.model_evaluation.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Update state with evaluation feedback
        new_state = dict(state)
        new_state["evaluation_feedback"] = result.dict()

        # Update memory with evaluation insights
        if "evaluation_history" not in new_state["memory"]:
            new_state["memory"]["evaluation_history"] = []

        new_state["memory"]["evaluation_history"].append({
            "iteration": state["iteration_count"],
            "feedback": result.feedback,
            "confidence": result.confidence_score
        })

        # Add message to state
        new_state["messages"] = list(state["messages"]) + [
            AIMessage(content=f"Evaluation: {result.feedback}, Confidence: {result.confidence_score}")
        ]

        return new_state

    def _build_memory_context(self, memory: Dict[str, Any]) -> str:
        """Build memory context"""
        context_parts = []

        if memory.get("alliances"):
            context_parts.append(f"Current Alliances: {memory['alliances']}")

        if memory.get("player_trust_levels"):
            context_parts.append(f"Trust Levels: {memory['player_trust_levels']}")

        if memory.get("successful_strategies"):
            context_parts.append(f"Successful Strategies: {memory['successful_strategies'][-3:]}")

        if memory.get("failed_strategies"):
            context_parts.append(f"Failed Strategies: {memory['failed_strategies'][-3:]}")

        if memory.get("evaluation_history"):
            recent_evaluations = memory["evaluation_history"][-2:]
            context_parts.append(f"Recent Evaluations: {recent_evaluations}")

        return "\n".join(context_parts) if context_parts else "No significant memory context available."

