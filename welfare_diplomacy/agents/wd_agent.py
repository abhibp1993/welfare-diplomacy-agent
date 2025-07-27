import json
import random
from typing import Dict, List, Optional, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

import diplomacy
from welfare_diplomacy_agent.welfare_diplomacy.agents.base_agent import DiplomacyAgent
from IPython.display import Image, display

Powers = Literal["FRA", "ITA", "RUS", "ENG", "GER", "AUS", "TUR"]


class NegotiationMessage(BaseModel):
    messages_to_send: Dict[Powers, str] = Field(default_factory=dict)


class AgentState(BaseModel):
    current_power: str
    phase: str
    received_messages: Dict[str, List[str]] = Field(default_factory=dict)
    messages_to_send: Dict[Powers, str] = Field(default_factory=dict)
    # game_state_summary: Optional[str] = None


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
        self.model_message = self.model.with_structured_output(NegotiationMessage)

        # Initialize generate-messages agent
        graph = StateGraph(state_schema=AgentState)
        graph.add_node("chatbot", self._node_chatbot)
        # Tools
        graph.add_node("therapist", self._node_therapist)
        graph.add_node("art_of_the_deal", self._node_art_of_the_deal)
        graph.add_node("back_burner", self._node_back_burner)
        graph.add_node("evaluator", self._node_evaluator)
        
        # Define the flow
        graph.add_edge(START,"chatbot")
        graph.add_conditional_edges("chatbot", self.tool_selector)
        

        graph.add_edge("therapist", "evaluator")
        graph.add_edge("art_of_the_deal", "evaluator")
        graph.add_edge("back_burner", "evaluator")

        self.generate_messages_agent = graph.compile()

    def tool_selector(self, state: AgentState):
        """Selects the appropriate tool based on the state of the agent."""
        # Placeholder: always select therapist
        return
    
    def _node_evaluator(self, state: AgentState):
        """
        Evaluator node: provides feedback, accesses memory, and decides whether to continue or end.
        """
        # You can expand this logic to provide feedback, loop, or terminate as needed
        return state
    
    def generate_messages(self):
        state = AgentState(
            current_power=self.pow_name,
            phase=self.game.get_current_phase(),
            received_messages={
                "FRA": ["Let's work together against AUS."],
                "GER": ["Can I trust FRA?"],
                "AUS": ["Peace in the south?"]
            },
        )
        # game_state_summary="FRA is posturing as cooperative. GER is cautious. AUS is hedging.",
        msg = self.generate_messages_agent.invoke(state)
        print(msg)

        trial = {
            "ENGLAND": "Hi!",
            "FRANCE": "Hi!",
            "TURKEY": "Hi!",
            "GERMANY": "Hi!",
            "RUSSIA": "Hi!",
            "AUSTRIA": "Hi!",
            "ITALY": "Hi!",
        }
        del trial[self.pow_name]
        return trial

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

    def _node_chatbot(self, state: AgentState):
        """
        Node function to generate messages using the LLM model.
        """
        # <editor-fold desc="Description">
        system_prompt = f"""
Background: 
    You are an expert in playing the Welfare Diplomacy game, resenting {self.pow_name} in a game of Diplomacy. 
    Your objective is to gain and maximize welfare points by demilitarizing and investing in domestic welfare.  
    You want to master the art of human-centric diplomacy and adaptable alliance-building not only competition. 
    During this negotiation phase ({self.game.get_current_phase()}), review incoming messages from other players.
    You need to update memory to to advance your plan from previous experience alliances, betrayals, shifting loyalties.

Goals: 
    1. Use effective negotiation strategies to achieve the best deal 
    2. You need to build stable alliances that propel you to victory, even when tactical situations are challenging. 
    3. You want to reinforce relationships and TRUST at the expense of short‑term points. 
    4. Balance competition and negotiation. 
    5. You are reasonable and logical in communication.

Tools: 
    therapist 
    art of the deal 
    back burner  
    
Constraints: 
    The Back Burner tool should only be selected when neither of the other tools can deliver a viable path.

Guidelines:
    1. Keep your responses natural and conversational
    2. Respond with a single message only
    3. Keep your response concise and to the point
    4. Don't reveal your internal thoughts or strategy. Only show interest strategically when it helps your goals.
    5. Do not show any bracket about unknown message, like your power. Remembered, this is a the real conversation.
    6. Make your response as concise and reasonable as possible, but do not lose any important information.
        """

        user_prompt = json.dumps(state.dict())

        return self.model_message.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
    def _node_therapist(self, state: AgentState):
        """You are a...
        """
        
        system_prompt = f"""
You aren’t a negotiator, you’re a therapist. You’re not sure why they stuck you in this negotiation, but your goal is to make the other side feel like you understand them 100%.  
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

        user_prompt = json.dumps(state.dict())

        return self.model_message.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    def _node_art_of_the_deal(self, state: AgentState):
        """You are a...
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

        user_prompt = json.dumps(state.dict())

        return self.model_message.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    def _node_back_burner(self, state: AgentState):
        """
        Back Burner tool: advanced, creative, and self-reflective negotiation agent.
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
        user_prompt = json.dumps(state.dict())
        return self.model_message.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
