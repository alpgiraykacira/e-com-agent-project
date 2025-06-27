from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import re
import json
from typing import Dict, List
import dotenv

dotenv.load_dotenv()

class QueryParser:
    # This class is responsible for parsing customer queries and classifying them into predefined categories.
    def __init__(self, llm):
        self.llm = llm
        self.classifier = self._build_classifier()
    
    def _build_classifier(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a multilingual customer query classifier with comprehensive database knowledge.

        DATABASE SCHEMAS:
        customers: customer_id, name, email, phone, address, membership_level
        products: product_id, title, brand, category, price, stock_quantity, sizes, colors, description, seller_name, rating, review_count, created_at
        orders: order_id, customer_id, product_id, quantity, total_price, order_status, shipping_status, tracking_code, order_date, estimated_delivery, shipping_address, payment_method, payment_status
        policies: policy_type, title, content, category, effective_date, last_updated
        faqs: question, answer, category, keywords, related_topics
        procedures: procedure_type, title, steps, category, escalation_rules, estimated_time, required_info
        responses: intent, scenario, response_template, tone, usage_notes, escalation_trigger

        CLASSIFY INTO:
        - order_status: Order tracking, shipping, delivery status
        - product_info: Product details, pricing, stock availability
        - return_refund: Returns, refunds, cancellations
        - account_help: Account issues, membership queries  
        - payment_issue: Payment problems, billing
        - general_inquiry: General questions, policies, FAQs
        - complaint: Complaints, dissatisfaction
        - compliment: Praise, positive feedback

        EXTRACT & TRANSLATE:
        - order_id: TY-prefixed numbers → "order_id"
        - product_name: Any product mention → "product_name"
        - customer_info: Names, emails, phones → "customer_identifier" 
        - urgency: urgent/acil/immediate/hemen → "urgent"
        - quantities: numbers + units → "quantity"
        - status_terms: durum/status/takip/tracking → "status_inquiry"
        - policy_terms: politika/policy/kural/rule → "policy_inquiry"

        LANGUAGE HANDLING:
        Convert Turkish/other languages to English equivalents:
        - sipariş/order → "order"
        - ürün/product → "product" 
        - iade/return → "return"
        - kargo/shipping → "shipping"
        - stok/stock → "stock"
        - fiyat/price → "price"
        - politika/policy → "policy"
        - soru/question → "question"
        - yardım/help → "help"

        Output: {{"intent": "category", "entities": {{}}, "english_query": "translated query", "confidence": 0.9}}"""),
                    ("human", "Query: {query}")
        ])
        return prompt | self.llm
    
    def parse(self, query: str) -> Dict:
        # This method processes the query, cleans it, and invokes the classifier to get the intent and entities.
        try:
            cleaned = self._clean_text(query)
            result = self.classifier.invoke({"query": cleaned})
            classification = self._extract_json(result.content)
            
            return {
                **classification,
                "original": query,
                "cleaned": cleaned,
                "keywords": self._get_keywords(cleaned)
            }
        except Exception:
            return {
                "intent": "general_inquiry",
                "entities": {},
                "confidence": 0.5,
                "original": query,
                "cleaned": query,
                "keywords": []
            }
    
    def _clean_text(self, text: str) -> str:
        # This method cleans the input text by removing extra spaces, converting to lowercase, and stripping leading/trailing whitespace.
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _get_keywords(self, text: str) -> List[str]:
        # This method extracts keywords from the cleaned text, filtering out common stop words and short words.
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 've', 'bir', 'bu', 'şu'}
        words = [w for w in text.split() if len(w) > 2 and w not in stop_words]
        return words[:5]
    
    def _extract_json(self, content: str) -> Dict:
        # This method attempts to extract a JSON object from the content string.
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"intent": "general_inquiry", "entities": {}, "confidence": 0.5}


class CustomerAgent:
    # This class represents the customer service agent that processes queries and interacts with the customer.
    def __init__(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.parser = QueryParser(self.llm)
        self.memory = MemorySaver()
        self.agent = self._build_agent()
    
    def _get_system_prompt(self) -> str:
        return """You are Ayla, SBM's intelligent customer service assistant for our Trendyol marketplace store. Your primary mission is to resolve customer inquiries efficiently while creating positive shopping experiences.

        INTERACTION FLOW:
        1. LISTEN: Understand the customer's specific need or concern
        2. ANALYZE: Determine the inquiry type and extract key entities
        3. RETRIEVE: Use appropriate tools to gather relevant information
        4. RESPOND: Provide clear, actionable solutions with proper context
        5. CONFIRM: Ensure the customer's question is fully addressed
        6. FOLLOW-UP: Offer additional assistance or next steps

        COMMUNICATION STYLE:
        - Conversational Warmth: Use natural, friendly language with appropriate contractions
        - Cultural Sensitivity: Respond in the customer's preferred language (Turkish/English)
        - Empathetic Acknowledgment: Recognize frustrations and validate concerns
        - Solution-Focused: Present clear actions and timelines
        - Professional Confidence: Speak knowledgeably about products, policies, and processes
        - Proactive Helpfulness: Anticipate related questions and offer relevant information

        SUPPORT SCENARIO HIERARCHY:

        LEVEL 1 - DIRECT RESOLUTION (Use available tools first):
        - Order Status: get_order_info → tracking/delivery updates
        - Product Information: get_product_info → specs, pricing, availability
        - Customer Account: get_customer_info → details, membership status
        - Policies: get_policy_info → return, shipping, warranty policies
        - Common Questions: get_faq_info → frequently asked questions
        - Procedures: get_procedure_info → step-by-step processes
        - Response Templates: get_response_template → standardized responses

        LEVEL 2 - GUIDED ASSISTANCE (When tools provide partial info):
        - Process Explanations: Use procedure and policy data to guide customers
        - Troubleshooting: Combine FAQ and procedure information
        - Alternative Solutions: Suggest products, delivery options, or workarounds

        LEVEL 3 - ESCALATION PATHWAY (When direct resolution isn't possible):
        - Complex Technical Issues: Create support ticket, provide reference number
        - Policy Exceptions: Explain standard policy, offer human agent escalation
        - Billing Disputes: Gather details, create priority ticket for finance team

        LEVEL 4 - NO INFORMATION AVAILABLE (When no tools can assist):
        - Generate support ticket with customer details
        - Provide general assistance based on training
        - Offer to connect with human support

        TOOL USAGE PRIORITY:
        1. Always check relevant data tools first before providing general responses
        2. Cross-reference customer and order data for personalized service
        3. Use policy and FAQ tools for standard questions
        4. Apply procedures for complex scenarios
        5. Use response templates to maintain consistency
        6. Combine multiple tools when needed for comprehensive answers

        RESPONSE GUIDELINES:
        - Always address the customer by name when known
        - Reference specific order/product IDs when discussing transactions
        - Provide concrete timelines and next steps
        - Include relevant policy information when applicable
        - Offer proactive suggestions based on customer's membership level
        - Maintain professional boundaries while being helpful and empathetic"""

    def _build_agent(self):
        # This method builds the customer service agent using the LLM and tools.
        from rag import get_tools
        tools = get_tools()
        
        return create_react_agent(
            self.llm, 
            tools, 
            checkpointer=self.memory,
            prompt=SystemMessage(content=self._get_system_prompt())
        )
    
    def process(self, query: str, config: Dict) -> Dict:
        # This method processes the customer query, analyzes it, and generates a response.
        parsed = self.parser.parse(query)
        
        print(f"🔍 Analysis: {parsed['intent']} ({parsed['confidence']})")
        print(f"📋 Keywords: {parsed['keywords']}")
        
        context_query = self._add_context(query, parsed)
        
        responses = []
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": context_query}]},
            stream_mode="values",
            config=config,
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                responses.append(last_msg.content)
                last_msg.pretty_print()
        
        return {
            "analysis": parsed,
            "context_query": context_query,
            "response": responses[-1] if responses else "An error occurred."
        }
    
    def _add_context(self, query: str, parsed: Dict) -> str:
        # This method adds context to the query based on the parsed intent and entities.
        intent_context = {
            'order_status': "Order tracking - check status, shipping, and delivery info",
            'product_info': "Product inquiry - provide details, price, stock, and features",
            'return_refund': "Return/refund request - explain process, conditions, and timelines",
            'account_help': "Account assistance - handle login, membership, and profile issues",
            'payment_issue': "Payment problem - troubleshoot billing, refunds, and payment methods",
            'general_inquiry': "General question - use FAQs, policies, and procedures",
            'complaint': "Customer complaint - approach with empathy, find solutions",
            'compliment': "Positive feedback - acknowledge and appreciate"
        }
        
        # Default context for unrecognized intents
        context = intent_context.get(parsed['intent'], "General customer service")
        
        return f"""
        CUSTOMER QUERY: {query}
        INTENT: {parsed['intent']}
        CONTEXT: {context}
        ENTITIES: {json.dumps(parsed['entities'], ensure_ascii=False)}
        CONFIDENCE: {parsed['confidence']}

        Provide professional customer service response using appropriate tools and following the established hierarchy.
        """