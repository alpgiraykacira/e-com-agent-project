from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import pandas as pd
from typing import List
import dotenv
from datetime import datetime

dotenv.load_dotenv()

class DataProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.docs = []
        self.retrievers = {}
        
    def load_data(self):
        # Load data from CSV files
        data_files = {
            'customers': 'data/customers.csv',
            'orders': 'data/orders.csv',
            'products': 'data/products.csv',
            'policies': 'data/policies.csv',
            'faqs': 'data/faqs.csv',
            'procedures': 'data/procedures.csv',
            'responses': 'data/responses.csv'
        }
        
        # Load each file and handle missing files gracefully
        loaded_data = {}
        for key, file_path in data_files.items():
            try:
                loaded_data[key] = pd.read_csv(file_path)
                print(f"✅ Loaded {key}: {len(loaded_data[key])} records")
            except FileNotFoundError:
                print(f"⚠️ Optional file not found: {file_path}")
                loaded_data[key] = None
        
        print("📊 Processing data...")
        
        # Process core data
        if loaded_data['customers'] is not None:
            self._process_customers(loaded_data['customers'])
        if loaded_data['products'] is not None:
            self._process_products(loaded_data['products'])
        if loaded_data['orders'] is not None:
            self._process_orders(loaded_data['orders'])
            
        # Process support data
        if loaded_data['policies'] is not None:
            self._process_policies(loaded_data['policies'])
        if loaded_data['faqs'] is not None:
            self._process_faqs(loaded_data['faqs'])
        if loaded_data['procedures'] is not None:
            self._process_procedures(loaded_data['procedures'])
        if loaded_data['responses'] is not None:
            self._process_responses(loaded_data['responses'])
        
        # Split documents into manageable chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # Combine all documents into a single list for splitting
        self.chunks = splitter.split_documents(self.docs)
        print(f"📄 Created {len(self.chunks)} document chunks")
        
        # Build retrievers for each document type
        self._build_retrievers()
        
    def _process_customers(self, df):
        for _, customer in df.iterrows():
            # Benefits based on membership level
            benefits = "Free shipping, priority support, extended returns" if customer['membership_level'] in ['Gold', 'Premium'] else "Standard services"
            
            # Format customer information
            content = f"""
            🧑‍💼 CUSTOMER INFO

            ID: {customer['customer_id']}
            Name: {customer['name']}
            Email: {customer['email']}
            Phone: {customer['phone']}
            Address: {customer['address']}
            Membership: {customer['membership_level']}

            Benefits: {benefits}
            Account Status: Active
            Registration: Verified
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "customer",
                    "customer_id": customer['customer_id'],
                    "membership": customer['membership_level'],
                    "search_terms": f"{customer['name']} {customer['email']} {customer['customer_id']}"
                }
            ))

    def _process_products(self, df):
        for _, product in df.iterrows():
            # Check stock and delivery status
            stock_status = "✅ In Stock" if product['stock_quantity'] > 0 else "❌ Out of Stock"
            delivery_info = "🚀 Fast delivery" if product['stock_quantity'] > 20 else "⚡ Limited stock"
            
            # Format product information
            content = f"""
            🛍️ PRODUCT INFO

            ID: {product['product_id']}
            Product: {product['title']}
            Brand: {product['brand']}
            Category: {product['category']}
            Price: ₺{product['price']:.2f}
            Stock: {product['stock_quantity']} units - {stock_status}

            Details:
            - Sizes: {product['sizes']}
            - Colors: {product['colors']}
            - Seller: {product['seller_name']}
            - Rating: ⭐ {product['rating']}/5 ({product['review_count']} reviews)
            - Warranty: 2 years manufacturer warranty
            - Return Period: 15 days for returns

            {product['description']}

            {delivery_info if product['stock_quantity'] > 0 else "🔔 Stock notification available"}
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "product",
                    "product_id": product['product_id'],
                    "category": product['category'],
                    "brand": product['brand'],
                    "price": float(product['price']),
                    "stock": int(product['stock_quantity']),
                    "search_terms": f"{product['title']} {product['brand']} {product['category']}"
                }
            ))

    def _process_orders(self, df):
        for _, order in df.iterrows():
            order_date = pd.to_datetime(order['order_date'])
            delivery_date = pd.to_datetime(order['estimated_delivery'])
            
            status_icons = {
                'Onaylandı': '✅', 'Hazırlanıyor': '📦', 'Kargoda': '🚚', 
                'Teslim Edildi': '🎉', 'Tamamlandı': '✨'
            }
            
            # Format order information
            content = f"""
            📦 ORDER INFO

            Order: {order['order_id']}
            Customer: {order['customer_id']}
            Product: {order['product_id']}
            Quantity: {order['quantity']}
            Total: ₺{order['total_price']:.2f}

            Status:
            {status_icons.get(order['order_status'], '📋')} Order: {order['order_status']}
            🚚 Shipping: {order['shipping_status']}
            🏷️ Tracking: {order['tracking_code']}

            Dates:
            📅 Ordered: {order_date.strftime('%d/%m/%Y')}
            🎯 Delivery: {delivery_date.strftime('%d/%m/%Y')}

            📍 Address: {order['shipping_address']}
            💳 Payment: {order['payment_method']} - {order['payment_status']}
            """

            days_left = (delivery_date - datetime.now()).days
            if order['shipping_status'] == 'Teslim Edildi':
                content += "\n🎉 Successfully delivered!"
            elif days_left < 0:
                content += "\n⚠️ Delivery date passed"
            else:
                content += f"\n⏰ Delivery in {days_left} days"

            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "order",
                    "order_id": order['order_id'],
                    "customer_id": order['customer_id'],
                    "status": order['order_status'],
                    "search_terms": f"{order['order_id']} {order['customer_id']} {order['tracking_code']}"
                }
            ))

    def _process_policies(self, df):
        for _, policy in df.iterrows():
            # Format policy information
            content = f"""
            📋 POLICY: {policy['title']}

            Category: {policy['category']}
            Effective Date: {policy.get('effective_date', 'Current')}

            {policy['content']}

            Last Updated: {policy.get('last_updated', 'Recent')}
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "policy",
                    "category": policy['category'],
                    "policy_type": policy.get('policy_type', 'general'),
                    "search_terms": f"{policy['title']} {policy['category']}"
                }
            ))

    def _process_faqs(self, df):
        for _, faq in df.iterrows():
            # Format FAQ information
            content = f"""
            ❓ FAQ: {faq['question']}

            Category: {faq['category']}
            Keywords: {faq.get('keywords', '')}

            ✅ Answer: {faq['answer']}

            Related Topics: {faq.get('related_topics', 'General')}
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "faq",
                    "category": faq['category'],
                    "keywords": faq.get('keywords', ''),
                    "search_terms": f"{faq['question']} {faq.get('keywords', '')}"
                }
            ))

    def _process_procedures(self, df):
        for _, procedure in df.iterrows():
            # Format procedure information
            content = f"""
            🔧 PROCEDURE: {procedure['title']}

            Type: {procedure['procedure_type']}
            Category: {procedure['category']}

            Steps:
            {procedure['steps']}

            Escalation: {procedure.get('escalation_rules', 'Contact supervisor if needed')}
            Estimated Time: {procedure.get('estimated_time', 'Varies')}
            Required Information: {procedure.get('required_info', 'Customer details')}
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "procedure",
                    "procedure_type": procedure['procedure_type'],
                    "category": procedure['category'],
                    "search_terms": f"{procedure['title']} {procedure['procedure_type']}"
                }
            ))

    def _process_responses(self, df):
        for _, response in df.iterrows():
            # Format response template
            content = f"""
            💬 RESPONSE TEMPLATE

            Intent: {response['intent']}
            Scenario: {response['scenario']}
            Tone: {response['tone']}

            Template:
            {response['response_template']}

            Usage Notes: {response.get('usage_notes', 'Standard customer service response')}
            Escalation Trigger: {response.get('escalation_trigger', 'Complex issues')}
            """
            self.docs.append(Document(
                page_content=content.strip(),
                metadata={
                    "type": "response",
                    "intent": response['intent'],
                    "scenario": response['scenario'],
                    "tone": response['tone'],
                    "search_terms": f"{response['intent']} {response['scenario']}"
                }
            ))

    def _build_retrievers(self):
        # Build retrievers for each document type
        doc_types = ['customer', 'product', 'order', 'policy', 'faq', 'procedure', 'response']
        
        for doc_type in doc_types:
            type_docs = [doc for doc in self.chunks if doc.metadata.get("type") == doc_type]
            if type_docs:
                store = InMemoryVectorStore.from_documents(type_docs, self.embeddings)
                base_retriever = store.as_retriever(search_kwargs={"k": 4})
                self.retrievers[doc_type] = self._wrap_retriever(base_retriever, doc_type)
    
    def _wrap_retriever(self, retriever, doc_type):
        # Create a smart retrieval function that expands the query
        def smart_retrieve(query: str) -> List[Document]:
            expanded = self._expand_query(query, doc_type)
            docs = retriever.invoke(expanded)
            return self._filter_results(docs, query)
        
        return RunnableLambda(smart_retrieve)

    def _expand_query(self, query: str, doc_type: str) -> str:
        # Expand the query with synonyms and related terms
        synonyms = {
            'customer': {'customer': ['user', 'account', 'müşteri'], 'membership': ['level', 'tier', 'üyelik']},
            'product': {'product': ['item', 'ürün'], 'price': ['cost', 'fee', 'fiyat'], 'stock': ['inventory', 'stok']},
            'order': {'order': ['purchase', 'sipariş'], 'shipping': ['delivery', 'kargo'], 'status': ['tracking', 'durum']},
            'policy': {'policy': ['rule', 'regulation', 'politika'], 'return': ['iade'], 'refund': ['geri ödeme']},
            'faq': {'question': ['soru'], 'help': ['yardım'], 'how': ['nasıl']},
            'procedure': {'process': ['süreç'], 'step': ['adım'], 'guide': ['rehber']},
            'response': {'template': ['şablon'], 'answer': ['cevap'], 'reply': ['yanıt']}
        }
        
        terms = [query]
        query_lower = query.lower()
        
        if doc_type in synonyms:
            for word, alternatives in synonyms[doc_type].items():
                if word in query_lower:
                    terms.extend(alternatives)
        
        return ' '.join(terms)

    def _filter_results(self, docs: List[Document], query: str) -> List[Document]:
        # Filter and score documents based on the query
        if not docs:
            return docs
            
        scored = []
        query_words = query.lower().split()
        
        for doc in docs:
            score = 0
            content = doc.page_content.lower()
            search_terms = doc.metadata.get('search_terms', '').lower()
            
            # Exact match bonus
            if query.lower() in content:
                score += 2
                
            # Word matches
            for word in query_words:
                if len(word) > 2:
                    if word in content:
                        score += 0.5
                    if word in search_terms:
                        score += 0.8
            
            if score > 0.3:
                scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:3]]


def get_tools():
    # Initialize the data processor and load data
    processor = DataProcessor()
    processor.load_data()
    
    tools = []
    
    # Core business tools
    core_tools = [
        ('customer', 'get_customer_info', 'Get customer information: name, contact, address, membership level'),
        ('product', 'get_product_info', 'Get product information: price, stock, features, description'),
        ('order', 'get_order_info', 'Get order information: status, shipping, delivery, tracking')
    ]
    
    # Support tools
    support_tools = [
        ('policy', 'get_policy_info', 'Get company policies: returns, shipping, privacy, terms'),
        ('faq', 'get_faq_info', 'Get frequently asked questions and answers'),
        ('procedure', 'get_procedure_info', 'Get step-by-step procedures for customer service processes'),
        ('response', 'get_response_template', 'Get standardized response templates for customer inquiries')
    ]
    
    # Create tools for available retrievers
    all_tools = core_tools + support_tools
    
    for retriever_type, tool_name, description in all_tools:
        if retriever_type in processor.retrievers:
            tool = create_retriever_tool(
                processor.retrievers[retriever_type],
                tool_name,
                description
            )
            tools.append(tool)
    
    print(f"✅ {len(tools)} tools ready: {[t.name for t in tools]}")
    return tools