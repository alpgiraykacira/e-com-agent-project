from agent import CustomerAgent

agent = CustomerAgent()
config = {"configurable": {"thread_id": "test"}}

def chat_with_agent(agent, customer_name="Müşteri"):
    """Interactive chat function for Jupyter notebook"""
    import uuid
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print(f"🤖 Merhaba {customer_name}! SBM müşteri hizmetlerine hoş geldiniz.")
    print("💬 Çıkmak için 'quit', 'exit' veya 'çıkış' yazın.\n")
    
    while True:
        try:
            user_input = input(f"👤 {customer_name}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çıkış', 'q']:
                print(f"🤖 İyi günler {customer_name}!")
                break
            
            if not user_input:
                continue
                
            result = agent.process(user_input, config)
            
            print(f"🤖 Ayla: {result['response']}")
            print(f"📊 [{result['analysis']['intent']} - {result['analysis']['confidence']:.2f}]\n")
            
        except KeyboardInterrupt:
            print(f"\n🤖 İyi günler {customer_name}!")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

# Start chatting:
chat_with_agent(agent, "Ahmet Yılmaz")