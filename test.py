from agent import CustomerAgent
import uuid
import json
from datetime import datetime

def stream_response_capture(agent, user_input, config, user_name=None):
    """Modified stream_response that captures output"""
    parsed = agent.parser.parse(user_input)
    context_query = agent._add_context(user_input, parsed)
    
    if user_name:
        context_query = f"Customer name: {user_name}\n{context_query}"
    
    responses = []
    tools_used = []
    
    for event in agent.agent.stream(
        {"messages": [{"role": "user", "content": context_query}]},
        stream_mode="values",
        config=config,
    ):
        if 'messages' in event:
            last_msg = event["messages"][-1]
            
            # Check for tool calls
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    tools_used.append(tool_call.get('name', 'unknown_tool'))
            
            # Check for AI responses
            if hasattr(last_msg, 'content') and last_msg.content:
                if hasattr(last_msg, 'type') and last_msg.type == 'ai':
                    responses.append(last_msg.content)
                elif not hasattr(last_msg, 'type') and not hasattr(last_msg, 'tool_calls'):
                    responses.append(last_msg.content)
    
    return {
        "parsed_intent": parsed.get('intent', 'unknown'),
        "confidence": parsed.get('confidence', 0.0),
        "entities": parsed.get('entities', {}),
        "agent_response": responses[-1] if responses else "No response generated",
        "tools_used": list(set(tools_used))  # Remove duplicates
    }

def run_all_scenarios():
    print("🚀 Initializing Customer Agent...")
    agent = CustomerAgent()
    
    scenarios = [
        {
            "name": "Order Tracking with Follow-up",
            "customer": "Ahmet Yılmaz",
            "steps": [
                "Merhaba, TY123456789 numaralı siparişim nerede?",
                "Kargo firması hangisi?", 
                "Teslimat tarihini değiştirebilir miyim?"
            ]
        },
        {
            "name": "Product Information Deep Dive",
            "customer": "Ayşe Demir",
            "steps": [
                "iPhone 15 Pro Max hakkında bilgi istiyorum",
                "256GB model stokta var mı?",
                "Hangi renkler mevcut?",
                "Garanti politikası nedir?"
            ]
        },
        {
            "name": "Return Process with Account Check",
            "customer": "Mehmet Kaya",
            "steps": [
                "Samsung telefonu iade etmek istiyorum",
                "Hesabımda hangi siparişler var?",
                "İade süreci nasıl işliyor?",
                "İade ne kadar sürer?"
            ]
        },
        {
            "name": "Stock and Product Search",
            "customer": "Zeynep Kaya",
            "steps": [
                "Nike ayakkabı var mı?",
                "Air Max 270 özellikle",
                "38 numara stokta mı?",
                "Benzer Nike modelleri neler?"
            ]
        },
        {
            "name": "Membership Benefits Inquiry",
            "customer": "Ali Şahin",
            "steps": [
                "Üyelik bilgilerimi öğrenebilir miyim?",
                "Gold üyeliğimin faydaları neler?",
                "Kargo ücretsiz mi benim için?"
            ]
        },
        {
            "name": "Payment Issue Resolution",
            "customer": "Fatma Arslan",
            "steps": [
                "Ödeme yaparken sorun yaşıyorum",
                "Kartımdan para çekildi ama sipariş yok",
                "İade ne zaman hesabıma yansır?"
            ]
        },
        {
            "name": "Delivery Problem Handling",
            "customer": "Emre Koç",
            "steps": [
                "TY789012345 kargom gecikti",
                "3 gündür evde kimse yok diyor",
                "Şubeden alabilirim",
                "Gecikme tazminatı var mı?"
            ]
        },
        {
            "name": "Product Recommendation Request",
            "customer": "Selin Yurt",
            "steps": [
                "Laptop önerisi lazım",
                "Yazılım geliştirme için",
                "Bütçem 50000 TL civarı",
                "Performans/fiyat en iyisi hangisi?"
            ]
        },
        {
            "name": "Account Access Problem",
            "customer": "Hasan Güler",
            "steps": [
                "Hesabıma giremiyorum",
                "Şifremi unuttum galiba",
                "E-postamı da değiştirdim",
                "Nasıl çözebiliriz?"
            ]
        },
        {
            "name": "Business/Bulk Order Inquiry",
            "customer": "Murat Öz",
            "steps": [
                "Toplu alım yapmak istiyorum",
                "50 adet tişört gerekiyor",
                "Toplu indirim var mı?",
                "Kurumsal hesap açabilir miyim?"
            ]
        }
    ]
    
    all_results = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(scenarios),
            "total_interactions": sum(len(s['steps']) for s in scenarios),
            "agent_version": "updated_with_rag"
        },
        "scenarios": []
    }
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"Customer: {scenario['customer']}")
        print(f"{'='*60}")
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        scenario_result = {
            "scenario_id": i+1,
            "name": scenario['name'],
            "customer": scenario['customer'],
            "interactions": [],
            "summary": {
                "total_steps": len(scenario['steps']),
                "errors": 0,
                "tools_used": set(),
                "intents_detected": set()
            }
        }
        
        for step_num, user_input in enumerate(scenario['steps']):
            print(f"\n--- Step {step_num + 1} ---")
            print(f"👤 {scenario['customer']}: {user_input}")
            
            try:
                result = stream_response_capture(agent, user_input, config, scenario['customer'])
                
                print(f"🤖 Ayla: {result['agent_response']}")
                print(f"📊 Intent: {result['parsed_intent']} (confidence: {result['confidence']:.2f})")
                if result['tools_used']:
                    print(f"🔧 Tools: {', '.join(result['tools_used'])}")
                if result['entities']:
                    print(f"📋 Entities: {result['entities']}")
                
                interaction = {
                    "step": step_num + 1,
                    "user_input": user_input,
                    "intent": result['parsed_intent'],
                    "confidence": result['confidence'],
                    "entities": result['entities'],
                    "agent_response": result['agent_response'],
                    "tools_used": result['tools_used'],
                    "response_length": len(result['agent_response'])
                }
                
                scenario_result["interactions"].append(interaction)
                
                # Update summary
                scenario_result["summary"]["tools_used"].update(result['tools_used'])
                scenario_result["summary"]["intents_detected"].add(result['parsed_intent'])
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                print(error_msg)
                
                scenario_result["interactions"].append({
                    "step": step_num + 1,
                    "user_input": user_input,
                    "error": str(e)
                })
                scenario_result["summary"]["errors"] += 1
        
        # Convert sets to lists for JSON serialization
        scenario_result["summary"]["tools_used"] = list(scenario_result["summary"]["tools_used"])
        scenario_result["summary"]["intents_detected"] = list(scenario_result["summary"]["intents_detected"])
        
        all_results["scenarios"].append(scenario_result)
    
    # Save detailed results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results, output_file)
    
    return all_results

def generate_summary_report(results, results_file):
    """Generate a human-readable summary report"""
    
    print(f"\n{'='*60}")
    print("📋 TEST EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    total_scenarios = results["test_metadata"]["total_scenarios"]
    total_interactions = results["test_metadata"]["total_interactions"]
    
    print(f"✅ Scenarios completed: {total_scenarios}")
    print(f"📊 Total interactions: {total_interactions}")
    
    # Error analysis
    total_errors = sum(s["summary"]["errors"] for s in results["scenarios"])
    if total_errors > 0:
        print(f"⚠️  Total errors: {total_errors}")
    else:
        print("✨ No errors encountered!")
    
    # Tool usage analysis
    all_tools = set()
    for scenario in results["scenarios"]:
        all_tools.update(scenario["summary"]["tools_used"])
    
    print(f"\n🔧 TOOLS UTILIZED ({len(all_tools)} total):")
    for tool in sorted(all_tools):
        usage_count = sum(1 for s in results["scenarios"] 
                         for i in s["interactions"] 
                         if tool in i.get("tools_used", []))
        print(f"  • {tool}: {usage_count} times")
    
    # Intent distribution
    all_intents = {}
    for scenario in results["scenarios"]:
        for interaction in scenario["interactions"]:
            if "intent" in interaction:
                intent = interaction["intent"]
                all_intents[intent] = all_intents.get(intent, 0) + 1
    
    print(f"\n🎯 INTENT DISTRIBUTION:")
    for intent, count in sorted(all_intents.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {intent}: {count} queries")
    
    # Performance metrics
    successful_interactions = total_interactions - total_errors
    success_rate = (successful_interactions / total_interactions) * 100 if total_interactions > 0 else 0
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  • Success rate: {success_rate:.1f}%")
    print(f"  • Average confidence: {calculate_avg_confidence(results):.2f}")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"{'='*60}")

def calculate_avg_confidence(results):
    """Calculate average confidence across all interactions"""
    confidences = []
    for scenario in results["scenarios"]:
        for interaction in scenario["interactions"]:
            if "confidence" in interaction and interaction["confidence"] > 0:
                confidences.append(interaction["confidence"])
    return sum(confidences) / len(confidences) if confidences else 0.0

if __name__ == "__main__":
    print("🤖 SBM Customer Agent Test Suite")
    print("=" * 40)
    results = run_all_scenarios()