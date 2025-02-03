import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

async def main() -> None:
    # Get the API key from the environment variable or user input
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key  # Store for future use

    # Define the model client using OpenAI (using gpt-3.5-turbo)
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    # Define specialized agents with valid Python identifiers for names
    research_analyst = AssistantAgent(
        "ResearchAnalyst",
        system_message=(
            "You are a research analyst. Your job is to gather official reports and statistics on aviation incidents "
            "from credible sources such as the NTSB, FAA, and government agencies. Ensure that the information is accurate."
        ),
        model_client=model_client,
    )

    investigative_journalist = AssistantAgent(
        "InvestigativeJournalist",
        system_message=(
            "You are an investigative journalist. Your task is to explore alternative explanations and conspiracy theories "
            "surrounding aviation incidents. Differentiate between credible alternative views and mere speculation."
        ),
        model_client=model_client,
    )

    timeline_specialist = AssistantAgent(
        "TimelineSpecialist",
        system_message=(
            "You are a timeline specialist. Your role is to compile a clear and detailed chronology of events for the aviation incidents "
            "being investigated, ensuring that the sequence is logically structured."
        ),
        model_client=model_client,
    )

    aviation_safety_expert = AssistantAgent(
        "AviationSafetyExpert",
        system_message=(
            "You are an aviation safety expert. Analyze trends in aviation safety based on recent incidents. Identify any patterns or broader concerns "
            "that may indicate systemic issues in the industry."
        ),
        model_client=model_client,
    )

    editor_in_chief = AssistantAgent(
        "EditorInChief",
        system_message=(
            "You are the Editor-in-Chief. Your job is to review, refine, and fact-check the final report. Ensure the output is coherent, accurate, "
            "and well-structured before submission."
        ),
        model_client=model_client,
    )

    # Create the AutoGen team with the specialized agents
    team = MagenticOneGroupChat(
        [research_analyst, investigative_journalist, timeline_specialist, aviation_safety_expert, editor_in_chief],
        model_client=model_client
    )

    # Define the investigation task for the team
    task = (
        """# Task: Automated Profit Generation with AI Agents  

## **Objective:**  
Develop and deploy a multi-agent AI system that generates profit by **promoting Amazon affiliate products** through **automated content creation, social media engagement (X API), and short-form video marketing (JSON2VIDEO API)**.  

## **Agent Responsibilities:**  

### **1️⃣ Product Selection & Affiliate Link Integration**  
- Identify trending products on Amazon (user manually provides affiliate links).  
- Generate high-converting product descriptions and reviews.  
- Ensure all content includes **Amazon Associates affiliate links**.  

### **2️⃣ Automated Social Media Marketing (X API)**  
- Generate engaging **tweets with product links**.  
- Auto-post tweets at **optimal engagement times**.  
- Include **hashtags & trending keywords** for max visibility.  
- Auto-reply to user comments/questions.  
- Retweet & engage with relevant posts for organic reach.  

### **3️⃣ AI-Powered Video Marketing (JSON2VIDEO API)**  
- Convert **product descriptions into short videos**.  
- Create **TikTok, Instagram Reels, YouTube Shorts**.  
- Use engaging visuals, text overlays, and call-to-actions.  
- Auto-post videos to **X, TikTok, Instagram, YouTube**.  
- Track engagement & optimize future videos.  

### **4️⃣ Traffic Generation & Optimization**  
- Track engagement metrics (**clicks, views, conversions**).  
- Adjust content strategy based on **best-performing tweets & videos**.  
- Request user input if manual approvals are needed.  

## **Execution Plan (7-Day Launch Strategy)**  

✅ **Day 1:**  
- Set up **X API** & **JSON2VIDEO API** access.  
- Create **social media accounts** (if not already set).  
- Gather **affiliate product links**.  

✅ **Day 2:**  
- Generate **5-10 tweets** & schedule posting.  
- Create **3+ short videos** with JSON2VIDEO API.  
- Post videos to **TikTok, Instagram, YouTube Shorts**.  

✅ **Day 3-4:**  
- AI tracks **click-through rates & conversions**.  
- Engage with **tweets & comments** using X API.  
- Optimize **hashtags & posting times** based on results.  

✅ **Day 5-7:**  
- Scale content strategy (focus on **top-performing products**).  
- Automate **daily tweet & video generation**.  
- Optimize AI workflow to maximize **profitability**.  

## **Technical Requirements:**  
- **AI Agents:** Multi-Agent System (Magnetic One, AutoGen).  
- **Hosting:** AWS (recommended) or local.  
- **X API:** Automates tweet posting & engagement.  
- **JSON2VIDEO API:** Automates short-form video creation.  
- **Affiliate API:** Not required (manual links will be used).  

## **User Input Required:**  
- Approve selected affiliate products.  
- Provide API keys for **X API & JSON2VIDEO API**.  
- Review and approve content before posting (optional).  

## **Expected Outcome:**  
- Generate a minimum of **$500+ in commissions** within a week.  
- Leverage **AI-powered social media & video marketing** for ongoing passive income.  

**Run this on AWS for best scalability.**
"""
    )

    # Run the team and stream the output to the console
    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())
