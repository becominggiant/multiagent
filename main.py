import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

async def main() -> None:
  # Get the API key from the environment variable or user input
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key  # Set it for future use
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Can you research into Aether and find out what studies have proven or disproven this and come up with a report to tell me if it is real or not"))

async def main() -> None:
    # ... (your existing API key handling code)
    # Get the API key from the environment variable or user input
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key  # Set it for future use
    # Define model_client within the main function
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    # Create agents for the team
    planner = AssistantAgent(
        "Planner",
        system_message="You are the project planner.  Outline the tasks and assign them to team members.",
        model_client=model_client,
    )
    writer = AssistantAgent(
        "Writer",
        system_message="You are the content writer. Focus on creating clear and concise text.",
        model_client=model_client,
    )
    reviewer = AssistantAgent(
        "Reviewer",
        system_message="You are the reviewer.  Check the quality and accuracy of the work.",
        model_client=model_client,
    )


    team = MagenticOneGroupChat([planner, writer, reviewer], model_client=model_client)

    # Define the task for the team
    task = """Personalized Email Automation System with AI-Powered Response Generation
Overview
We are seeking a skilled developer to build a personalized email automation system that replicates a user’s writing style using AI. This system will integrate with Gmail, process email data, fine-tune a language model, and generate contextually appropriate responses while maintaining conversation threads.

Project Scope
1. Gmail API Integration
Set up and manage Gmail API authentication and credentials.
Develop Python scripts to extract emails based on query parameters (e.g., date range, sent/received status).
2. Email Data Processing
Group emails into conversation threads in chronological order.
Extract and clean email content by removing signatures, disclaimers, and unnecessary characters.
Standardize text formatting and extract relevant metadata.
Organize processed email data into a structured format suitable for model training.
3. Model Development & Fine-Tuning
Prepare a training dataset from processed emails.
Select and fine-tune a base language model to replicate the user’s writing style.
Ensure the model generates accurate and contextually appropriate responses.
Optimize for minimal computational overhead given the small dataset (<5GB).
4. Email Automation System
Develop a service to monitor incoming emails in real-time.
Implement a classification system to determine email importance and type.
Use the fine-tuned model to generate response drafts.
Implement a confidence-based system to queue responses for review or auto-send when appropriate.
5. System Integration & Deployment
Set up Gmail API authentication for sending emails.
Implement email sending functionality, including attachments and formatting.
Maintain conversation context through message threading.
Develop a user-friendly dashboard to monitor system performance, response accuracy, and automated actions.
Provide manual override options for user intervention when necessary.
Ideal Candidate Qualifications
Strong experience with Python and NLP (e.g., OpenAI’s GPT, Hugging Face, fine-tuning LLMs).
Familiarity with Gmail API and OAuth authentication.
Experience in text processing, data structuring, and cleaning.
Knowledge of AI-based text generation and automation workflows.
Ability to build and deploy scalable automation solutions."""

    # Run the team and stream the output
    await Console(team.run_stream(task=task))
    
if __name__ == "__main__":
    asyncio.run(main())  # Properly handles the event loop

