from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional, Annotated
from dotenv import load_dotenv
from yt_dlp import YoutubeDL
from transcript_loader import youtube_id
from main import
load_dotenv()

class VideoTopic(BaseModel):
    main_topics: Annotated[Optional[list[str]], Field(default=None, description="Primary topics discussed in the video")]

    related_concepts: Annotated[Optional[list[str]] , Field(description="Concepts closely related to the video topics")]

    technologies: Annotated[Optional[list[str]], Field(description="Frameworks, tools, libraries, platforms")]

    people: Annotated[Optional[list[str]], Field(description="Important people mentioned")]

    research_areas: Annotated[Optional[list[str]] , Field(description="Academic or scientific domains discussed")]




#chunk related to the topic

url = f"https://www.youtube.com/watch?v={youtube_id}"

with YoutubeDL({}) as ydl:
    info = ydl.extract_info(url, download=False)



#model

model = ChatGroq(model='llama-3.3-70b-versatile',temperature= 0.1,max_tokens=500)

#output parser
parser = PydanticOutputParser(pydantic_object=VideoTopic)


#prompt template

prompt = PromptTemplate(
    template="""
                The following is a YouTube video description and the transcription related to the User Query.

                Remove:
                    - Sponsorships
                    - Affiliate links
                    - Social media links
                    - Podcast information
                    - Patreon/support information

                Keep:
                    - Information about the guest
                    - Information about the topic
                    - Episode outline/timestamps

                Description: \n
                {description}\n
                
                Transcript:\n
                {transcript}\n

                Return only the relevant content.\n
                {format_instruction}
""",
    input_variables=['description','transcript'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = prompt | model | parser

print(chain.invoke({'description': info['description']}))