from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

youtube_id = 'Gfr50f6ZBvo'

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(video_id=youtube_id,languages=['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No caption available for this video")
    exit()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunk = splitter.create_documents([transcript])
# print(chunk)
