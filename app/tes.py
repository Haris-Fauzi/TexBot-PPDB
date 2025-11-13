import google.generativeai as genai

genai.configure(api_key="AIzaSyBRAQhHXrwkh9C9rXEHQ7vPoOPbIiTyhc0")

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Tuliskan satu kalimat perkenalan tentang PPDB SMK Texmaco Semarang.")
print(response.text)
