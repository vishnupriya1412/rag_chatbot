# RAG Chatbot — Website (HTML/CSS/JS + Python backend)

Upload PDFs, Word docs, or text files through a real website, then ask
questions about them. Fully local — no API keys, nothing leaves your
machine.

## Project structure

```
rag_chatbot/
├── rag_core.py         # RAG logic: parsing, chunking, retrieval, generation
├── server.py           # Flask backend — serves the site + API endpoints
├── rag_chatbot.py       # optional: command-line version (still works)
├── requirements.txt
├── sample_docs/         # optional starter .txt files
└── static/
    ├── index.html        # page structure
    ├── style.css          # styling
    └── script.js          # upload + chat behavior
```

## How it works

1. You drop a `.txt`, `.pdf`, or `.docx` file onto the page
2. The browser sends it to the Flask backend (`/api/upload`)
3. The backend extracts the text, splits it into chunks, and embeds
   each chunk with a local model (`all-MiniLM-L6-v2`)
4. You ask a question in the chat box
5. The backend embeds your question, finds the most similar chunks
   (cosine similarity), and passes them + your question to a local
   language model (`flan-t5-base`) to generate a grounded answer
6. The answer — plus which document it came from — is sent back and
   shown in the chat

## Setup in VS Code

1. Open this folder in VS Code
2. Open the integrated terminal
3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Confirm it's active: `where python` (Windows) / `which python`
   (Mac/Linux) should point inside your project's `venv` folder.

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the website

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

The first run downloads the two local ML models (~330MB total) and
caches them — this can take a minute or two. After that, startup is
fast and no internet is needed.

## Using it

1. Drag a PDF, Word doc, or text file onto the upload box on the left
   (or click it to choose a file) — you can select multiple at once
2. Wait for it to appear in the "Indexed" list
3. Type a question in the box at the bottom and press Enter
4. The answer appears in the chat, with the source document named underneath

Uploaded files are saved to the `uploads/` folder (created automatically)
and stay indexed for as long as the server keeps running. Restarting
`server.py` clears the index — re-upload your documents after a restart.

## Customizing

- **Chunk size**: `chunk_size` / `overlap` in `chunk_text()` (`rag_core.py`)
- **Number of retrieved chunks**: `top_k` in `bot.ask()` calls
- **Generation model**: `flan-t5-base` is small and fast but not very
  powerful. For better answers, swap in `google/flan-t5-large` in
  `rag_core.py`'s `build_chatbot()` (needs more RAM)
- **More file types**: add a new branch to `extract_text()` in `rag_core.py`
- **Colors/fonts**: all in `static/style.css` under the `:root` block at the top
- **Port**: change `app.run(port=5000)` at the bottom of `server.py`

## Troubleshooting

- **"Unsupported file type" error**: only `.txt`, `.pdf`, and `.docx`
  are supported right now
- **PDF gives an empty/odd answer**: scanned/image-only PDFs have no
  extractable text — this project doesn't do OCR
- **`transformers` import errors**: make sure it's pinned below
  version 5.0 (already set in `requirements.txt`) — v5 removed the
  pipeline type this project uses for `flan-t5-base`
- **Nothing happens when you open the URL**: check the terminal — the
  server needs to finish loading models first ("Server ready." message)
