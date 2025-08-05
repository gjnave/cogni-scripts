<<<<<<< HEAD
from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch
import ebooklib
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import pymupdf as fitz
import pymupdf4llm
import json
import time

# Environment settings
IS_DUPLICATE = not os.getenv('SPACE_ID', '').startswith('hexgrad/')
CHAR_LIMIT = None if IS_DUPLICATE else 5000

# Model initialization
CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def normalize_text(text):
    """Normalize text by stripping whitespace and handling special characters."""
    return ' '.join(text.split()).strip()

def extract_chapters_from_epub(epub_file):
    book = epub.read_epub(epub_file)
    chapters = []

    def process_toc_entry(toc_entry, parent_title=None):
        # Handle (Section, subitems) tuples
        if isinstance(toc_entry, tuple) and len(toc_entry) == 2:
            section, subitems = toc_entry
            # Section can be a string or a Section object
            title = section.title if hasattr(section, 'title') else section
            display_title = f"{parent_title} - {title}" if parent_title else title
            # Recursively process subitems
            for subitem in subitems:
                process_toc_entry(subitem, parent_title=display_title)
        # Handle Link objects
        elif hasattr(toc_entry, 'title') and hasattr(toc_entry, 'href'):
            title = toc_entry.title or f"Chapter {len(chapters)+1}"
            href = toc_entry.href
            display_title = f"{parent_title} - {title}" if parent_title else title
            # Extract content if href is valid
            if href and isinstance(href, str):
                file_href = href.split('#')[0]
                item = book.get_item_with_href(file_href)
                if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    if '#' in href:
                        fragment = href.split('#')[1]
                        element = soup.find(id=fragment)
                        text = element.get_text(separator=' ') if element else soup.get_text(separator=' ')
                    else:
                        text = soup.get_text(separator=' ')
                    chapters.append({"title": display_title, "text": normalize_text(text)})
            else:
                print(f"Skipping entry with invalid href: {href}")
        else:
            print(f"Skipping unknown TOC entry: {toc_entry}")

    # Process the TOC
    print("TOC:", book.toc)  # Debug: see the TOC structure
    if book.toc:
        for toc_entry in book.toc:
            process_toc_entry(toc_entry)
        if chapters:
            return chapters

    # Fallback 1: Try using the spine if TOC fails
    if not chapters and book.spine:
        print("No chapters extracted from TOC, falling back to spine")
        for href_tuple in book.spine:
            href = href_tuple[0]
            item = book.get_item_with_href(href)
            if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                title_tag = soup.find('title')
                title = title_tag.get_text(strip=True) if title_tag and title_tag.get_text(strip=True) else item.get_name()
                text = soup.get_text(separator=' ')
                if text.strip(): # Only add chapters with content
                    chapters.append({"title": title, "text": normalize_text(text)})
        if chapters:
            return chapters

    # Fallback 2: extract all documents if TOC and spine fail
    print("No chapters extracted from TOC or spine, falling back to full document")
    full_text = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(separator=' ')
        full_text += text + " "
    
    if full_text.strip():
        return [{"title": "Full Document", "text": normalize_text(full_text)}]
    else:
        return []

class PdfParser:
    """Parser for extracting chapters from PDF files."""
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        self.doc = fitz.open(self.pdf_file)

    def _is_header_or_footer(self, block_bbox, page_bbox, margin_y=50):
        """Check if a text block is likely a header or footer based on vertical position."""
        _x0, y0, _x1, y1 = block_bbox
        page_height = page_bbox[3]
        # Check if the block is in the top or bottom margin
        if y1 < margin_y or y0 > page_height - margin_y:
            return True
        return False

    def _get_text_for_page_range(self, start_page, end_page):
        """Extracts clean text from a range of pages, removing headers/footers."""
        full_text = []
        for page_num in range(start_page, end_page):
            if page_num >= self.doc.page_count:
                break
            page = self.doc[page_num]
            page_bbox = page.rect
            blocks = page.get_text("blocks")
            for block in blocks:
                # block format: (x0, y0, x1, y1, "text", block_no, block_type)
                if not self._is_header_or_footer(block[:4], page_bbox):
                    full_text.append(block[4])
        return " ".join(full_text)

    def get_chapters(self):
        toc = self.doc.get_toc()
        chapters = []
        if toc:
            for i, (level, title, page) in enumerate(toc):
                # Determine the end page for the current chapter entry
                end_page = self.doc.page_count + 1 # Default to end of doc
                # Look for the next entry at the same or a higher level
                for next_level, _, next_page in toc[i+1:]:
                    if next_level <= level:
                        end_page = next_page
                        break
                
                # Get text for the page range. fitz pages are 0-indexed, toc is 1-indexed
                text = self._get_text_for_page_range(page - 1, end_page - 1)
                
                # Indent title to represent hierarchy
                display_title = "  " * (level - 1) + title
                chapters.append({'title': display_title, 'text': text})
            
            if chapters:
                return chapters

        # Fallback if TOC is missing or processing fails
        print("PDF TOC missing or failed, falling back to markdown conversion.")
        try:
            md_text = pymupdf4llm.to_markdown(self.pdf_file)
            chapters = [{'title': 'Full Document', 'text': md_text}]
        except Exception as e:
            print(f"Markdown conversion failed: {e}, falling back to raw text extraction.")
            full_text = self._get_text_for_page_range(0, self.doc.page_count)
            chapters = [{'title': 'Full Document', 'text': full_text}]
        
        return chapters if chapters else [{'title': 'Full Document', 'text': ''}]

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[min(len(ps)-1, 509)]
        try:
            audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.numpy()), ps
    return None, ''

def predict(text, voice='af_heart', speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, stream_all=False, document=None, chapter_title=None):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    # If stream_all is False, just stream the current chapter
    if not stream_all or not document or not chapter_title:
        text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
        return

    # If stream_all is True, stream the selected chapter and all subsequent chapters
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = data["chapters"]
    start_index = next(i for i, chapter in enumerate(chapters) if chapter["title"] == chapter_title)
    
    for chapter in chapters[start_index:]:
        text = normalize_text(chapter["text"]) if CHAR_LIMIT is None else normalize_text(chapter["text"].strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()

def process_file_with_timestamps(file, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = normalize_text(process_file(file.name))
    text = text if CHAR_LIMIT is None else text[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    chunks = []
    html = '<table><tr><th>Start</th><th>End</th><th>Text</th></tr>'
    for gs, ps, tks in pipeline(text, voice, speed, model=models[use_gpu]):
        ref_s = pack[min(len(ps)-1, 509)]
        audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
        chunks.append(audio.numpy())
        if tks and tks[0].start_ts is not None:
            for t in tks:
                if t.start_ts is None or t.end_ts is None:
                    continue
                html += f'<tr><td>{t.start_ts:.2f}</td><td>{t.end_ts:.2f}</td><td>{t.text}</td></tr>'
    html += '</table>'
    return chunks, html

def stream_file(file, voice, speed, use_gpu):
    chunks, html = process_file_with_timestamps(file, voice, speed, use_gpu)
    return chunks, html

def stream_audio_from_chunks(chunks, streaming_active):
    for rate, audio in chunks:
        if not streaming_active:
            break
        yield rate, audio

def stream_file_from_chunk(chunk_state, chunk_data):
    start_chunk = chunk_data['chunk'] if chunk_data and 'chunk' in chunk_data else 0
    for rate, audio in chunk_state[start_chunk:]:
        yield rate, audio

with open('en.txt', 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with open('gatsby5k.md', 'r') as r:
        return r.read().strip()

def get_frankenstein():
    with open('frankenstein5k.md', 'r') as r:
        return r.read().strip()

CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
'''

STREAM_NOTE = '\n\n'.join(['⚠️ Gradio bug might yield no audio on first Stream click'] +
                         ([f'✂️ Capped at {CHAR_LIMIT} characters'] if CHAR_LIMIT else []))

if not os.path.exists("processed_documents"):
    os.makedirs("processed_documents")

def process_file(file):
    """Process uploaded files (EPUB, PDF, TXT) and return chapters or an error dict."""
    try:
        file_path = file.name if hasattr(file, 'name') else file
        if file_path.endswith('.epub'):
            chapters = extract_chapters_from_epub(file_path)
        elif file_path.endswith('.pdf'):
            parser = PdfParser(file_path)
            chapters = parser.get_chapters()
            # Normalize text for all chapters post-extraction
            for chapter in chapters:
                chapter['text'] = normalize_text(chapter['text'])
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            normalized_text = normalize_text(text)
            chapters = [{'title': 'Full Document', 'text': normalized_text}]
        return chapters
    except Exception as e:
        print(f"Error processing file {getattr(file, 'name', 'file')}: {e}")
        # Return an error structure that can be handled by the caller
        return {"error": f"Failed to process file. It may be corrupt or in an unsupported format. Details: {e}"}

def generate_unique_name(original_name):
    base_name = os.path.splitext(os.path.basename(original_name))[0]
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.json"

def save_chapters(chapters, unique_name):
    data = {"document_name": unique_name, "chapters": chapters}
    with open(os.path.join("processed_documents", unique_name), "w") as f:
        json.dump(data, f)

def get_documents():
    return [f for f in os.listdir("processed_documents") if f.endswith('.json')]

def get_chapters(document):
    if not document:
        return gr.update(choices=[], value=None)
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = [chapter["title"] for chapter in data["chapters"]]
    return gr.update(choices=chapters, value=chapters[0] if chapters else None)

def load_chapter_text(document, chapter_title):
    if not document or not chapter_title:
        return ""
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    for chapter in data["chapters"]:
        if chapter["title"] == chapter_title:
            return chapter["text"]
    return ""

def process_and_save(file):
    if file is None:
        return gr.update(), gr.update(), "Please upload a file."
    
    result = process_file(file)
    
    # Check if processing returned an error
    if isinstance(result, dict) and 'error' in result:
        error_message = result['error']
        return gr.update(), gr.update(), f"Error: {error_message}"

    chapters = result
    if not chapters:
        return gr.update(), gr.update(), "Could not extract any chapters from the document."

    unique_name = generate_unique_name(file.name)
    save_chapters(chapters, unique_name)
    documents = get_documents()
    chapter_titles = [chapter["title"] for chapter in chapters]
    return (
        gr.update(choices=documents, value=unique_name),
        gr.update(choices=chapter_titles, value=chapter_titles[0] if chapter_titles else None),
        f"Document '{os.path.basename(unique_name)}' processed and saved."
    )
    
    
def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, streaming_active=True, stream_all=False, document=None, chapter_title=None):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    # If stream_all is False, just stream the current chapter (original behavior)
    if not stream_all or not document or not chapter_title:
        text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            if not streaming_active:
                break
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
        return

    # If stream_all is True, stream the selected chapter and all subsequent chapters
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = data["chapters"]
    # Find the index of the selected chapter
    try:
        start_index = next(i for i, chapter in enumerate(chapters) if chapter["title"] == chapter_title)
    except StopIteration:
        print(f"Chapter '{chapter_title}' not found in document '{document}'.")
        return

    # Stream each chapter from the selected one onward
    for chapter in chapters[start_index:]:
        if not streaming_active:
            break
        text = normalize_text(chapter["text"]) if CHAR_LIMIT is None else normalize_text(chapter["text"].strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            if not streaming_active:
                break
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
                
                

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False)
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
        stream_file_btn = gr.Button('Stream File', variant='primary')  # Ensure this is defined here
        process_btn = gr.Button('Process and Save Chapters', variant='secondary')
    with gr.Row():
        file_upload = gr.File(label="Upload EPUB/PDF/TXT")
    gr.Markdown("Upload a file and click 'Process and Save Chapters' to extract and save chapters for later use.")
    status = gr.Textbox(label="Status", interactive=False)
    file_viewer = gr.HTML(label="File Viewer")
    chunk_state = gr.State(value=[])
    streaming_active = gr.State(value=False)
    gr.HTML("""
    <script>
    function stopAudio() {
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            audio.pause();
            audio.currentTime = 0;
        });
    }
    window.addEventListener('message', (event) => {
        if (event.data.chunk !== undefined) {
            document.getElementById('chunk_input').value = JSON.stringify({chunk: event.data.chunk});
            document.getElementById('chunk_trigger').click();
        }
    });
    </script>
    """)
    chunk_input = gr.JSON(visible=False, elem_id="chunk_input")
    chunk_trigger = gr.Button("Trigger Chunk", visible=False, elem_id="chunk_trigger")

BANNER_TEXT = '''
# Kokoro-Plus [(getgoingfast.pro)](https://www.getgoingfast.pro)

[***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)  

[Listen to good music!](https://music.youtube.com/channel/UCY658vbL6S2zlRomNHoX54Q)
'''

API_OPEN = os.getenv('SPACE_ID') != 'hexgrad/Kokoro-TTS'



with gr.Blocks() as app:
    gr.Markdown(BANNER_TEXT)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Load Pre-processed Document")
            with gr.Row():
                document_dropdown = gr.Dropdown(
                    label="Select Document",
                    choices=[""] + get_documents(),  # Blank default option
                    value=""  # Pre-select the blank option
                )
                chapter_dropdown = gr.Dropdown(
                    label="Select Chapter",
                    choices=[],
                    value=None
                )
            text = gr.Textbox(label='Input Text')
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice')
                use_gpu = gr.Dropdown([('ZeroGPU 🚀', True), ('CPU 🐌', False)], value=CUDA_AVAILABLE, label='Hardware')
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            stream_all_checkbox = gr.Checkbox(label="Stream All Chapters", value=False)
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
            frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])

    # Event handlers
    document_dropdown.change(fn=get_chapters, inputs=[document_dropdown], outputs=[chapter_dropdown])
    chapter_dropdown.change(fn=load_chapter_text, inputs=[document_dropdown, chapter_dropdown], outputs=[text])
    process_btn.click(fn=process_and_save, inputs=[file_upload], outputs=[document_dropdown, chapter_dropdown, status])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])

    stream_event = stream_btn.click(
        fn=lambda: (True, None),
        outputs=[streaming_active, out_stream]
    ).then(
        fn=generate_all,
        inputs=[text, voice, speed, use_gpu, streaming_active, stream_all_checkbox, document_dropdown, chapter_dropdown],
        outputs=[out_stream]
    )

    file_stream_event = stream_file_btn.click(
        fn=lambda: (True, None),
        outputs=[streaming_active, out_stream]
    ).then(
        fn=stream_file,
        inputs=[file_upload, voice, speed, use_gpu],
        outputs=[chunk_state, file_viewer]
    ).then(
        fn=stream_audio_from_chunks,
        inputs=[chunk_state, streaming_active],
        outputs=[out_stream]
    )

    stop_btn.click(
        fn=lambda: False,
        outputs=[streaming_active],
        js="stopAudio",
        cancels=[stream_event]
    )

    chunk_trigger.click(fn=stream_file_from_chunk, inputs=[chunk_state, chunk_input], outputs=[out_stream])

if __name__ == '__main__':
    app.queue(api_open=API_OPEN).launch(share=API_OPEN)
=======
@echo off
:: Check if the script is run as Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges.
    echo Please run this script as an administrator.
    pause
    exit /b
)

cd /d %~dp0
IF EXIST "disclaimer.md" (
    TYPE "disclaimer.md"
    pause
)

IF EXIST "type about.nfo" TYPE type about.nfo

echo.
:: Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not installed or not found in PATH.
    echo Please install Anaconda/Miniconda and ensure it's added to PATH.
    pause
    exit /b
)

:: Check if conda environment 'kokoro' exists
call conda env list | findstr /C:"kokoro " >nul
if %errorlevel% equ 0 (
    echo Conda environment 'kokoro' already exists.
    set /p replace_env="Do you want to replace it? (y/n): "
    if /i "!replace_env!"=="y" (
        echo Removing existing 'kokoro' environment...
        call conda deactivate
        call conda env remove --name kokoro
        echo Creating new 'kokoro' environment...
        call conda create --name kokoro python=3.12 -y
    ) else (
        echo Using existing 'kokoro' environment...
    )
) else (
    echo Creating new 'kokoro' environment...
    call conda create --name kokoro python=3.12 -y
)
call conda activate kokoro
git clone https://github.com/gjnave/kokoro-tts-plus
cd kokoro-tts-plus
git config --system --add safe.directory "kokoro-tts-plus"
cd kokoro-tts-plus
curl -LO https://raw.githubusercontent.com/gjnave/cogni-scripts/refs/heads/main/koko/app.py
curl -LO https://raw.githubusercontent.com/gjnave/cogni-scripts/refs/heads/main/koko/en.txt
curl -LO https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
curl -LO https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
REM call conda install nvidia/label/cuda-12.6.3::cuda-toolkit -y
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp312-cp312-win_amd64.whl
pip install sageattention
pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"

pip install kokoro
pip install ebooklib
pip install PyMuPDF
pip install pymupdf4llm
pip install beautifulsoup4
pip install gradio
echo Installation Complete...
pause
>>>>>>> 8caec683981b015786334bf6e46e0aa1c8110920
