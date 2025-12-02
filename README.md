# Kerala Ayurveda -- Internal Q&A & Content Generation System

*A practical design document for building a RAG-based assistant and article drafting workflow*

---

## Part A -- Designing the RAG System

### My Approach to Chunking the Corpus

After going through the content pack, I noticed the documents fall into a few distinct categories, and each needs slightly different handling:

**For the foundational docs** (`ayurveda_foundations.md`, `dosha_guide_vata_pitta_kapha.md`):

These are longer explainer pieces. I'd split them by their natural heading structure (`##` and `###` breaks), aiming for roughly 200--350 tokens per chunk. Where a paragraph sits awkwardly between two headings, I'd add a small overlap (maybe 20--40 tokens) so we don't lose context mid-thought. The key is to keep policy-type content (like "How we position Ayurveda" or the content boundaries section) together in one piece---those shouldn't be fragmented.

**For the FAQ** (`faq_general_ayurveda_patients.md`):

This one's straightforward. Each Q&A pair becomes its own chunk. They're already self-contained and compact, so no overlap needed. I'd tag each with a `faq_id` and the question text itself---helpful for matching user questions that are phrased similarly.

**For the product dossiers** (Ashwagandha, Brahmi Tailam, Triphala):

These have a consistent internal structure: Basic Info, Traditional Positioning, Key Messages, Safety & Precautions, Internal Tags. I'd chunk by those sections since they're already well-sized (150--250 tokens each). The important bit is tagging each chunk with the product name and section type---so when someone asks specifically about Ashwagandha safety, we can retrieve exactly that section.

**For the product catalog CSV**:

I'd treat each row as a mini-document. Mostly useful for routing---if someone types "Ashwagandha tablets," we can map that to the right dossier file and prioritize those chunks.

### Why Hybrid Retrieval Makes Sense Here

I'd start with a hybrid approach combining BM25 (keyword matching) and embedding-based search. Here's my thinking:

- **BM25 handles the obvious stuff well**---when someone asks about "Triphala" or "dosha," exact keyword matches are reliable and fast.
- **Embeddings help with fuzzier queries**---things like "evening wind-down routine" or "feeling wired but tired" where the exact words might not appear in the docs.

I'd weight them something like 60% embeddings, 40% BM25, but honestly that's a starting point to tune based on what queries actually come in.

**How many chunks to pull?** I'm thinking 6--8 by default. If we detect a specific product name in the query (via the catalog), I'd force-include 3--4 chunks from that product's dossier (positioning, messages, safety), then fill the remaining slots with general content from the foundations and FAQ. Also worth limiting to 2--3 chunks max per document so we don't end up with answers that only draw from one source.

### Passing Context to the LLM

The prompt structure I have in mind:

1.  **System message** with Kerala Ayurveda's brand rules and medical safety guardrails (pulled from `ayurveda_foundations.md`).
2.  **Context block** with numbered sources like `[Source 1] filename / section title`.
3.  **The user's question**.
4.  **Instructions** telling the model to only use the provided context, cite sources inline, and keep answers concise.

For citations, I'd have the LLM append `[Source 1]`, `[Source 2]` etc. after sentences that use specific chunks, then post-process to map those numbers back to actual document references.

### Safety Layer

This is important given the medical-adjacent nature of Ayurveda content. I'd add a post-processing check that scans for:

- Disease claims (words like "cure," "treat," "prevent," "diagnose")
- Dosage instructions ("take 2 tablets," specific dosing)
- Benefits not present in the retrieved text

If something triggers, either regenerate with stricter instructions or fall back to a safe response like "I don't have enough information to answer that---please consult a practitioner."

---

### The Core Function (Pseudo-code)

Here's a sketch of how the main pieces fit together:

```python
from typing import List, Dict, Any, TypedDict
import re

class Citation(TypedDict):
    doc_id: str
    section_id: str

class AnswerResult(TypedDict):
    answer: str
    citations: List[Citation]

def retrieve_chunks(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Runs hybrid retrieval over the Kerala Ayurveda corpus.
    Returns chunks like:
    {
      "doc_id": "product_ashwagandha_tablets_internal.md",
      "section_id": "traditional_positioning",
      "title": "Traditional Positioning",
      "text": "...",
      "score": 0.83
    }
    """
    # 1. Check if query mentions a known product (via catalog lookup)
    # 2. Run BM25 search for keyword candidates
    # 3. Run embedding search for semantic matches
    # 4. Merge results with weighted scoring
    # 5. Apply diversity limits (max 2-3 per doc) and return top-k
    pass

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    system_msg = """
You are an internal assistant for Kerala Ayurveda.

Ground rules:
- Use ONLY the provided context to answer. If it's not there, say you don't know.
- Never claim to diagnose, treat, cure, or prevent diseases.
- No dosing instructions.
- Use gentle language: "traditionally used to support...", "may help maintain..."
- When in doubt, encourage consulting a healthcare provider.
"""
    # Build the context block with numbered sources
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk['doc_id']} / {chunk['section_id']}\n{chunk['text']}"
        )

    return f"""{system_msg}

CONTEXT:
{chr(10).join(context_parts)}

USER QUESTION: {query}

Answer using only the context above. Add [Source N] citations after relevant sentences. Keep it concise (2-5 paragraphs).
"""

def safety_filter(answer: str) -> str:
    """Quick check for disallowed language patterns."""
    red_flags = [
        r"\b(cure[s]?|treat[s]?|prevent[s]?|diagnose[s]?)\b",
        r"\bdose\b", r"\bdosage\b", r"\btake \d+",
    ]
    for pattern in red_flags:
        if re.search(pattern, answer, re.IGNORECASE):
            answer += "\n\n_Note: Please verify this with a healthcare professional._"
            break
    return answer

def extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> List[Citation]:
    """Parse [Source N] markers and map back to doc references."""
    citations = []
    seen = set()
    for match in re.finditer(r"\[Source (\d+)\]", answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks) and idx not in seen:
            citations.append({
                "doc_id": chunks[idx]["doc_id"],
                "section_id": chunks[idx]["section_id"]
            })
            seen.add(idx)
    return citations

def answer_user_query(query: str) -> AnswerResult:
    chunks = retrieve_chunks(query, k=8)
    prompt = build_prompt(query, chunks)
    raw_answer = call_llm(prompt)  # Your LLM API call here
    safe_answer = safety_filter(raw_answer)
    citations = extract_citations(safe_answer, chunks)
    return {"answer": safe_answer, "citations": citations}
```

---

### Testing It Out: Example Queries

I picked three questions that feel like realistic internal use cases and traced through what I'd expect the system to do:

#### Example 1: "What are the key benefits of Ashwagandha tablets, and who are they for?"

**What should get retrieved:**
- The Ashwagandha dossier's "Traditional Positioning" section (stress adaptation, calmness, sleep support)
- The "Key Messages" section (stress resilience framing, target audience)
- The "Safety" section (pregnancy, allergies, thyroid conditions)
- Probably FAQ Q3 about stress and sleep (mentions Ashwagandha)
- Maybe the general positioning section from `ayurveda_foundations.md`

**What a good answer looks like:**
> The Ashwagandha Stress Balance Tablets draw on the traditional Ayurvedic use of Ashwagandha root to support the body's natural ability to adapt to stress, promote calmness, and help maintain restful sleep. [Source 1]
>
> They're positioned as daily support for stress resilience---think of them as help for winding down, not a sedative or instant fix. They tend to resonate with people who have demanding schedules, feel "wired and tired," or struggle with occasional restlessness at night. [Source 2]
>
> That said, they're not a substitute for professional care. Anyone who's pregnant, has thyroid or autoimmune issues, or takes ongoing medication should check with their doctor first. [Source 3]

**Where this could go wrong:**
The model might add generic claims like "supports immunity" or "helps with blood sugar" that sound plausible but aren't in our dossier. Mitigation: require that any specific benefit claim has a matching phrase in the retrieved text, or flag it for review.

---

#### Example 2: "Any precautions I should know about for Brahmi Tailam?"

**What should get retrieved:**
- Brahmi Tailam's "Safety & Precautions" section (external use only, avoid eyes, patch test)
- The "Traditional Positioning" for context (head oil, Vata/Pitta patterns)
- General safety language from `ayurveda_foundations.md`

**What a good answer looks like:**
> Brahmi Tailam is meant for external use only---don't ingest it, and keep it away from your eyes. [Source 1]
>
> If you have sensitive or allergy-prone skin, do a patch test on a small area first. And if you have a diagnosed scalp condition, it's better to follow your dermatologist's guidance rather than self-treating. [Source 1]
>
> Like other Ayurvedic products, this is positioned as part of a self-care routine, not a medical treatment. Anyone with ongoing health concerns should loop in their healthcare provider. [Source 2]

**Where this could go wrong:**
The model might invent extra contraindications ("not safe during pregnancy," "avoid in children") that sound reasonable but aren't actually in the document. We'd need to make sure population-specific warnings only appear if they're explicitly stated in the source material.

---

#### Example 3: "How does Triphala support digestion in Ayurveda?"

**What should get retrieved:**
- Triphala dossier's "Traditional Positioning" (digestive comfort, elimination, gentle cleansing)
- The "Key Messages" section (gentle/long-term, whole-system view, suitable use cases)
- Safety section
- General Ayurveda positioning on digestion from foundations doc

**What a good answer looks like:**
> Triphala is a classic Ayurvedic formula made from three fruits, traditionally used to support digestive comfort, regular elimination, and gentle internal cleansing. [Source 1]
>
> The way we talk about it emphasizes that it's a mild, long-term support---not a harsh "detox" or quick fix. In Ayurveda, digestion is seen as foundational to overall wellbeing, so supporting it is considered pretty central. [Source 1][Source 2]
>
> It tends to work well for people dealing with occasional post-meal heaviness, irregular habits, or mild digestive discomfort who want something gentle. As always, it's not meant to diagnose or treat any condition. [Source 3]

**Where this could go wrong:**
Easy to slip into medical language here---"treats constipation," "helps IBS," or overhyped detox/weight-loss claims. The dossier explicitly avoids these, so we'd need banned-phrase scanning and possibly negative examples in the prompt to keep things grounded.

---

## Part B -- Agentic Workflow for Article Generation

Now for the more ambitious piece: a system that takes a content brief, generates an article draft, fact-checks it against our corpus, and polishes the tone---all before handing it to a human editor.

### The Pipeline I'd Build

I'm envisioning 5 steps, each with a clear job:

---

**Step 1: Brief → Outline**

*What it does:* Takes a short brief (topic, audience, target length, products to feature) and produces a structured outline plus some retrieval queries to seed the next step.

*Example input:*

```json
{
  "brief_id": "BRF-001",
  "topic": "Evening routine for stress relief with Ashwagandha and Brahmi Tailam",
  "audience": "busy professionals",
  "length": "800-1000 words",
  "products": ["Ashwagandha Stress Balance Tablets", "Brahmi Tailam"]
}
```

*Example output:*

```json
{
  "outline": [
    "Intro: The Ayurvedic perspective on stress and evening wind-down",
    "How routines support the nervous system",
    "Ashwagandha for stress resilience",
    "Brahmi Tailam head massage for relaxation",
    "Safety notes and when to seek professional help"
  ],
  "retrieval_queries": [
    "Ayurveda stress sleep routines",
    "Ashwagandha stress resilience",
    "Brahmi Tailam Vata soothing head massage"
  ]
}
```

*What could go wrong:* It might forget to include a safety section, or not tie the products clearly into the outline.

*Guardrail:* Automatically check that (a) there's at least one safety-related section, and (b) each product from the brief appears in at least one section heading. If not, append them and flag for review.

---

**Step 2: Outline → Draft with Citations**

*What it does:* For each outline section, retrieves relevant chunks and writes the section, citing sources inline.

*What could go wrong:* Adds plausible-sounding benefits or advice that isn't actually in the corpus.

*Guardrail:* Any sentence with a product name or specific benefit claim needs at least one citation. A simple checker can flag sentences that make claims without citations.

---

**Step 3: Fact-Check Pass**

*What it does:* Goes sentence by sentence, re-retrieves from the corpus, and labels each claim as "supported," "unsupported," or "contradicted."

*What could go wrong:* Might be too lenient---marking things as "close enough" when the specific claim isn't actually there.

*Guardrail:* For medical-adjacent claims (immunity, metabolism, disease-related language), require explicit lexical overlap---the word has to actually appear in the source, not just be semantically similar.

---

**Step 4: Tone & Style Polish**

*What it does:* Rewrites the supported content to match Kerala Ayurveda's voice---calm, supportive, non-medical---while preserving the citations.

*What could go wrong:* Might accidentally remove or shuffle citation markers during rewriting.

*Guardrail:* Hard rule in the prompt not to touch `[Source N]` markers. Post-editing validation confirms they're all still there and correctly mapped.

---

**Step 5: Assembly & Export**

*What it does:* Stitches sections together into a final Markdown draft with a bibliography-style citation list at the end.

*What could go wrong:* Might drop a section or duplicate content.

*Guardrail:* Cross-check final headings against the original outline---if anything's missing, flag the draft as incomplete.

---

### How I'd Evaluate This

Before shipping, we need some way to know if the system is actually producing good output.

**A small "golden set":**

I'd put together maybe 5--10 representative briefs/questions---things like:
- "Explain Triphala for digestion (beginner-friendly)"
- "Evening self-care with Brahmi Tailam for professionals"
- "Who should use Ashwagandha tablets and what precautions apply?"

For each, I'd have:
- A reference answer written or approved by the content team
- A checklist of must-haves (e.g., "mentions 'gentle'," "includes safety note")
- A list of no-go phrases for that topic

**What to score:**
- *Grounding*: What % of claims are actually backed by the cited sources?
- *Coverage*: Does it hit all the required elements?
- *Tone*: Does it avoid medical claims and match the brand voice?
- *Editor verdict*: "Ready with light edits" / "Needs major work" / "Not usable"

**Metrics to track over time:**
- Hallucination rate (unsupported claims / total claims)
- Citation accuracy (% judged correct by reviewers)
- Editor acceptance rate (how often drafts need minimal vs major edits)
- How often the safety filters trigger (helps tune the prompts)

---

### What I'd Actually Build in 2 Weeks

Being realistic about scope:

**Definitely shipping:**
1.  **A working Q&A tool** --- the RAG pipeline from Part A, with a simple internal UI (query box, answer panel, source list).
2.  **A basic draft generator** --- brief → outline → RAG-written sections with citations. Single-pass, no fancy multi-agent orchestration. Just a script or service that runs the steps sequentially.
3.  **An evaluation notebook** --- golden set in a JSON file, script to run the system on those prompts and compute basic scores (missing safety notes, banned phrases, citation counts).
4.  **Safety guardrails** --- prompt constraints plus lexical filters for disease claims, dosing, miracle language.

**Explicitly not doing yet:**
- *Complex agent frameworks* --- overkill for MVP; a simple orchestrator function is fine.
- *Full LLM-based fact-checking* --- too slow and expensive for every draft; rely on prompt constraints and human review initially.
- *Multi-language support* --- English first, expand later.
- *Fancy dashboards* --- logs and a basic report are enough to start.
- *Lots of user config options* --- keep it simple until people actually request specific controls.

The goal is a stable, safe tool that saves the content team real time on FAQ answers and first drafts, even if it still needs human review before publishing.

---

## Reflection

**Time spent:** About 2.5--3 hours total---reading through the corpus, sketching the RAG design, working through the agent steps, and writing up examples.

**What I found interesting:** The balancing act between being helpful and staying safe. Ayurveda has real nuance, but we can't make medical claims. Figuring out where the line is between "okay to say" and "needs to come verbatim from the docs" is tricky and probably needs ongoing calibration with the content team.

**What's still unclear:** The exact threshold for "acceptable soft extrapolation" (generic wellbeing language) vs. "must be explicitly in the corpus." That's a judgment call that'll need real feedback from editors.

**Tools used:** I used AI assistance (Gemini-3pro) to help draft and organize this document, working from the content pack files. No external research---everything is grounded in the provided corpus.
```
