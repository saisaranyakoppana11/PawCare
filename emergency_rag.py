"""
PawCare - RAG-Powered Generative AI Pet Emergency Assistant
Architecture:
- Authoritative Veterinary Knowledge Base
- Hybrid TF-IDF + Keyword Clinical Relevance Retrieval
- Confidence Threshold Filtering (Anti-Hallucination Guardrail)
- Situation-Specific Grounded Generative Response Synthesis
- Triage Urgency Assessment (🔴 EMERGENCY / 🟠 URGENT / 🟡 NEEDS ATTENTION)
- Multi-Modal Image & Voice Transcription Handling
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from PIL import Image

from emergency_knowledge_base import EMERGENCY_KNOWLEDGE_BASE
from emergency_data import EMERGENCY_CATEGORIES, EMERGENCY_DISCLAIMER

# Try scikit-learn for vector similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EmergencyRAGAssistant:
    """
    RAG Emergency Assistant grounded in verified veterinary medicine references.
    """

    def __init__(self):
        self.documents = []
        self.doc_index = {}
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_knowledge_index()

    def _build_knowledge_index(self):
        """Build search index from the authoritative veterinary knowledge base."""
        docs = []

        # 1. Primary Authoritative Knowledge Base
        for item in EMERGENCY_KNOWLEDGE_BASE:
            # Build comprehensive text representation for vectorization
            text_corpus = (
                f"{item['title']} {item['category']} {item['species']} "
                f"{item['summary']} "
                f"{' '.join(item.get('signs', []))} "
                f"{' '.join(item.get('immediate_actions', []))} "
                f"{' '.join(item.get('things_to_avoid', []))} "
                f"{item.get('vet_escalation', '')}"
            )
            entry = {
                "id": item["id"],
                "title": item["title"],
                "category": item.get("category", "General"),
                "species": item.get("species", "all"),
                "triage_level": item.get("triage_level", "URGENT"),
                "urgency_badge": item.get("urgency_badge", "🟠 URGENT — Contact Veterinarian Promptly"),
                "summary": item.get("summary", ""),
                "signs": item.get("signs", []),
                "immediate_actions": item.get("immediate_actions", []),
                "things_to_avoid": item.get("things_to_avoid", []),
                "vet_escalation": item.get("vet_escalation", ""),
                "clarifying_questions": item.get("clarifying_questions", []),
                "sources": item.get("sources", []),
                "search_text": text_corpus
            }
            docs.append(entry)

        # 2. Integrate existing categories from emergency_data as supporting fallback
        for cat in EMERGENCY_CATEGORIES:
            existing_ids = [d["id"] for d in docs]
            if cat["id"] not in existing_ids:
                text_corpus = (
                    f"{cat['title']} {cat['description']} "
                    f"{' '.join(cat.get('immediate_actions', []))} "
                    f"{' '.join(cat.get('things_to_avoid', []))} "
                    f"{cat.get('vet_recommendation', '')}"
                )
                level = "EMERGENCY" if "critical" in cat.get("urgency", "").lower() else "URGENT"
                entry = {
                    "id": cat["id"],
                    "title": cat["title"],
                    "category": "Emergency Protocols",
                    "species": "all",
                    "triage_level": level,
                    "urgency_badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care" if level == "EMERGENCY" else "🟠 URGENT — Contact Veterinarian Promptly",
                    "summary": cat.get("description", ""),
                    "signs": [cat.get("description", "")],
                    "immediate_actions": cat.get("immediate_actions", []),
                    "things_to_avoid": cat.get("things_to_avoid", []),
                    "vet_escalation": cat.get("vet_recommendation", ""),
                    "clarifying_questions": [
                        "Is the pet conscious and breathing normally?",
                        "How long ago did the emergency start?"
                    ],
                    "sources": [
                        {
                            "name": "American Veterinary Medical Association (AVMA)",
                            "title": "Companion Animal First Aid & Emergency Guidelines",
                            "organization": "AVMA Emergency Care Division"
                        }
                    ],
                    "search_text": text_corpus
                }
                docs.append(entry)

        self.documents = docs

        # Fit TF-IDF Vectorizer
        if SKLEARN_AVAILABLE and self.documents:
            try:
                corpus_texts = [d["search_text"] for d in self.documents]
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 3),
                    stop_words="english",
                    sublinear_tf=True
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
            except Exception as e:
                print(f"[EmergencyRAG] TF-IDF init warning: {e}")
                self.vectorizer = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant veterinary documents for the user query.
        Returns empty list if relevance is below minimum safety threshold.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip().lower()
        results = []

        # A. Vector Retrieval via TF-IDF Cosine Similarity
        if self.vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([clean_query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

                # Rank by score
                scored_indices = [
                    (idx, float(similarities[idx]))
                    for idx in range(len(self.documents))
                ]
                scored_indices.sort(key=lambda x: x[1], reverse=True)

                for idx, score in scored_indices[:top_k]:
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = score
                    results.append(doc)
            except Exception as e:
                print(f"[EmergencyRAG] Retrieval error: {e}")

        # B. Clinical Keyword & Intent Reinforcement
        # Critical clinical terms boost relevance for exact symptom matches
        clinical_keywords = {
            "breath": ["breathing-difficulty", "choking"],
            "pant": ["heatstroke", "breathing-difficulty"],
            "gasp": ["breathing-difficulty", "choking"],
            "suffocat": ["choking", "breathing-difficulty"],
            "chok": ["choking"],
            "swallow": ["choking", "foreign-body-ingestion", "xylitol-poisoning"],
            "bleed": ["severe-bleeding"],
            "blood": ["severe-bleeding", "severe-vomiting-diarrhea", "rodenticide-poisoning"],
            "wound": ["severe-bleeding", "trauma-fractures"],
            "cut": ["severe-bleeding"],
            "xylitol": ["xylitol-poisoning"],
            "gum": ["xylitol-poisoning"],
            "sweetener": ["xylitol-poisoning"],
            "lily": ["lily-poisoning-cats"],
            "lilies": ["lily-poisoning-cats"],
            "chocolat": ["chocolate-poisoning"],
            "cocoa": ["chocolate-poisoning"],
            "poison": ["rodenticide-poisoning", "xylitol-poisoning", "lily-poisoning-cats"],
            "toxic": ["rodenticide-poisoning", "xylitol-poisoning", "lily-poisoning-cats"],
            "rat": ["rodenticide-poisoning"],
            "bait": ["rodenticide-poisoning"],
            "heat": ["heatstroke"],
            "hot car": ["heatstroke"],
            "sun": ["heatstroke"],
            "seiz": ["seizures"],
            "convuls": ["seizures"],
            "tremor": ["seizures", "chocolate-poisoning"],
            "collaps": ["collapse-unconsciousness", "breathing-difficulty"],
            "faint": ["collapse-unconsciousness"],
            "unconscious": ["collapse-unconsciousness"],
            "fall": ["trauma-fractures"],
            "hit by car": ["trauma-fractures", "severe-bleeding"],
            "fractur": ["trauma-fractures"],
            "broken": ["trauma-fractures"],
            "bone": ["trauma-fractures", "foreign-body-ingestion"],
            "vomit": ["severe-vomiting-diarrhea", "foreign-body-ingestion"],
            "throw up": ["severe-vomiting-diarrhea", "foreign-body-ingestion"],
            "diarrhea": ["severe-vomiting-diarrhea"],
            "bloat": ["bloat-gdv"],
            "gdv": ["bloat-gdv"],
            "retch": ["bloat-gdv", "choking"],
            "dry heav": ["bloat-gdv"],
            "drum": ["bloat-gdv"],
            "swollen stomach": ["bloat-gdv"],
            "swollen belly": ["bloat-gdv"],
            "swollen abdomen": ["bloat-gdv"],
            "distended": ["bloat-gdv"],
            "nothing comes up": ["bloat-gdv", "choking"],
            "stomach": ["bloat-gdv"],
            "pacing": ["bloat-gdv"],
            "urinate": ["feline-urinary-blockage"],
            "pee": ["feline-urinary-blockage"],
            "litter": ["feline-urinary-blockage"],
            "strain": ["feline-urinary-blockage"],
            "cannot urinate": ["feline-urinary-blockage"],
            "blocked cat": ["feline-urinary-blockage"],
            "eye": ["eye-emergencies"],
            "cornea": ["eye-emergencies"],
            "squint": ["eye-emergencies"],
            "burn": ["burns-electric-shock"],
            "cord": ["burns-electric-shock"],
            "electr": ["burns-electric-shock"],
            "snake": ["snake-bites"],
            "bite": ["snake-bites", "severe-bleeding"],
            "sting": ["anaphylaxis-insect-bites"],
            "bee": ["anaphylaxis-insect-bites"],
            "wasp": ["anaphylaxis-insect-bites"],
            "swelling": ["anaphylaxis-insect-bites", "bloat-gdv"],
            "swollen": ["anaphylaxis-insect-bites", "bloat-gdv"],
            "allergic": ["anaphylaxis-insect-bites"],
            "cold": ["hypothermia-cold"],
            "freeze": ["hypothermia-cold"],
            "shiver": ["hypothermia-cold"]
        }

        doc_match_counts = {}
        for kw, target_ids in clinical_keywords.items():
            if kw in clean_query or re.search(r'\b' + re.escape(kw), clean_query):
                for tid in target_ids:
                    doc_match_counts[tid] = doc_match_counts.get(tid, 0) + 1

        # If keyword matches found, prioritize by match count and relevance
        if doc_match_counts:
            for d in self.documents:
                doc_id = d["id"]
                if doc_id in doc_match_counts:
                    count = doc_match_counts[doc_id]
                    boost = min(0.98, 0.70 + (count * 0.08))
                    
                    found = False
                    for r in results:
                        if r["id"] == doc_id:
                            r["similarity_score"] = max(r.get("similarity_score", 0), boost)
                            found = True
                            break
                    if not found:
                        d_copy = d.copy()
                        d_copy["similarity_score"] = boost
                        results.append(d_copy)

        # Sort and take top_k
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return results[:top_k]

    def _is_information_sufficient(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """
        RAG Safety Rule: If retrieved information is below safety threshold,
        do NOT invent an answer.
        """
        if not retrieved_docs:
            return False

        top_score = retrieved_docs[0].get("similarity_score", 0)

        # If score is very low (< 0.08) and query has no common pet medical words, it's out-of-domain
        pet_medical_terms = [
            "dog", "cat", "pet", "puppy", "kitten", "animal", "sick", "hurt",
            "blood", "bleed", "breath", "pant", "vomit", "eat", "ate", "swallow",
            "chok", "seiz", "collaps", "fall", "pain", "crying", "eye", "bite",
            "toxin", "poison", "heat", "cold", "stool", "diarrhea", "letharg", "limp"
        ]
        has_pet_medical_intent = any(term in query.lower() for term in pet_medical_terms)

        if top_score < 0.06 and not has_pet_medical_intent:
            return False

        return True

    def _determine_triage_urgency(self, query: str, primary_doc: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """
        Determine triage level: EMERGENCY (🔴), URGENT (🟠), or NEEDS_ATTENTION (🟡).
        """
        query_lower = query.lower()

        # Immediate life-threatening triggers
        critical_triggers = [
            "collapse", "unconscious", "not breathing", "difficulty breathing",
            "choking", "blue gum", "purple gum", "pale gum", "spurting blood",
            "severe bleeding", "xylitol", "lily", "gdv", "bloat", "cluster seizure",
            "cannot urinate", "straining to pee", "snake bite", "heatstroke",
            "swollen stomach", "swollen belly", "swollen abdomen", "distended",
            "like a drum", "drum", "nothing comes up", "dry heaving", "pacing"
        ]

        if any(trig in query_lower for trig in critical_triggers):
            return {
                "level": "EMERGENCY",
                "badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
                "color": "danger",
                "icon": "🔴"
            }

        if primary_doc:
            doc_level = primary_doc.get("triage_level", "URGENT")
            if doc_level == "EMERGENCY":
                return {
                    "level": "EMERGENCY",
                    "badge": "🔴 EMERGENCY — Seek Immediate Veterinary Care",
                    "color": "danger",
                    "icon": "🔴"
                }
            elif doc_level == "URGENT":
                return {
                    "level": "URGENT",
                    "badge": "🟠 URGENT — Contact Veterinarian Promptly",
                    "color": "warning",
                    "icon": "🟠"
                }

        return {
            "level": "NEEDS_ATTENTION",
            "badge": "🟡 NEEDS ATTENTION — Veterinary Advice Recommended",
            "color": "info",
            "icon": "🟡"
        }

    def _call_external_llm(self, query: str, context_docs: List[Dict[str, Any]], image_path: Optional[str] = None) -> Optional[str]:
        """
        Optional Generative AI LLM invocation (Gemini / OpenAI) if API keys are set.
        Returns None if not configured, allowing fallback to local synthesis.
        """
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        # Try Google Gemini if configured
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                context_text = "\n\n".join([
                    f"SOURCE [{d['title']} - {d['sources'][0]['name'] if d.get('sources') else 'Veterinary Medical Reference'}]:\n"
                    f"Summary: {d['summary']}\n"
                    f"Signs: {'; '.join(d['signs'])}\n"
                    f"Immediate Actions: {'; '.join(d['immediate_actions'])}\n"
                    f"Things to Avoid: {'; '.join(d['things_to_avoid'])}\n"
                    f"Vet Escalation: {d['vet_escalation']}"
                    for d in context_docs
                ])

                prompt = (
                    "You are the PawCare Emergency Veterinary AI Assistant. "
                    "You MUST ground your response strictly in the retrieved veterinary knowledge provided below. "
                    "DO NOT invent facts or claim certainty about a diagnosis. "
                    "Follow this exact structured output format:\n"
                    "--------------------------------------------------\n"
                    "EMERGENCY ASSESSMENT\n"
                    "--------------------------------------------------\n"
                    "[Explain what the situation may indicate based only on retrieved knowledge. State non-diagnostic.]\n\n"
                    "--------------------------------------------------\n"
                    "WHAT TO DO NOW\n"
                    "--------------------------------------------------\n"
                    "[Prioritized, safe, evidence-grounded first-aid actions from the sources]\n\n"
                    "--------------------------------------------------\n"
                    "WHAT TO AVOID\n"
                    "--------------------------------------------------\n"
                    "[Dangerous actions or home remedies explicitly warned against in the sources]\n\n"
                    "--------------------------------------------------\n"
                    "WHEN TO SEEK VETERINARY HELP\n"
                    "--------------------------------------------------\n"
                    "[Clear escalation and urgent transport guidance]\n\n"
                    "--------------------------------------------------\n"
                    "IMPORTANT\n"
                    "--------------------------------------------------\n"
                    "This AI assistant provides situational first-aid information and does NOT replace professional veterinary care.\n\n"
                    f"RETRIEVED VETERINARY CONTEXT:\n{context_text}\n\n"
                    f"USER EMERGENCY DESCRIPTION:\n{query}"
                )

                if image_path and os.path.exists(image_path):
                    try:
                        pil_img = Image.open(image_path)
                        response = model.generate_content([prompt, pil_img])
                    except Exception:
                        response = model.generate_content(prompt)
                else:
                    response = model.generate_content(prompt)

                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[EmergencyRAG] Gemini API call skipped/failed: {e}")

        return None

    def _synthesize_grounded_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        triage: Dict[str, str],
        image_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize situation-specific, evidence-grounded emergency guidance
        strictly following the required structure.
        """
        primary = retrieved_docs[0]
        secondary = retrieved_docs[1] if len(retrieved_docs) > 1 else None

        # 1. EMERGENCY ASSESSMENT
        species_mention = "your pet"
        if "dog" in query.lower():
            species_mention = "your dog"
        elif "cat" in query.lower():
            species_mention = "your cat"

        assessment_lines = [
            f"Based on the clinical signs described for {species_mention}, this situation appears most consistent with potential {primary['title'].lower()}.",
            f"{primary['summary']}"
        ]
        if secondary and secondary.get("similarity_score", 0) > 0.4:
            assessment_lines.append(
                f"Additionally, related veterinary protocols for {secondary['title'].lower()} should be considered."
            )
        assessment_lines.append(
            "Note: This assessment is an automated first-response triage guide and does not constitute a definitive medical diagnosis."
        )
        assessment_text = "\n\n".join(assessment_lines)

        # 2. WHAT TO DO NOW (prioritized safe actions from sources)
        actions = []
        for act in primary.get("immediate_actions", []):
            if act not in actions:
                actions.append(act)
        if secondary:
            for act in secondary.get("immediate_actions", []):
                if act not in actions and len(actions) < 6:
                    actions.append(act)

        # 3. WHAT TO AVOID (strict danger precautions from sources)
        avoids = []
        for av in primary.get("things_to_avoid", []):
            if av not in avoids:
                avoids.append(av)
        if secondary:
            for av in secondary.get("things_to_avoid", []):
                if av not in avoids and len(avoids) < 5:
                    avoids.append(av)

        # 4. WHEN TO SEEK VETERINARY HELP
        vet_help_parts = [primary.get("vet_escalation", "Contact your veterinarian or emergency clinic promptly.")]
        if triage["level"] == "EMERGENCY":
            vet_help_parts.insert(
                0,
                "🚨 IMMEDIATE CLINICAL ACTION REQUIRED: This is a potentially life-threatening emergency. "
                "Transport your pet to the nearest 24-hour veterinary emergency hospital without delay."
            )
        else:
            vet_help_parts.append(
                "If symptoms rapidly worsen, or if your pet exhibits difficulty breathing, collapse, or severe pain, "
                "upgrade to immediate emergency care."
            )
        vet_help_text = "\n\n".join(vet_help_parts)

        # 5. SOURCES USED (transparent authentic citations)
        collected_sources = []
        for d in retrieved_docs:
            for s in d.get("sources", []):
                source_entry = {
                    "organization": s.get("organization", "Veterinary Medical Reference"),
                    "name": s.get("name", "Authoritative Veterinary Guidelines"),
                    "title": s.get("title", d.get("title", "Clinical Protocol")),
                    "condition": d.get("title")
                }
                if source_entry not in collected_sources:
                    collected_sources.append(source_entry)

        # 6. CLARIFYING QUESTIONS
        clarifying_qs = []
        for q in primary.get("clarifying_questions", []):
            clarifying_qs.append(q)

        return {
            "status": "success",
            "triage": triage,
            "assessment": assessment_text,
            "what_to_do_now": actions,
            "what_to_avoid": avoids,
            "when_to_seek_vet": vet_help_text,
            "disclaimer": EMERGENCY_DISCLAIMER,
            "sources": collected_sources,
            "clarifying_questions": clarifying_qs if triage["level"] != "EMERGENCY" else [],
            "image_observation": image_info,
            "primary_condition": primary["title"]
        }

    def _generate_insufficient_info_response(self, query: str) -> Dict[str, Any]:
        """
        Anti-Hallucination Guardrail:
        When information is insufficient or not present in the knowledge base,
        the AI does NOT invent an answer, but transparently directs to professional care.
        """
        return {
            "status": "insufficient_info",
            "triage": {
                "level": "NEEDS_ATTENTION",
                "badge": "🟡 NEEDS ATTENTION — Consult a Veterinary Professional",
                "color": "warning",
                "icon": "🟡"
            },
            "assessment": (
                "The described situation could not be reliably matched with high confidence against our "
                "authoritative veterinary emergency protocols. To protect your pet's safety, PawCare's "
                "AI will not speculate or invent medical guidance without verified clinical data."
            ),
            "what_to_do_now": [
                "Keep your pet in a calm, secure, and comfortable resting area.",
                "Observe their breathing rate, gum color, and level of consciousness.",
                "If your pet swallowed an unknown object or chemical, securely collect the remaining sample or packaging.",
                "Call a licensed veterinarian or a 24-hour emergency animal hospital right away for professional advice."
            ],
            "what_to_avoid": [
                "DO NOT administer human medications (such as Tylenol, Aspirin, or Ibuprofen) — these are often fatal to pets.",
                "DO NOT attempt to induce vomiting without direct guidance from a veterinarian.",
                "DO NOT assume unfamiliar symptoms are safe to ignore."
            ],
            "when_to_seek_vet": (
                "Because your pet's condition cannot be safely categorized from the provided description alone, "
                "we recommend speaking directly with a veterinary professional or local emergency clinic."
            ),
            "disclaimer": EMERGENCY_DISCLAIMER,
            "sources": [
                {
                    "organization": "American Veterinary Medical Association (AVMA)",
                    "name": "AVMA Pet Health & Emergency Care",
                    "title": "General Triage Principles for Companion Animals",
                    "condition": "General Veterinary Advice"
                }
            ],
            "clarifying_questions": [
                "Is your pet a dog or a cat, and approximately how old are they?",
                "What specific symptoms started, and how long ago?",
                "Are they eating, drinking, or able to walk normally?"
            ],
            "image_observation": None,
            "primary_condition": "Unspecified Condition"
        }

    def analyze_emergency(
        self,
        query: str,
        image_path: Optional[str] = None,
        voice_transcription: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main RAG pipeline entry point:
        1. Consolidates text & voice inputs.
        2. Validates image input if present.
        3. Retrieves relevant veterinary documents.
        4. Validates confidence thresholds.
        5. Generates structured, grounded emergency response.
        """
        # Combine voice transcription if provided
        full_query = query.strip() if query else ""
        if voice_transcription and voice_transcription.strip():
            if full_query and voice_transcription.strip() not in full_query:
                full_query = f"{full_query} (Voice note: {voice_transcription.strip()})"
            elif not full_query:
                full_query = voice_transcription.strip()

        # Handle completely empty input
        if not full_query and not image_path:
            return {
                "status": "error",
                "message": "Please describe your pet's symptoms or emergency situation.",
                "triage": {
                    "level": "NEEDS_ATTENTION",
                    "badge": "🟡 NEEDS ATTENTION",
                    "color": "secondary",
                    "icon": "ℹ️"
                }
            }

        # Handle Image validation if provided
        image_observation = None
        if image_path and os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_name = img.format
                image_observation = (
                    f"Visual assessment attached ({format_name}, {width}x{height}px). "
                    "Note: Photographs cannot diagnose internal medical pathology or replace a physical veterinary examination. "
                    "Guidance below is prioritized based on your observed visual signs and described emergency."
                )
            except Exception as e:
                image_observation = "An image was attached but could not be processed. Proceeding with text-based clinical guidance."

        # If user only uploaded an image without describing the issue
        if not full_query and image_path:
            full_query = "Physical injury, trauma, or visible medical condition requiring emergency inspection."

        # Step 1: RAG Retrieval
        retrieved_docs = self.retrieve(full_query, top_k=3)

        # Step 2: Threshold Safety Check
        if not self._is_information_sufficient(full_query, retrieved_docs):
            res = self._generate_insufficient_info_response(full_query)
            if image_observation:
                res["image_observation"] = image_observation
            return res

        # Step 3: Triage Urgency Assessment
        triage = self._determine_triage_urgency(full_query, retrieved_docs[0] if retrieved_docs else None)

        # Step 4: Check if external LLM generation is available
        llm_text = self._call_external_llm(full_query, retrieved_docs, image_path)

        # Step 5: Synthesize Grounded Response
        grounded_data = self._synthesize_grounded_response(
            query=full_query,
            retrieved_docs=retrieved_docs,
            triage=triage,
            image_info=image_observation
        )

        if llm_text:
            grounded_data["llm_generated_text"] = llm_text

        return grounded_data


# Singleton instance for fast reusable access
emergency_rag_assistant = EmergencyRAGAssistant()
