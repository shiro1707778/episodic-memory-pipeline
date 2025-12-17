"""
Memory worthiness classifier.

Determines whether input text contains information worth storing as memory.
This is a crucial gate to prevent memory bloat and noise.
"""
import json
from typing import Optional
from dataclasses import dataclass

from ..llm import LLMProvider
from ..prompts import PromptTemplates
from ..utils import as_bool, as_float, as_str


@dataclass
class ClassificationResult:
    """Result of memory worthiness classification."""
    is_memory_worthy: bool
    confidence: float
    reason: str
    memory_type: str  # episodic, fact, goal, preference, reflection, none
    
    @classmethod
    def not_worthy(cls, reason: str = "Not memory-worthy") -> "ClassificationResult":
        return cls(
            is_memory_worthy=False,
            confidence=1.0,
            reason=reason,
            memory_type="none"
        )


class MemoryWorthinessClassifier:
    """
    Classifies whether input text is worth storing as memory.
    
    Uses a combination of:
    1. Heuristic pre-filtering (fast, no LLM call)
    2. LLM-based classification (accurate, slower)
    
    Heuristics catch obvious cases to save LLM calls.
    """
    
    # Patterns that are almost never memory-worthy
    NOT_WORTHY_PATTERNS = [
        # Greetings
        r"^(hi|hello|hey|good morning|good evening|good night)[\s!.]*$",
        # Simple acknowledgments
        r"^(ok|okay|sure|yes|no|thanks|thank you|got it|understood)[\s!.]*$",
        # Generic questions
        r"^(what is|how do|can you|could you|will you|would you)\s+(the|a)\s+",
        # System commands
        r"^(help|exit|quit|stop|cancel|clear)[\s!.]*$",
    ]
    
    # Minimum length for LLM classification
    MIN_LENGTH_FOR_LLM = 10
    
    # Patterns that are likely memory-worthy (skip to extraction)
    LIKELY_WORTHY_PATTERNS = [
        r"\b(i am|i'm|my name is|i live|i work)\b",
        r"\b(i want|i need|my goal|i prefer|i like|i don't like|i hate)\b",
        r"\b(i learned|i realized|i decided|i think that)\b",
        r"\b(yesterday|last week|today i|this morning)\b",
    ]
    
    def __init__(self, llm: LLMProvider, threshold: float = 0.6):
        """
        Initialize classifier.
        
        Args:
            llm: LLM provider for classification
            threshold: Minimum confidence to consider worthy (default 0.6)
        """
        self.llm = llm
        self.threshold = threshold
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        import re
        self._not_worthy_re = [
            re.compile(p, re.IGNORECASE) for p in self.NOT_WORTHY_PATTERNS
        ]
        self._likely_worthy_re = [
            re.compile(p, re.IGNORECASE) for p in self.LIKELY_WORTHY_PATTERNS
        ]
    
    def classify(
        self,
        text: str,
        previous_topic: Optional[str] = None,
        context: Optional[str] = None,
        use_llm: bool = True
    ) -> ClassificationResult:
        """
        Classify whether text is memory-worthy.
        
        Args:
            text: Input text to classify
            previous_topic: Topic from previous turn (if any)
            context: Additional context about the conversation
            use_llm: Whether to use LLM for classification
            
        Returns:
            ClassificationResult with decision and reasoning
        """
        text = text.strip()
        
        # Empty or too short
        if not text or len(text) < 3:
            return ClassificationResult.not_worthy("Empty or too short")
        
        # Check not-worthy patterns (fast reject)
        for pattern in self._not_worthy_re:
            if pattern.match(text):
                return ClassificationResult.not_worthy("Matches non-memory pattern")
        
        # Check likely-worthy patterns (fast accept, still classify type)
        for pattern in self._likely_worthy_re:
            if pattern.search(text):
                if use_llm and len(text) >= self.MIN_LENGTH_FOR_LLM:
                    # Use LLM to get accurate type
                    return self._llm_classify(text, previous_topic, context)
                else:
                    return ClassificationResult(
                        is_memory_worthy=True,
                        confidence=0.7,
                        reason="Matches memory-worthy pattern",
                        memory_type="episodic"  # Default
                    )
        
        # For medium-length text, use LLM
        if use_llm and len(text) >= self.MIN_LENGTH_FOR_LLM:
            return self._llm_classify(text, previous_topic, context)
        
        # Default: not worthy for short, ambiguous text
        return ClassificationResult.not_worthy("Too short or ambiguous for memory")
    
    def _llm_classify(
        self,
        text: str,
        previous_topic: Optional[str],
        context: Optional[str]
    ) -> ClassificationResult:
        """Use LLM to classify memory worthiness."""
        prompt = PromptTemplates.MEMORY_WORTHINESS.format(
            text=text,
            previous_topic=previous_topic or "none",
            context=context or "none"
        )
        
        try:
            response = self.llm.complete(prompt)
            result = json.loads(response)
            
            # Sanitize all LLM output fields using centralized helpers
            is_worthy = as_bool(result.get("is_memory_worthy"), default=False)
            confidence = as_float(result.get("confidence"), default=0.5)
            reason = as_str(result.get("reason"), default="LLM classification") or "LLM classification"
            memory_type = as_str(result.get("memory_type"), default="none") or "none"
            
            return ClassificationResult(
                is_memory_worthy=is_worthy,
                confidence=confidence,
                reason=reason,
                memory_type=memory_type
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If LLM response is malformed, be conservative
            return ClassificationResult(
                is_memory_worthy=False,
                confidence=0.3,
                reason=f"Classification error: {str(e)}",
                memory_type="none"
            )
    
    def batch_classify(
        self,
        texts: list[str],
        use_llm: bool = True
    ) -> list[ClassificationResult]:
        """Classify multiple texts (uses heuristics + batched LLM when needed)."""
        results = []
        llm_needed = []
        llm_indices = []
        
        # First pass: heuristics
        for i, text in enumerate(texts):
            result = self.classify(text, use_llm=False)
            if result.is_memory_worthy or not use_llm:
                results.append(result)
            else:
                # Need LLM classification
                results.append(None)
                llm_needed.append(text)
                llm_indices.append(i)
        
        # Second pass: LLM for uncertain cases
        if llm_needed:
            for idx, text in zip(llm_indices, llm_needed):
                results[idx] = self._llm_classify(text, None, None)
        
        return results

