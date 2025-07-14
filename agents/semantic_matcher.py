from typing import List, Tuple, Optional
import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

class SemanticMatch:
    """Represents a semantic match result"""
    
    def __init__(self, text: str, similarity: float, start_pos: int, end_pos: int):
        self.text = text
        self.similarity = similarity  # 0.0 to 1.0
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __repr__(self):
        return f"SemanticMatch(text='{self.text}', similarity={self.similarity:.2f})"

class SemanticTextMatcher:
    """
    Handles semantic text matching for precision editing.
    Finds similar text when exact matches fail.
    """
    
    def __init__(self, min_similarity: float = 0.6):
        self.min_similarity = min_similarity
    
    def find_semantic_matches(self, target: str, content: str, max_matches: int = 3) -> List[SemanticMatch]:
        """
        Find semantically similar text in content when exact match fails.
        
        Args:
            target: The text we're looking for
            content: The content to search in
            max_matches: Maximum number of matches to return
            
        Returns:
            List of SemanticMatch objects sorted by similarity (highest first)
        """
        if not target or not content:
            return []
        
        target_clean = self._normalize_text(target)
        matches = []
        
        # Try different matching strategies
        matches.extend(self._find_fuzzy_matches(target, target_clean, content))
        matches.extend(self._find_partial_matches(target, target_clean, content))
        matches.extend(self._find_word_order_matches(target, target_clean, content))
        
        # Remove duplicates and sort by similarity
        unique_matches = self._deduplicate_matches(matches)
        unique_matches.sort(key=lambda m: m.similarity, reverse=True)
        
        # Filter by minimum similarity and limit results
        filtered_matches = [m for m in unique_matches if m.similarity >= self.min_similarity]
        
        logger.info(f"Found {len(filtered_matches)} semantic matches for '{target}' (min_similarity={self.min_similarity})")
        
        return filtered_matches[:max_matches]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize contractions
        contractions = {
            "i'm": "i am",
            "you're": "you are", 
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "won't": "will not",
            "can't": "cannot",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text.strip()
    
    def _find_fuzzy_matches(self, original_target: str, target_clean: str, content: str) -> List[SemanticMatch]:
        """Find matches using fuzzy string matching"""
        matches = []
        content_lower = content.lower()
        
        # Try sliding window approach with different window sizes
        target_len = len(original_target)
        
        for window_size in [target_len, target_len + 10, target_len - 5, target_len + 20]:
            if window_size <= 0:
                continue
                
            for i in range(len(content) - window_size + 1):
                window = content[i:i + window_size]
                window_clean = self._normalize_text(window)
                
                # Calculate similarity
                similarity = SequenceMatcher(None, target_clean, window_clean).ratio()
                
                if similarity >= self.min_similarity:
                    matches.append(SemanticMatch(
                        text=window.strip(),
                        similarity=similarity,
                        start_pos=i,
                        end_pos=i + window_size
                    ))
        
        return matches
    
    def _find_partial_matches(self, original_target: str, target_clean: str, content: str) -> List[SemanticMatch]:
        """Find partial matches (target is substring or vice versa)"""
        matches = []
        content_lower = content.lower()
        target_lower = original_target.lower()
        
        # Look for target as substring of content
        start = 0
        while start < len(content_lower):
            pos = content_lower.find(target_lower, start)
            if pos == -1:
                break
            
            # Expand to word boundaries if possible
            expanded_start = pos
            expanded_end = pos + len(target_lower)
            
            # Expand to include nearby punctuation/whitespace
            while expanded_start > 0 and content[expanded_start - 1] in ' \t':
                expanded_start -= 1
            while expanded_end < len(content) and content[expanded_end] in ' \t.,!?;:':
                expanded_end += 1
            
            match_text = content[expanded_start:expanded_end].strip()
            similarity = 0.9  # High similarity for substring matches
            
            matches.append(SemanticMatch(
                text=match_text,
                similarity=similarity,
                start_pos=expanded_start,
                end_pos=expanded_end
            ))
            
            start = pos + 1
        
        # Look for content substrings that match target (but avoid infinite recursion)
        words = original_target.split()
        if len(words) > 1:
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    partial = ' '.join(words[i:j])
                    if len(partial) >= 3 and partial != original_target:  # Only consider meaningful partials, avoid self-recursion
                        # Use simple substring matching instead of recursive call
                        partial_lower = partial.lower()
                        start = 0
                        while start < len(content_lower):
                            pos = content_lower.find(partial_lower, start)
                            if pos == -1:
                                break
                            
                            # Expand to include nearby punctuation/whitespace
                            expanded_start = pos
                            expanded_end = pos + len(partial_lower)
                            
                            while expanded_start > 0 and content[expanded_start - 1] in ' \t':
                                expanded_start -= 1
                            while expanded_end < len(content) and content[expanded_end] in ' \t.,!?;:':
                                expanded_end += 1
                            
                            match_text = content[expanded_start:expanded_end].strip()
                            similarity = 0.7  # Reduced similarity for partial matches
                            
                            matches.append(SemanticMatch(
                                text=match_text,
                                similarity=similarity,
                                start_pos=expanded_start,
                                end_pos=expanded_end
                            ))
                            
                            start = pos + 1
        
        return matches
    
    def _find_word_order_matches(self, original_target: str, target_clean: str, content: str) -> List[SemanticMatch]:
        """Find matches where words might be in different order"""
        matches = []
        target_words = set(target_clean.split())
        
        if len(target_words) < 2:
            return matches
        
        content_lower = content.lower()
        
        # Look for areas with many target words
        for i in range(len(content)):
            for j in range(i + len(original_target) // 2, min(i + len(original_target) * 2, len(content) + 1)):
                window = content[i:j]
                window_clean = self._normalize_text(window)
                window_words = set(window_clean.split())
                
                # Calculate word overlap
                common_words = target_words.intersection(window_words)
                if len(common_words) >= len(target_words) * 0.7:  # At least 70% word overlap
                    similarity = len(common_words) / len(target_words) * 0.8  # Max 0.8 for word order matches
                    
                    matches.append(SemanticMatch(
                        text=window.strip(),
                        similarity=similarity,
                        start_pos=i,
                        end_pos=j
                    ))
        
        return matches
    
    def _deduplicate_matches(self, matches: List[SemanticMatch]) -> List[SemanticMatch]:
        """Remove duplicate and overlapping matches"""
        if not matches:
            return []
        
        # Sort by position
        matches.sort(key=lambda m: m.start_pos)
        
        deduplicated = []
        for match in matches:
            # Check if this match significantly overlaps with any existing match
            overlaps = False
            for existing in deduplicated:
                overlap_start = max(match.start_pos, existing.start_pos)
                overlap_end = min(match.end_pos, existing.end_pos)
                overlap_length = max(0, overlap_end - overlap_start)
                
                match_length = match.end_pos - match.start_pos
                existing_length = existing.end_pos - existing.start_pos
                
                # If more than 50% overlap, consider it a duplicate
                if (overlap_length > match_length * 0.5 or 
                    overlap_length > existing_length * 0.5):
                    overlaps = True
                    # Keep the one with higher similarity
                    if match.similarity > existing.similarity:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return deduplicated
    
    def suggest_replacements(self, target: str, content: str) -> List[str]:
        """
        Suggest user-friendly replacement options when exact match fails.
        
        Returns:
            List of user-friendly suggestion strings
        """
        matches = self.find_semantic_matches(target, content)
        
        suggestions = []
        for i, match in enumerate(matches):
            confidence = "high" if match.similarity > 0.85 else "medium" if match.similarity > 0.7 else "low"
            suggestions.append(
                f"Option {i + 1}: '{match.text}' ({confidence} confidence - {match.similarity:.0%} similar)"
            )
        
        return suggestions
    
    def get_best_match(self, target: str, content: str) -> Optional[SemanticMatch]:
        """Get the single best semantic match for a target"""
        matches = self.find_semantic_matches(target, content, max_matches=1)
        return matches[0] if matches else None

# Global instance
semantic_matcher = SemanticTextMatcher(min_similarity=0.6) 