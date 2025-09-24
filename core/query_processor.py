"""
Natural Language Query Processing for RAG System
Handles complex queries and converts them to search parameters
"""
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from config.settings import METADATA_CATEGORIES, QUERY_TEMPLATES
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the intent of a search query"""
    query_type: str  # text, image, hybrid, metadata
    search_terms: List[str]
    filters: Dict[str, Any]
    image_path: Optional[str] = None
    similarity_threshold: float = 0.7
    limit: int = 10

class QueryProcessor:
    """Process natural language queries and extract search parameters"""
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.synonyms = self._build_synonyms()
        self.brand_aliases = self._build_brand_aliases()
    
    def _build_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for query parsing"""
        return {
            "brand": [
                r"(nike|adidas|puma|converse|vans|reebok|new balance|asics|under armour|jordan)",
                r"(\w+)\s+(shoes?|sneakers?|footwear)"
            ],
            "pattern": [
                r"(zigzag|circular|square|diamond|brand logo|logo)",
                r"(pattern|design|tread)\s+(zigzag|circular|square|diamond|logo)"
            ],
            "shape": [
                r"(round|square|oval|irregular|elongated)",
                r"(shape|outline)\s+(round|square|oval|irregular|elongated)"
            ],
            "size": [
                r"(small|medium|large|extra large|xl)",
                r"size\s+(small|medium|large|extra large|xl)"
            ],
            "color": [
                r"(red|blue|green|yellow|black|white|gray|grey|brown|pink|purple|orange)",
                r"color\s+(red|blue|green|yellow|black|white|gray|grey|brown|pink|purple|orange)"
            ],
            "style": [
                r"(athletic|running|basketball|tennis|casual|dress|formal|sport)",
                r"(sneakers?|shoes?|boots?|sandals?|flip flops?)"
            ],
            "activity": [
                r"(running|basketball|tennis|walking|hiking|gym|workout|sport)",
                r"for\s+(running|basketball|tennis|walking|hiking|gym|workout|sport)"
            ]
        }
    
    def _build_synonyms(self) -> Dict[str, List[str]]:
        """Build synonym mappings"""
        return {
            "shoe": ["shoes", "sneaker", "sneakers", "footwear", "footgear"],
            "red": ["crimson", "scarlet", "burgundy", "maroon"],
            "blue": ["navy", "azure", "cobalt", "royal"],
            "black": ["ebony", "charcoal", "jet"],
            "white": ["ivory", "cream", "pearl"],
            "large": ["big", "huge", "oversized"],
            "small": ["tiny", "mini", "petite"],
            "round": ["circular", "spherical"],
            "square": ["rectangular", "boxy"],
            "athletic": ["sport", "sports", "fitness", "exercise"]
        }
    
    def _build_brand_aliases(self) -> Dict[str, str]:
        """Build brand name aliases"""
        return {
            "nike air": "nike",
            "air jordan": "jordan",
            "jordan brand": "jordan",
            "adidas originals": "adidas",
            "three stripes": "adidas",
            "puma suede": "puma",
            "converse chuck": "converse",
            "chuck taylor": "converse",
            "vans old skool": "vans",
            "reebok classic": "reebok",
            "new balance 990": "new balance",
            "asics gel": "asics",
            "under armour curry": "under armour"
        }
    
    def process_query(self, query: str) -> QueryIntent:
        """Process a natural language query and extract intent"""
        try:
            # Normalize query
            normalized_query = self._normalize_query(query)
            
            # Extract search terms
            search_terms = self._extract_search_terms(normalized_query)
            
            # Extract filters
            filters = self._extract_filters(normalized_query)
            
            # Determine query type
            query_type = self._determine_query_type(normalized_query, filters)
            
            # Extract image path if present
            image_path = self._extract_image_path(normalized_query)
            
            # Determine similarity threshold
            similarity_threshold = self._extract_similarity_threshold(normalized_query)
            
            # Determine limit
            limit = self._extract_limit(normalized_query)
            
            return QueryIntent(
                query_type=query_type,
                search_terms=search_terms,
                filters=filters,
                image_path=image_path,
                similarity_threshold=similarity_threshold,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryIntent(
                query_type="text",
                search_terms=[query],
                filters={}
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'ve": " have",
            "'ll": " will"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        return normalized
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract main search terms from query"""
        # Remove filter words and extract core terms
        filter_words = [
            "with", "that", "have", "are", "is", "in", "of", "for", "and", "or",
            "the", "a", "an", "this", "these", "those", "my", "your", "his", "her"
        ]
        
        words = query.split()
        search_terms = []
        
        for word in words:
            if word not in filter_words and len(word) > 2:
                # Check for synonyms
                expanded_terms = self._expand_term(word)
                search_terms.extend(expanded_terms)
        
        return list(set(search_terms))  # Remove duplicates
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query"""
        filters = {}
        
        # Extract brand
        brand_match = self._match_patterns(query, self.patterns["brand"])
        if brand_match:
            brand = self._normalize_brand(brand_match)
            if brand in METADATA_CATEGORIES["brands"]:
                filters["brand"] = brand
        
        # Extract pattern
        pattern_match = self._match_patterns(query, self.patterns["pattern"])
        if pattern_match:
            pattern = pattern_match.lower()
            if pattern in METADATA_CATEGORIES["patterns"]:
                filters["pattern"] = pattern
        
        # Extract shape
        shape_match = self._match_patterns(query, self.patterns["shape"])
        if shape_match:
            shape = shape_match.lower()
            if shape in METADATA_CATEGORIES["shapes"]:
                filters["shape"] = shape
        
        # Extract size
        size_match = self._match_patterns(query, self.patterns["size"])
        if size_match:
            size = self._normalize_size(size_match)
            if size in METADATA_CATEGORIES["sizes"]:
                filters["size"] = size
        
        # Extract color
        color_match = self._match_patterns(query, self.patterns["color"])
        if color_match:
            filters["color"] = color_match.lower()
        
        # Extract style
        style_match = self._match_patterns(query, self.patterns["style"])
        if style_match:
            filters["style"] = style_match.lower()
        
        return filters
    
    def _match_patterns(self, query: str, patterns: List[str]) -> Optional[str]:
        """Match query against patterns and return first match"""
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _normalize_brand(self, brand: str) -> str:
        """Normalize brand name"""
        brand = brand.lower()
        
        # Check aliases
        for alias, canonical in self.brand_aliases.items():
            if alias in brand:
                return canonical
        
        # Check if brand is in categories
        if brand in METADATA_CATEGORIES["brands"]:
            return brand
        
        return "other"
    
    def _normalize_size(self, size: str) -> str:
        """Normalize size string"""
        size = size.lower()
        
        if size in ["xl", "extra large", "extra-large"]:
            return "extra_large"
        
        if size in METADATA_CATEGORIES["sizes"]:
            return size
        
        return "medium"  # Default size
    
    def _expand_term(self, term: str) -> List[str]:
        """Expand term using synonyms"""
        expanded = [term]
        
        for canonical, synonyms in self.synonyms.items():
            if term in synonyms:
                expanded.append(canonical)
            elif term == canonical:
                expanded.extend(synonyms)
        
        return expanded
    
    def _determine_query_type(self, query: str, filters: Dict[str, Any]) -> str:
        """Determine the type of query"""
        # Check for image-related keywords
        image_keywords = ["similar to", "like this", "matching", "comparable"]
        if any(keyword in query for keyword in image_keywords):
            return "image"
        
        # Check for metadata-only queries
        if not any(word in query for word in ["find", "show", "search", "get", "look for"]):
            if filters:
                return "metadata"
        
        # Check for hybrid queries
        if len(filters) > 2 or any(word in query for word in ["and", "with", "that have"]):
            return "hybrid"
        
        return "text"
    
    def _extract_image_path(self, query: str) -> Optional[str]:
        """Extract image path from query"""
        # Look for file paths or image references
        path_patterns = [
            r"image\s+([^\s]+\.(jpg|jpeg|png|bmp|tiff))",
            r"file\s+([^\s]+\.(jpg|jpeg|png|bmp|tiff))",
            r"([^\s]+\.(jpg|jpeg|png|bmp|tiff))"
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_similarity_threshold(self, query: str) -> float:
        """Extract similarity threshold from query"""
        # Look for similarity keywords
        if "very similar" in query or "exact match" in query:
            return 0.9
        elif "similar" in query or "like" in query:
            return 0.7
        elif "somewhat similar" in query or "related" in query:
            return 0.5
        else:
            return 0.7  # Default threshold
    
    def _extract_limit(self, query: str) -> int:
        """Extract result limit from query"""
        # Look for number words
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "ten": 10, "twenty": 20, "fifty": 50, "hundred": 100
        }
        
        for word, number in number_words.items():
            if word in query:
                return number
        
        # Look for numeric patterns
        number_match = re.search(r"(\d+)\s+(results?|items?|shoes?)", query)
        if number_match:
            return int(number_match.group(1))
        
        return 10  # Default limit
    
    def generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of a query for better search coverage"""
        variations = [query]
        
        # Add synonym variations
        words = query.split()
        for i, word in enumerate(words):
            for canonical, synonyms in self.synonyms.items():
                if word in synonyms:
                    for synonym in synonyms:
                        if synonym != word:
                            new_words = words.copy()
                            new_words[i] = synonym
                            variations.append(" ".join(new_words))
        
        # Add question variations
        if not query.endswith("?"):
            variations.append(query + "?")
        
        # Add command variations
        if not query.startswith(("find", "show", "search", "get")):
            variations.append(f"find {query}")
            variations.append(f"show me {query}")
        
        return list(set(variations))  # Remove duplicates
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate query and return validation result"""
        if not query or len(query.strip()) < 2:
            return False, "Query too short"
        
        if len(query) > 500:
            return False, "Query too long"
        
        # Check for malicious patterns
        malicious_patterns = [
            r"<script.*?>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"exec\s*\("
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially malicious content"
        
        return True, "Valid query"

# Utility functions
def create_query_processor() -> QueryProcessor:
    """Create a query processor instance"""
    return QueryProcessor()

def process_natural_query(query: str) -> QueryIntent:
    """Process a natural language query"""
    processor = create_query_processor()
    return processor.process_query(query)

if __name__ == "__main__":
    # Test query processor
    processor = create_query_processor()
    
    test_queries = [
        "Find red Nike sneakers with zigzag pattern",
        "Show me round shaped large shoes",
        "Search for athletic footwear in medium size",
        "Find shoes similar to this image: test.jpg"
    ]
    
    for query in test_queries:
        intent = processor.process_query(query)
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print("-" * 50)

