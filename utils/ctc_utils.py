# utils/ctc_utils.py
import torch
import itertools
import logging
import unicodedata # For building combined vocab

logger = logging.getLogger(__name__)

# --- CTCDecoder class remains the same ---
class CTCDecoder:
    """Basic CTC Greedy Decoder."""
    def __init__(self, idx_to_char_map, blank_idx=0, output_delimiter=''):
        self.idx_to_char = idx_to_char_map
        self.blank_idx = blank_idx
        self.output_delimiter = output_delimiter

    def __call__(self, logits):
        if logits.ndim != 3: raise ValueError(f"Logits 3D, got {logits.shape}")
        logits = logits.cpu().detach()
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_batch = []
        for pred_seq in predicted_ids:
            # Group same characters
            merged = [k for k, _ in itertools.groupby(pred_seq.tolist())]
            # Remove blanks
            cleaned = [k for k in merged if k != self.blank_idx]
            try:
                decoded_elements = [self.idx_to_char.get(idx, '?') for idx in cleaned]
                decoded_string = self.output_delimiter.join(decoded_elements)
            except Exception as e:
                logger.warning(f"Decode map error: {e}. Sequence: {cleaned}")
                decoded_string = "<DECODE_ERROR>"
            decoded_batch.append(decoded_string)
        return decoded_batch


# --- build_ctc_vocab remains the same ---
def build_ctc_vocab(char_list, add_blank=True, add_unk=True, unk_token='[UNK]'):
    vocab = []
    if add_blank: vocab.append('<blank>')
    # Ensure char_list elements are unique before extending vocab to avoid duplicate UNK if it's in char_list
    processed_char_list = []
    seen_chars = set(vocab) # Chars already in vocab (e.g. <blank>)
    for char_item in char_list:
        if char_item not in seen_chars:
            processed_char_list.append(char_item)
            seen_chars.add(char_item)
    
    # Sort unique characters from input, excluding any already in vocab (like <blank>)
    unique_chars_from_input = sorted(list(set(processed_char_list)))
    vocab.extend(unique_chars_from_input)

    if add_unk:
        if unk_token not in vocab: # Check if UNK is already present from input or initial vocab
            vocab.append(unk_token)
        # No warning needed if it's already there, as it might be intended.
        # else: logger.warning(f"'{unk_token}' was already in char_list or initial vocab.")

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    logger.info(f"Built CTC vocab with {len(vocab)} tokens (Blank: {add_blank}, UNK: {add_unk})")
    return vocab, char_to_idx, idx_to_char


# --- build_combined_vietnamese_charset remains the same ---
def build_combined_vietnamese_charset(include_basic_latin=True, include_digits=True, include_punctuation=True, additional_chars=""):
    """Generates a comprehensive set of combined Vietnamese characters."""
    charset = set()
    # Using NFD for decomposition, then NFC for recomposition to handle stacking order correctly
    vowels_base = "aeiouy" # Base vowels for decomposition reference
    
    # Diacritic Unicode combining characters (NFD form)
    diacritics_map = {
        'acute': "\u0301",      # Sắc
        'grave': "\u0300",      # Huyền
        'hook': "\u0309",       # Hỏi
        'tilde': "\u0303",      # Ngã
        'dot': "\u0323",        # Nặng
        'circumflex': "\u0302", # Â, Ê, Ô
        'breve': "\u0306",      # Ă
        'horn': "\u031b",       # Ơ, Ư
        'stroke': "\u0336" # d with stroke (đ), though 'đ' is often treated as a base. Using for completeness if d + stroke exists.
                           # For 'đ', unicodedata.normalize('NFD', 'đ') gives 'd\u0336'
    }
    
    # Generate vowels with single tone marks
    for v_char_base in vowels_base:
        for case_func in [str.lower, str.upper]:
            bv_cased = case_func(v_char_base)
            charset.add(bv_cased) # Add base vowel itself
            for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
                combined = unicodedata.normalize('NFC', bv_cased + diacritics_map[tone_name])
                charset.add(combined)

    # Generate vowels with modifiers (Ă, Â, Ê, Ô, Ơ, Ư) and their tones
    # Ă (a + breve)
    for case_func in [str.lower, str.upper]:
        a_breve_base = unicodedata.normalize('NFC', case_func('a') + diacritics_map['breve']) # ă, Ă
        charset.add(a_breve_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', a_breve_base + diacritics_map[tone_name]))
            
    # Â (a + circumflex)
    for case_func in [str.lower, str.upper]:
        a_circ_base = unicodedata.normalize('NFC', case_func('a') + diacritics_map['circumflex']) # â, Â
        charset.add(a_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', a_circ_base + diacritics_map[tone_name]))

    # Ê (e + circumflex)
    for case_func in [str.lower, str.upper]:
        e_circ_base = unicodedata.normalize('NFC', case_func('e') + diacritics_map['circumflex']) # ê, Ê
        charset.add(e_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', e_circ_base + diacritics_map[tone_name]))

    # Ô (o + circumflex)
    for case_func in [str.lower, str.upper]:
        o_circ_base = unicodedata.normalize('NFC', case_func('o') + diacritics_map['circumflex']) # ô, Ô
        charset.add(o_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', o_circ_base + diacritics_map[tone_name]))
            
    # Ơ (o + horn)
    for case_func in [str.lower, str.upper]:
        o_horn_base = unicodedata.normalize('NFC', case_func('o') + diacritics_map['horn']) # ơ, Ơ
        charset.add(o_horn_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', o_horn_base + diacritics_map[tone_name]))

    # Ư (u + horn)
    for case_func in [str.lower, str.upper]:
        u_horn_base = unicodedata.normalize('NFC', case_func('u') + diacritics_map['horn']) # ư, Ư
        charset.add(u_horn_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', u_horn_base + diacritics_map[tone_name]))
    
    # Basic Latin Consonants (as defined in your BASE_CHAR_VOCAB_HIER)
    if include_basic_latin:
        # From your BASE_CHAR_VOCAB_HIER, excluding vowels and 'đ', 'Đ'
        consonants_from_base_vocab = "bcdfghklmnpqrstvxzfjw" # Lowercase only for generation
        for c in consonants_from_base_vocab: 
            charset.add(c)
            charset.add(c.upper())

    # Special Vietnamese consonants
    charset.add('đ'); charset.add('Đ') # 'd' itself is added if include_basic_latin is true

    if include_digits:
        for d in "0123456789": charset.add(d)

    if include_punctuation:
        # From your BASE_CHAR_VOCAB_HIER
        punct = " .,-_()[]{}:;\"'/\\?!@#$%^&*+=<>|" 
        for p in punct: charset.add(p)
        charset.add(' ') # Ensure space is included if not covered by punct string

    for char in additional_chars: charset.add(char)
    
    # Remove empty string if it accidentally got added
    charset.discard("")
    
    return sorted(list(charset))


# --- NEW: Vietnamese Character Decomposition Utilities ---

# Mapping of combining diacritics to their canonical names for your DIACRITIC_VOCAB_HIER
# This helps standardize the diacritic names obtained from unicodedata.
VIETNAMESE_DIACRITIC_UNICODE_TO_NAME = {
    '\u0301': 'acute',      # Sắc
    '\u0300': 'grave',      # Huyền
    '\u0309': 'hook',       # Hỏi (hook above)
    '\u0303': 'tilde',      # Ngã
    '\u0323': 'dot',        # Nặng (dot below)
    '\u0302': 'circumflex', # Â, Ê, Ô
    '\u0306': 'breve',      # Ă
    '\u031b': 'horn',       # Ơ, Ư
    '\u0336': 'stroke'      # For đ
}
# Inverse map for convenience if needed later
VIETNAMESE_DIACRITIC_NAME_TO_UNICODE = {v: k for k, v in VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.items()}

# Define which diacritics are tones vs. modifiers
VIETNAMESE_TONE_NAMES = {'acute', 'grave', 'hook', 'tilde', 'dot'}
VIETNAMESE_MODIFIER_NAMES = {'circumflex', 'breve', 'horn', 'stroke'}

def decompose_vietnamese_char(char_in):
    """
    Decomposes a single Vietnamese character into its base, modifier(s), and tone.
    Returns: (base_char_nfc, diacritic_name_combined, tone_name_only)
    Example: 'ệ' -> ('e', 'dot_circumflex', 'dot')
             'đ' -> ('d', 'stroke', None)
             'á' -> ('a', 'acute', 'acute')
             'a' -> ('a', 'no_diacritic', None)
    """
    if not char_in or len(char_in) != 1:
        return (char_in, 'no_diacritic', None) # Or handle error

    # Normalize to NFD to separate base character and combining diacritics
    nfd_char = unicodedata.normalize('NFD', char_in)
    
    base_char = ""
    combining_diacritics_unicode = []

    for ch_part in nfd_char:
        if unicodedata.category(ch_part) != 'Mn': # Mn = Mark, Nonspacing (combining characters)
            base_char += ch_part
        else:
            combining_diacritics_unicode.append(ch_part)

    if not base_char: # Should not happen for single chars but as a safeguard
        return (char_in, 'no_diacritic', None)

    # Normalize base_char back to NFC in case it's a multi-part base (rare for Vietnamese alphabet)
    base_char_nfc = unicodedata.normalize('NFC', base_char)

    if not combining_diacritics_unicode:
        return (base_char_nfc, 'no_diacritic', None)

    # Identify tone marks and modifiers from the Unicode list
    tone_unicode_parts = []
    modifier_unicode_parts = []

    for diac_unicode in combining_diacritics_unicode:
        diac_name_from_map = VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(diac_unicode)
        if diac_name_from_map:
            if diac_name_from_map in VIETNAMESE_TONE_NAMES:
                tone_unicode_parts.append(diac_unicode)
            elif diac_name_from_map in VIETNAMESE_MODIFIER_NAMES:
                modifier_unicode_parts.append(diac_unicode)
        # Else: unknown combining mark, ignore for this purpose

    # Determine the names
    tone_name_only = None
    if tone_unicode_parts:
        # For simplicity, if multiple tones (shouldn't happen in valid NFC), take the first recognized one.
        # Or, sort them by Unicode value and take the one that typically applies last if needed.
        # Here, we assume valid Vietnamese, so at most one tone mark type.
        tone_name_only = VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(tone_unicode_parts[0])
        
    modifier_names_list = sorted([VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(m) for m in modifier_unicode_parts if VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(m)])

    # Construct the combined diacritic name for your DIACRITIC_VOCAB_HIER
    # Order: modifier(s) then tone. Sort modifiers alphabetically for consistency.
    # Example: circumflex_dot, breve_acute
    
    diacritic_name_parts = []
    diacritic_name_parts.extend(modifier_names_list) # Modifiers first
    if tone_name_only:
        diacritic_name_parts.append(tone_name_only)
    
    if not diacritic_name_parts:
        diacritic_name_combined = 'no_diacritic'
    else:
        # Check against your DIACRITIC_VOCAB_HIER structure.
        # If your vocab has "circumflex_dot", join with "_".
        # If it has separate "circumflex", "dot", then this logic needs adjustment
        # or your DIACRITIC_VOCAB_HIER must contain these combined forms.
        # The current DIACRITIC_VOCAB_HIER implies combined names like 'circumflex_grave'.
        
        # A more robust way if multiple modifiers can co-exist (e.g. theoretical 'breve_circumflex_acute'):
        # Sort modifier_names_list alphabetically to ensure consistent naming like "breve_circumflex" vs "circumflex_breve"
        # For Vietnamese, typically one modifier (breve, horn, circumflex) + one tone. 'stroke' is standalone.
        
        if len(modifier_names_list) > 1: # e.g. if a char could have 'breve' and 'horn' (not typical Vietnamese)
            # Prioritize based on typical Vietnamese combinations or alphabetical sort for consistency
            # For now, let's assume at most one of (breve, horn, circumflex) or 'stroke' alone.
            logger.debug(f"Multiple modifiers found for '{char_in}': {modifier_names_list}. This might need specific handling.")

        # Construct the combined name as per DIACRITIC_VOCAB_HIER
        final_diac_name_parts = []
        if 'stroke' in modifier_names_list: # 'stroke' is usually exclusive
            final_diac_name_parts = ['stroke']
        else:
            # Pick one primary modifier if multiple non-stroke present (should be rare for valid Viet chars)
            primary_modifier = None
            if 'circumflex' in modifier_names_list: primary_modifier = 'circumflex'
            elif 'breve' in modifier_names_list: primary_modifier = 'breve'
            elif 'horn' in modifier_names_list: primary_modifier = 'horn'
            
            if primary_modifier:
                final_diac_name_parts.append(primary_modifier)
            if tone_name_only:
                final_diac_name_parts.append(tone_name_only)
        
        if not final_diac_name_parts: # Only a tone was present, no modifier
             diacritic_name_combined = tone_name_only if tone_name_only else 'no_diacritic'
        else:
            diacritic_name_combined = "_".join(final_diac_name_parts)

        if not diacritic_name_combined: # Fallback if somehow empty
            diacritic_name_combined = 'no_diacritic'


    return base_char_nfc, diacritic_name_combined, tone_name_only


def get_char_type(char_in):
    """
    Determines if a single character is a Vietnamese vowel, consonant, digit, or symbol.
    Focuses on the base form of the character.
    """
    if not char_in or len(char_in) != 1:
        return "symbol" # Or "unknown"

    # Decompose to get the base character for type checking
    # We only care about the base form for this classification
    nfd_char = unicodedata.normalize('NFD', char_in)
    base_form = ""
    for ch_part in nfd_char:
        if unicodedata.category(ch_part) != 'Mn': # Not a combining mark
            base_form += ch_part
    base_form_nfc = unicodedata.normalize('NFC', base_form) # NFC of the base
    
    # Use lowercase for matching against known sets
    base_lower = base_form_nfc.lower()

    vietnamese_vowels_base_lower = {'a', 'e', 'i', 'o', 'u', 'y', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư'}
    # Consonants, including 'đ'. 'd' is base, 'đ' formed by d+stroke.
    # If 'đ' is input, its base_form_nfc will be 'đ'.
    vietnamese_consonants_base_lower = {'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'f', 'j', 'w', 'z'}
    digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    if base_lower in vietnamese_vowels_base_lower:
        return "vowel"
    elif base_lower in vietnamese_consonants_base_lower:
        return "consonant"
    elif base_lower in digits:
        return "digit"
    else:
        return "symbol"

# --- Example Usage (for testing the new functions) ---
if __name__ == "__main__":
    test_chars = ['a', 'à', 'ă', 'ằ', 'â', 'ấ', 'đ', 'd', 'ệ', 'k', ' ', '1', '!', 'ườ']
    print("--- Decomposing Vietnamese Characters ---")
    for char_to_test in test_chars:
        base, diac_name, tone_name = decompose_vietnamese_char(char_to_test)
        char_type = get_char_type(char_to_test)
        print(f"Char: '{char_to_test}' (Type: {char_type}) -> Base: '{base}', CombinedDiacName: '{diac_name}', ToneOnlyName: '{tone_name}'")

    print("\n--- Testing build_combined_vietnamese_charset ---")
    full_charset = build_combined_vietnamese_charset()
    print(f"Generated {len(full_charset)} combined Vietnamese characters. First 10: {full_charset[:10]}, Last 10: {full_charset[-10:]}")
    
    # Verify a few specific complex characters are present
    expected_chars = ['ệ', 'ườ', 'ặ', 'ẫ']
    for ec in expected_chars:
        if ec in full_charset:
            print(f"'{ec}' found in generated charset.")
        else:
            print(f"ERROR: '{ec}' NOT found in generated charset.")

    print("\n--- Testing build_ctc_vocab ---")
    sample_chars = ['a', 'b', 'c', 'a']
    vocab, c2i, i2c = build_ctc_vocab(sample_chars)
    print(f"Sample vocab: {vocab}")
    print(f"Char to Idx: {c2i}")