import json
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


class TriggerWordFilter:
    def __init__(self, json_file=None, min_freq=5, max_freq=1000):
        """
        Initialize the filter
        :param json_file: Path to the JSON file containing trigger words and their frequencies
        :param min_freq: Minimum frequency threshold (words below this value will be filtered out)
        :param max_freq: Maximum frequency threshold (high-frequency generic words above this value will be filtered out)
        """
        # Basic stopwords
        self.base_stopwords = set(stopwords.words('english'))

        # Extended stopwords - added based on event extraction characteristics
        self.extended_stopwords = {
            'be', 'have', 'do'
        }

        # Domain-independent generic high-frequency words
        self.generic_words = {
            'thing', 'something', 'anything', 'everything', 'nothing',
            'way', 'case', 'point', 'example', 'instance', 'kind', 'sort'
        }

        # Allowed POS tags (Penn Treebank tag set)
        self.allowed_pos_tags = {
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'JJ', 'JJR', 'JJS'  # Adjectives (some events may involve them)
        }

        # Filtering parameters
        self.min_word_length = 3
        self.min_freq = min_freq
        self.max_freq = max_freq

        # Load trigger word data
        self.trigger_data = self._load_trigger_data(json_file)
        self.trigger_words = list(self.trigger_data.keys()) if self.trigger_data else []

    def _load_trigger_data(self, json_file):
        """Load trigger word data from a JSON file"""
        if not json_file:
            return self._load_example_data()

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all values are integers (frequencies)
                return {k: int(v) for k, v in data.items()}
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return self._load_example_data()

    def _load_example_data(self):
        """Example data (used when no JSON file is provided)"""
        example_triggers = [
            "Ate", "Aid", "Add", "Act", "ATM", "-/-", "'ve", "up", "to", "ta", "on", "of", "it", "is", "in", "if",
            "by", "be", "at", "as", "an", "ad", "UE", "PM", "KD", "It", "In", "IT", "ID", "GA", "CD", "'s",
        ]
        return {word: 10 for word in example_triggers}  # Assign default frequency 10 to example words

    def _is_valid_pos(self, word):
        """Check whether the POS tag is appropriate"""
        tagged = pos_tag([word])
        pos = tagged[0][1] if tagged else None
        return pos in self.allowed_pos_tags

    def _is_event_indicative(self, word):
        """Determine whether the word is event-indicative based on semantics"""
        synsets = wordnet.synsets(word)
        for synset in synsets:
            # Check whether it has action- or event-related senses
            if synset.pos() in ['v', 'n'] and any(
                    lemma.name() for lemma in synset.lemmas()
                    if 'event' in lemma.name() or 'action' in lemma.name()
                       or 'process' in lemma.name()
            ):
                return True
        return False

    def _is_valid_phrase(self, phrase):
        """Check whether a phrase is valid"""
        words = word_tokenize(phrase.lower())
        # The phrase must contain at least one valid trigger word
        return any(
            word not in self.base_stopwords and
            word not in self.extended_stopwords and
            word not in self.generic_words
            for word in words
        )

    def filter_by_frequency(self, trigger_data):
        """
        Filter trigger words by frequency
        :param trigger_data: Dictionary {trigger_word: frequency}
        :return: Filtered dictionary
        """
        return {
            k: v for k, v in trigger_data.items()
            if self.min_freq <= v <= self.max_freq
        }

    def filter_triggers(self):
        """Main filtering function"""
        if not self.trigger_data:
            return []

        # First apply frequency filtering
        freq_filtered = self.filter_by_frequency(self.trigger_data)

        filtered_triggers = []

        for trigger, freq in freq_filtered.items():
            # Skip empty strings
            if not trigger.strip():
                continue

            # Handle phrase-type triggers
            if ' ' in trigger or '-' in trigger:
                if self._is_valid_phrase(trigger):
                    filtered_triggers.append((trigger, freq))
                continue

            # Convert to lowercase for processing (preserve original case)
            lower_trigger = trigger.lower()

            # Basic filtering
            if (len(trigger) < self.min_word_length or
                    lower_trigger in self.base_stopwords or
                    lower_trigger in self.extended_stopwords or
                    lower_trigger in self.generic_words or
                    not any(c.isalpha() for c in trigger)):  # Must contain at least one alphabetic character
                continue

            # POS filtering
            if not self._is_valid_pos(trigger):
                continue

            filtered_triggers.append((trigger, freq))

        return filtered_triggers

    def analyze_triggers(self):
        """Analyze statistical properties of trigger words"""
        if not self.trigger_data:
            print("No trigger data to analyze")
            return

        word_counts = Counter()
        pos_counts = Counter()
        freq_dist = []

        for trigger, freq in self.trigger_data.items():
            if ' ' in trigger or '-' in trigger:
                continue

            word_counts[trigger.lower()] += freq
            freq_dist.append(freq)
            tagged = pos_tag([trigger])
            if tagged:
                pos_counts[tagged[0][1]] += 1

        print("\nMost common words (by frequency):")
        print(word_counts.most_common(20))

        print("\nPOS tag distribution:")
        print(pos_counts.most_common())

        print("\nFrequency statistics:")
        print(f"Min frequency: {min(freq_dist)}")
        print(f"Max frequency: {max(freq_dist)}")
        print(f"Average frequency: {sum(freq_dist) / len(freq_dist):.2f}")


# Usage example
if __name__ == "__main__":
    filter = TriggerWordFilter(json_file="trigger_frequencies.json", min_freq=2, max_freq=100000000)
    with open("trigger_frequencies.json", "r", encoding="utf-8") as f:
        trigger_data = json.load(f)
        trigger_all = set(trigger_data.keys())

    print("\nOriginal trigger count:", len(filter.trigger_data))

    filter.analyze_triggers()

    filtered = filter.filter_triggers()
    filtered_dict = {}
    filter_keys = set()
    for trigger, freq in filtered:
        filtered_dict[trigger] = freq
        filter_keys.add(trigger)
    print("\nFiltered trigger count:", len(filtered))
    with open("trigger_filter.json", "w") as f:
        json.dump(filtered_dict, f, ensure_ascii=False, indent=4)

    bad_trigger = list(trigger_all - filter_keys)
    with open("bad_trigger.json", "w") as f:
        json.dump(bad_trigger, f, ensure_ascii=False, indent=4)
